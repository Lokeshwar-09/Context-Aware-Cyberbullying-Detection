import os
import torch
import numpy as np
import pickle
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
logging.getLogger("accelerate").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.utils.class_weight import compute_class_weight

from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from transformers import Trainer, TrainingArguments

from preprocess import load_dataset


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("GPU Name    :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(BASE_DIR, "data", "improved_conversation_bullying_dataset.csv")
    print("Loading dataset from:", dataset_path)

    df, encoder = load_dataset(dataset_path)

    # Shuffle to remove any ordering pattern
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print("Dataset shuffled.")

    text_a_list = df["text_a"].tolist()
    text_b_list = df["text_b"].tolist()
    labels      = df["label"].tolist()

    print("\nDataset size:", len(labels))

    (
        train_a, temp_a,
        train_b, temp_b,
        train_labels, temp_labels
    ) = train_test_split(
        text_a_list, text_b_list, labels,
        test_size=0.30, random_state=42, stratify=labels
    )

    (
        val_a, test_a,
        val_b, test_b,
        val_labels, test_labels
    ) = train_test_split(
        temp_a, temp_b, temp_labels,
        test_size=0.50, random_state=42, stratify=temp_labels
    )

    print("\nTrain      :", len(train_labels))
    print("Validation :", len(val_labels))
    print("Test       :", len(test_labels))

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
    print("\nClass weights:", class_weights_tensor)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def tokenize(a_list, b_list):
        # TRUE sentence-pair encoding [CLS] prev [SEP] reply [SEP]
        return tokenizer(
            a_list,
            b_list,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_overflowing_tokens=False
        )

    print("\nTokenizing...")
    train_encodings = tokenize(train_a, train_b)
    val_encodings   = tokenize(val_a,   val_b)
    test_encodings  = tokenize(test_a,  test_b)
    print("Tokenization done.")

    class CyberDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels    = labels
        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item
        def __len__(self):
            return len(self.labels)

    train_dataset = CyberDataset(train_encodings, train_labels)
    val_dataset   = CyberDataset(val_encodings,   val_labels)
    test_dataset  = CyberDataset(test_encodings,  test_labels)

    class WeightedTrainer(Trainer):
        def __init__(self, class_weights, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights.to(device)
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels  = inputs.pop("labels")
            outputs = model(**inputs)
            logits  = outputs.logits
            loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights)
            loss    = loss_fn(logits, labels)
            return (loss, outputs) if return_outputs else loss

    num_classes = len(encoder.classes_)
    print("\nNumber of classes:", num_classes)

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=num_classes,
        hidden_dropout_prob=0.3,
        attention_probs_dropout_prob=0.3
    )
    model.to(device)

    training_args = TrainingArguments(
        output_dir="./models",
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        logging_dir="./logs",
        logging_steps=200,
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=True,
        dataloader_num_workers=0,
        save_steps=5000,
        report_to="none"
    )

    trainer = WeightedTrainer(
        class_weights=class_weights_tensor,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    print("\nTraining BERT model...\n")
    trainer.train()

    def print_metrics(true_labels, pred_labels, split_name):
        acc  = accuracy_score(true_labels, pred_labels)
        prec = precision_score(true_labels, pred_labels, average="weighted")
        rec  = recall_score(true_labels, pred_labels, average="weighted")
        f1   = f1_score(true_labels, pred_labels, average="weighted")
        cm   = confusion_matrix(true_labels, pred_labels)
        print(f"\n🧪 {split_name} METRICS")
        print(f"Accuracy  : {acc}")
        print(f"Precision : {prec}")
        print(f"Recall    : {rec}")
        print(f"F1 Score  : {f1}")
        print(f"\n📊 Confusion Matrix:")
        print(cm)
        print(f"\n📄 Detailed Classification Report:")
        print(classification_report(true_labels, pred_labels, target_names=encoder.classes_))

    print("\n" + "=" * 50)
    val_preds = trainer.predict(val_dataset)
    val_pred  = np.argmax(val_preds.predictions, axis=1)
    print_metrics(val_labels, val_pred, "VALIDATION")

    print("\n" + "=" * 50)
    test_preds = trainer.predict(test_dataset)
    test_pred  = np.argmax(test_preds.predictions, axis=1)
    print_metrics(test_labels, test_pred, "TEST")

    os.makedirs("models", exist_ok=True)
    model.save_pretrained("models")
    tokenizer.save_pretrained("models")
    with open("models/label_encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)

    print("\n✅ Model trained and saved successfully.")