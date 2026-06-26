import torch
import pickle
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.bert.modeling_bert import BertForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_dir="models"):
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model     = BertForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    with open(f"{model_dir}/label_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    return model, tokenizer, encoder


def predict(previous_comment, reply_comment, model, tokenizer, encoder):
    # TRUE context-aware: sentence pair [CLS] prev [SEP] reply [SEP]
    inputs = tokenizer(
        previous_comment,
        reply_comment,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits     = outputs.logits
    probs      = torch.softmax(logits, dim=1)
    confidence = probs.max().item()
    prediction = torch.argmax(probs, dim=1).item()
    label      = encoder.inverse_transform([prediction])[0]
    return label, round(confidence * 100, 2)


if __name__ == "__main__":
    print("\nLoading model...\n")
    model, tokenizer, encoder = load_model("models")
    print("Model loaded successfully!\n")

    while True:
        print("-" * 40)
        previous_comment = input("Enter Previous Comment (or 'quit' to exit): ").strip()
        if previous_comment.lower() == "quit":
            break
        reply_comment = input("Enter Reply Comment: ").strip()
        label, confidence = predict(previous_comment, reply_comment, model, tokenizer, encoder)
        print(f"\nPrediction : {label}")
        print(f"Confidence : {confidence} %")