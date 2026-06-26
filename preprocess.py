import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
from collections import Counter


def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    return text


def load_dataset(path):

    try:
        df = pd.read_csv(path, encoding="utf-8")
    except:
        df = pd.read_csv(path, encoding="latin1")

    print("Raw dataset size:", len(df))

    df = df.dropna()
    df = df.drop_duplicates()

    print("After cleaning:", len(df))

    df["previous_comment"] = df["previous_comment"].astype(str)
    df["reply_comment"]    = df["reply_comment"].astype(str)

    df["previous_comment"] = df["previous_comment"].apply(clean_text)
    df["reply_comment"]    = df["reply_comment"].apply(clean_text)

    # TRUE CONTEXT-AWARE: pass both as sentence pair
    # previous_comment now crosses all 4 label types — no leakage
    df["text_a"] = df["previous_comment"]
    df["text_b"] = df["reply_comment"]

    df["text"] = df["previous_comment"] + " [SEP] " + df["reply_comment"]

    encoder = LabelEncoder()
    df["label"] = encoder.fit_transform(df["bullying_type"])

    print("\nClass distribution:")
    counts = Counter(df["bullying_type"])
    for cls, cnt in sorted(counts.items()):
        print(f"  {cls}: {cnt} ({cnt/len(df)*100:.1f}%)")

    print("\nLabel mapping:")
    for i, cls in enumerate(encoder.classes_):
        print(f"  {i} -> {cls}")

    df = df[["text_a", "text_b", "text", "label"]]

    return df, encoder