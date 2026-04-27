import pickle
import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

DATASET_DIR = Path("dataset")

def load_dataset():
    news_path = DATASET_DIR / "news.csv"
    true_path = DATASET_DIR / "True.csv"
    fake_path = DATASET_DIR / "Fake.csv"

    if news_path.exists():
        df = pd.read_csv(news_path)
        if {"title", "text", "label"}.issubset(df.columns):
            return df

    if true_path.exists() and fake_path.exists():
        true_df = pd.read_csv(true_path)
        fake_df = pd.read_csv(fake_path)

        true_df["label"] = 1
        fake_df["label"] = 0

        return pd.concat([true_df, fake_df], ignore_index=True)

    raise FileNotFoundError("Dataset not found. Add news.csv or True.csv and Fake.csv inside dataset folder.")

df = load_dataset()
df = df.dropna(subset=["text"])
df["title"] = df.get("title", "").fillna("")
df["text"] = df["text"].fillna("")
df["content"] = df["title"] + " " + df["text"]

X_train, X_test, y_train, y_test = train_test_split(
    df["content"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.85,
    ngram_range=(1, 2),
    min_df=2 if len(df) > 1000 else 1
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = PassiveAggressiveClassifier(
    max_iter=1000,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train_vec, y_train)

predictions = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

metrics = {
    "accuracy": round(float(accuracy), 4),
    "model": "PassiveAggressiveClassifier",
    "vectorizer": "TfidfVectorizer",
    "rows": int(len(df))
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Saved model.pkl, vectorizer.pkl, and metrics.json")
