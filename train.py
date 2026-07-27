import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "spam.csv"
MODEL_PATH = BASE_DIR / "model" / "spam_model.pkl"
VECTORIZER_PATH = BASE_DIR / "model" / "tfidf_vectorizer.pkl"


def train_model():
    df = pd.read_csv(DATASET_PATH, encoding="latin-1", usecols=[0, 1])
    df.columns = ["label", "message"]
    df = df.dropna(subset=["message"]).copy()
    df["label"] = df["label"].astype(str).str.lower().map({"ham": 0, "spam": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        df["message"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LinearSVC()
    model.fit(X_train_tfidf, y_train)

    predictions = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, predictions)

    print(f"\nAccuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report")
    print(classification_report(y_test, predictions, target_names=["Ham", "Spam"]))
    print("Confusion Matrix")
    print(confusion_matrix(y_test, predictions))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(model, model_file)

    with VECTORIZER_PATH.open("wb") as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)

    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Vectorizer saved to {VECTORIZER_PATH}")

    return accuracy


if __name__ == "__main__":
    train_model()
