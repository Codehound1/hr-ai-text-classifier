from pathlib import Path
import pandas as pd, joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "hr_tickets_sample.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "hr_text_classifier.joblib"

def main():
    df = pd.read_csv(DATA_PATH)
    X, y = df["text"], df["category"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1,2), sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"))
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    print("HR Ticket Classifier Evaluation")
    print("-"*35)
    print(f"Accuracy: {accuracy_score(y_test, pred):.2f}\n")
    print(classification_report(y_test, pred))
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    print(f"Saved model to: {MODEL_PATH}")
if __name__ == "__main__":
    main()
