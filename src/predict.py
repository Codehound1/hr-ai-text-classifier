from pathlib import Path
import argparse, joblib
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "hr_text_classifier.joblib"
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", type=str)
    args = parser.parse_args()
    pipe = joblib.load(MODEL_PATH)
    probs = pipe.predict_proba([args.text])[0]
    classes = pipe.named_steps["classifier"].classes_
    pred = pipe.predict([args.text])[0]
    print(f"Input text: {args.text}")
    print(f"Predicted category: {pred}")
    print(f"Confidence: {max(probs):.2f}")
    print("\nProbabilities:")
    for c,p in sorted(zip(classes, probs), key=lambda x: x[1], reverse=True):
        print(f"- {c}: {p:.2f}")
if __name__ == "__main__":
    main()
