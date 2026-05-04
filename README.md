# HR AI Text Classification Project v4

Expanded data + slightly improved model configuration.

## Run

```bash
python -m pip install -r requirements.txt
python src/train_model.py
python src/train_resume_classifier.py
python -m streamlit run src/app.py
```

## CLI Tests

```bash
python src/predict.py "I did not receive my direct deposit today"
python src/predict_resume.py "Epic Clarity analyst with SQL, Caboodle, Tableau, and healthcare compliance reporting experience"
```

This is still a proof of concept. Real HR use requires anonymized labeled data, bias testing, human review, privacy controls, and compliance review.
