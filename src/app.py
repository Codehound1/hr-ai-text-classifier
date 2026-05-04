from pathlib import Path
import joblib, streamlit as st
ROOT = Path(__file__).resolve().parents[1]
HR = ROOT/"models/hr_text_classifier.joblib"
RES = ROOT/"models/resume_classifier.joblib"
st.set_page_config(page_title="HR AI Text Classification Demo", page_icon="🤖")
st.title("🤖 HR AI Text Classification Demo")
st.info("Proof of concept. Predictions should support human review, not replace it.")

def load(p): return joblib.load(p) if p.exists() else None
def predict(model, text, label):
    probs = model.predict_proba([text])[0]
    classes = model.named_steps["classifier"].classes_
    pred = model.predict([text])[0]
    st.success(f"{label}: **{pred}**")
    st.write(f"Confidence: **{max(probs):.2f}**")
    if max(probs) < 0.45: st.warning("Low confidence: route for human review.")
    for c,p in sorted(zip(classes, probs), key=lambda x:x[1], reverse=True):
        st.write(f"**{c}**: {p:.2f}")
        st.progress(float(p))

tab1, tab2, tab3 = st.tabs(["HR Ticket Classifier","Resume Classifier","About"])
with tab1:
    st.header("HR Ticket Routing")
    text = st.text_area("HR ticket text", "I did not receive my direct deposit today.")
    m = load(HR)
    if not m: st.warning("Run: python src/train_model.py")
    elif st.button("Classify HR Ticket"): predict(m, text, "Predicted HR category")
with tab2:
    st.header("Resume Classification")
    text = st.text_area("Resume text", "Epic Clarity analyst with SQL, Caboodle, Tableau, and healthcare compliance reporting experience.")
    m = load(RES)
    if not m: st.warning("Run: python src/train_resume_classifier.py")
    elif st.button("Classify Resume"): predict(m, text, "Predicted job family")
with tab3:
    st.markdown("""
### v4 Improvements
- More sample data for both models
- Better model settings: `sublinear_tf=True` and `class_weight="balanced"`
- Low-confidence warning for human review
""")
