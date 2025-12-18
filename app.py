import streamlit as st
import joblib, numpy as np, csv
from datetime import datetime
from analysis_utils import arabic_emergency_detect

# ----------------------------
# LOAD ARTIFACTS
# ----------------------------
model = joblib.load("medical_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
le_drug = joblib.load("drug_encoder.pkl")
le_cond = joblib.load("condition_encoder.pkl")
le_gender = joblib.load("gender_encoder.pkl")
le_target = joblib.load("decision_encoder.pkl")
scaler = joblib.load("numeric_scaler.pkl")

# ----------------------------
# STREAMLIT UI
# ----------------------------
st.set_page_config(page_title="Medical AI", layout="centered")
lang = st.selectbox("Language / اللغة", ["English","العربية"])
st.title("Medical AI Decision Support" if lang=="English" else "نظام دعم القرار الطبي")

# Inputs
drug = st.selectbox("Drug", le_drug.classes_)
condition = st.selectbox("Condition", le_cond.classes_)
side_effects = st.text_area("Side Effects" if lang=="English" else "الأعراض")

age = st.number_input("Age", 1, 120, 30)
weight = st.number_input("Weight (kg)", 20.0, 200.0, 70.0)
gender = st.selectbox("Gender", ["Male","Female"])
smoker = st.selectbox("Smoker", ["No","Yes"])
chronic = st.selectbox("Chronic Disease", ["None","Diabetes","Hypertension","Asthma"])

# ----------------------------
# EMERGENCY CHECK
# ----------------------------
if lang=="العربية" and arabic_emergency_detect(side_effects):
    st.error("🚨 حالة طارئة – توجه للطوارئ فورًا")
    st.stop()

# ----------------------------
# PREDICTION
# ----------------------------
if st.button("Get Recommendation"):

    drug_enc = le_drug.transform([drug])[0]
    cond_enc = le_cond.transform([condition])[0]
    gender_enc = le_gender.transform([gender])[0]
    smoker_val = 1 if smoker=="Yes" else 0
    has_chronic = 0 if chronic=="None" else 1

    text_vec = vectorizer.transform([side_effects])
    X_num = scaler.transform([[age, weight]])

    X = np.hstack([
        text_vec.toarray(),
        [[drug_enc, cond_enc, gender_enc, smoker_val, has_chronic]],
        X_num
    ])

    probs = model.predict_proba(X)[0]
    idx = np.argmax(probs)
    confidence = probs[idx]
    decision = le_target.inverse_transform([idx])[0]

    # SAFETY RULES
    if confidence < 0.5 or age>65 or has_chronic==1:
        decision = "See_Doctor"

    st.success(f"Decision: {decision}")
    st.write(f"Confidence: {confidence:.2f}")

    # Feedback
    feedback = st.radio("Feedback", ["Correct","Incorrect","Not Sure"])
    correct = None
    if feedback=="Incorrect":
        correct = st.selectbox("Correct decision", le_target.classes_)

    if st.button("Submit Feedback"):
        with open("feedback_log.csv","a",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow([
                datetime.now(),drug,condition,side_effects,
                decision,correct,confidence,lang
            ])
        st.success("Saved ✔")
