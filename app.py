import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# =============================
# تحميل الموديل والـ Artifacts
# =============================
model = joblib.load("medical_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
le_drug = joblib.load("drug_encoder.pkl")
le_gender = joblib.load("gender_encoder.pkl")
le_target = joblib.load("decision_encoder.pkl")
scaler = joblib.load("numeric_scaler.pkl")

# =============================
# إعداد الصفحة
# =============================
st.set_page_config(page_title="Medical AI CDS", layout="centered")

# =============================
# اختيار اللغة
# =============================
lang_choice = st.selectbox("Language / اللغة", ["English", "العربية"], key="lang_select")
lang = "en" if lang_choice == "English" else "ar"

# =============================
# القاموس النصي
# =============================
t = {
    "title": "Medical AI Decision Support" if lang=="en" else "نظام دعم القرار الطبي",
    "welcome": "Welcome to Medical AI CDS" if lang=="en" else "مرحبًا بك في النظام الطبي الذكي",
    "description": "This system helps you with medicine reminders." if lang=="en" else "هذا النظام يساعدك في تذكيرك بالأدوية.",
    "caption": "" if lang=="en" else "",
    "warning": "⚠️ This system does NOT replace professional medical advice." if lang=="en" else "⚠️ هذا النظام لا يغني عن الاستشارة الطبية المتخصصة.",
    "patient_info": "Patient Information" if lang=="en" else "معلومات المريض",
    "prediction_feedback": "Prediction & Feedback" if lang=="en" else "التنبؤ والتغذية الراجعة",
    "history_analysis": "History & Analysis" if lang=="en" else "السجل والتحليل",
    "age": "Age" if lang=="en" else "العمر",
    "gender": "Gender" if lang=="en" else "الجنس",
    "weight": "Weight (kg)" if lang=="en" else "الوزن (كجم)",
    "smoker": "Smoker" if lang=="en" else "مدخن",
    "chronic_diseases": "Chronic Diseases" if lang=="en" else "الأمراض المزمنة",
    "drug": "Drug" if lang=="en" else "الدواء",
    "new_drug": "Add new Drug" if lang=="en" else "أضف دواء جديد",
    "condition": "Condition" if lang=="en" else "الحالة",
    "side_effects": "Side Effects" if lang=="en" else "الأعراض",
    "get_recommendation": "Get Recommendation" if lang=="en" else "الحصول على التوصية",
    "emergency": "🚨 EMERGENCY – Seek immediate medical help!" if lang=="en" else "🚨 حالة طارئة – توجه للطوارئ فورًا",
    "risk_score": "Patient Risk Score" if lang=="en" else "درجة مخاطر المريض",
    "drug_warning": "This drug was not in the original model" if lang=="en" else "هذا الدواء غير موجود في الموديل الأصلي",
    "chronic_warning": "Some chronic diseases are unknown" if lang=="en" else "بعض الأمراض المزمنة غير معروفة",
    "prediction_probs": "Prediction Probabilities" if lang=="en" else "احتمالات التنبؤ",
    "final_recommendation": "Final Recommendation" if lang=="en" else "التوصية النهائية",
    "continue": "✅ Continue medication and monitor." if lang=="en" else "✅ تابع تناول الدواء مع المراقبة",
    "see_doctor": "⚠️ Consult a doctor as soon as possible." if lang=="en" else "⚠️ استشر الطبيب في أقرب وقت ممكن",
    "emergency_msg": "🚨 Seek emergency medical attention." if lang=="en" else "🚨 توجه للطوارئ فورًا",
    "feedback": "Feedback" if lang=="en" else "التغذية الراجعة",
    "correct": "Correct" if lang=="en" else "صحيح",
    "incorrect": "Incorrect" if lang=="en" else "خاطئ",
    "not_sure": "Not Sure" if lang=="en" else "لست متأكد",
    "correct_decision": "Correct Decision" if lang=="en" else "القرار الصحيح",
    "feedback_saved": "Feedback saved successfully ✔" if lang=="en" else "تم حفظ التغذية الراجعة ✔"
}

# =============================
# العنوان والوصف
# =============================
st.title(t["title"])
st.write(t["description"])
st.caption(t["caption"])
st.warning(t["warning"])

# =============================
# Session State
# =============================
if 'history' not in st.session_state:
    st.session_state.history = []

# =============================
# Tabs
# =============================
tab1, tab2, tab3 = st.tabs([t["patient_info"], t["prediction_feedback"], t["history_analysis"]])

# =============================
# Tab 1: Patient Info
# =============================
with tab1:
    with st.form("patient_form"):
        age = st.number_input(t["age"], 0, 120, 30, key="age")
        gender = st.selectbox(t["gender"], ["Male","Female","Other"] if lang=="en" else ["ذكر","أنثى","آخر"], key="gender")
        weight = st.number_input(t["weight"], 1.0, 300.0, 70.0, key="weight")
        smoker = st.selectbox(t["smoker"], ["No","Yes"] if lang=="en" else ["لا","نعم"], key="smoker")
        chronic_diseases = st.multiselect(
            t["chronic_diseases"],
            ["Diabetes","Hypertension","Heart Disease","Kidney Disease","None"] if lang=="en" else ["سكري","ضغط الدم","أمراض القلب","أمراض الكلى","لا يوجد"],
            key="chronic"
        )
        drug = st.selectbox(t["drug"], le_drug.classes_, key="drug")
        new_drug = st.text_input(t["new_drug"], key="new_drug")
        condition = st.selectbox(t["condition"], le_target.classes_, key="condition")
        side_effects = st.text_area(t["side_effects"], placeholder="e.g. nausea, dizziness" if lang=="en" else "مثال: غثيان، دوخة", key="side_effects")
        submitted = st.form_submit_button(t["get_recommendation"])

# =============================
# Tab 2: Prediction & Feedback
# =============================
with tab2:
    if submitted and side_effects.strip():
        # معالجة الدواء الجديد
        if new_drug.strip():
            drug_to_use = new_drug.strip()
            try:
                existing_drugs = pd.read_csv("new_drugs.csv")["Drug"].tolist()
            except FileNotFoundError:
                existing_drugs = []
            if drug_to_use not in existing_drugs:
                pd.DataFrame([{"Drug": drug_to_use, "Time": datetime.now().strftime("%Y-%m-%d %H:%M")}]) \
                  .to_csv("new_drugs.csv", mode="a", header=not pd.io.common.file_exists("new_drugs.csv"), index=False)
        else:
            drug_to_use = drug

        # Emergency check
        EMERGENCY = ['breathing','chest pain','seizure','unconscious','anaphylaxis'] if lang=="en" else ['صعوبة في التنفس','ألم في الصدر','تشنجات','فقدان الوعي','صدمة تحسسية']
        if any(k in side_effects.lower() for k in EMERGENCY):
            st.error(t["emergency"])
            st.stop()

        # Risk score
        risk_score = 0
        if age >= 65: risk_score += 2
        if smoker=="Yes" or smoker=="نعم": risk_score += 1
        chronic_options = ["Diabetes","Hypertension","Heart Disease","Kidney Disease","None"] if lang=="en" else ["سكري","ضغط الدم","أمراض القلب","أمراض الكلى","لا يوجد"]
        for d in chronic_diseases:
            if d in ["Heart Disease","أمراض القلب"]: risk_score += 2
            if d in ["Kidney Disease","أمراض الكلى"]: risk_score += 2
        st.metric(t["risk_score"], risk_score, "/10")

        # Feature encoding
        try:
            drug_enc = le_drug.transform([drug_to_use])[0]
        except ValueError:
            st.warning(t["drug_warning"])
            drug_enc = 0
        chronic_vector = [1 if d in chronic_diseases else 0 for d in chronic_options]
        has_chronic = 1 if sum(chronic_vector)>0 else 0

        text_vec = vectorizer.transform([side_effects])
        X_num = scaler.transform([[age, weight]])
        X = np.hstack([text_vec.toarray(), [[drug_enc, *chronic_vector, has_chronic]], X_num])

        probs = model.predict_proba(X)[0]
        idx = np.argmax(probs)
        decision = le_target.inverse_transform([idx])[0]
        confidence = probs[idx]

        thresholds = {"Continue":0.55, "See_Doctor":0.55, "Emergency":0.40}
        if confidence < thresholds.get(decision,0.5) or np.max(probs)<0.45 or risk_score>=4:
            decision = "See_Doctor"

        # Plotly Chart for Prediction Probabilities
        pred_df = pd.DataFrame({
            "Decision": le_target.inverse_transform(range(len(probs))),
            "Probability": probs
        })
        fig = px.bar(pred_df, x="Decision", y="Probability", text=pred_df["Probability"].apply(lambda x: f"{x:.1%}"),
                     color="Probability", color_continuous_scale="Viridis", title=t["prediction_probs"])
        st.plotly_chart(fig)

        # Final Recommendation
        st.subheader(t["final_recommendation"])
        if decision=="Continue":
            st.success(t["continue"])
        elif decision=="See_Doctor":
            st.warning(t["see_doctor"])
        else:
            st.error(t["emergency_msg"])

        # Feedback
        st.subheader(t["feedback"])
        feedback = st.radio(t["feedback"], [t["correct"], t["incorrect"], t["not_sure"]], key="feedback_radio")
        correct_decision = None
        if feedback==t["incorrect"]:
            correct_decision = st.selectbox(t["correct_decision"], le_target.classes_, key="correct_decision_select")

        # Save feedback
        record = {
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Age": age,
            "Gender": gender,
            "Weight": weight,
            "Smoker": smoker,
            "RiskScore": risk_score,
            "Drug": drug_to_use,
            "Condition": condition,
            "ChronicDiseases": ",".join(chronic_diseases),
            "Symptoms": side_effects,
            "Decision": decision,
            "Confidence": round(confidence,3),
            "Feedback": feedback,
            "CorrectDecision": correct_decision
        }
        st.session_state.history.append(record)
        if st.button("💾 Submit Feedback", key="submit_feedback"):
            pd.DataFrame([record]).to_csv("feedback_log.csv", mode="a", index=False, header=not pd.io.common.file_exists("feedback_log.csv"))
            st.success(t["feedback_saved"])

# =============================
# Tab 3: History & Analysis
# =============================
with tab3:
    if st.session_state.history:
        df_history = pd.DataFrame(st.session_state.history)
        with st.expander(t["previous_decisions"], expanded=True):
            st.dataframe(df_history)

        with st.expander("📊 Feedback Analysis / تحليل التغذية الراجعة", expanded=True):
            feedback_counts = df_history['Feedback'].value_counts().reset_index()
            feedback_counts.columns = ["Feedback","Count"]
            fig_fb = px.bar(feedback_counts, x="Feedback", y="Count", text="Count", color="Count", color_continuous_scale="Inferno")
            st.plotly_chart(fig_fb)

            incorrect_df = df_history[df_history['Feedback']==t["incorrect"]]
            if not incorrect_df.empty:
                top_corrections = incorrect_df['CorrectDecision'].value_counts().reset_index()
                top_corrections.columns = ["Decision","Count"]
                fig_corr = px.bar(top_corrections.head(5), x="Decision", y="Count", text="Count", color="Count", color_continuous_scale="Blues",
                                  title="Most corrected decisions / أكثر القرارات التي تم تصحيحها")
                st.plotly_chart(fig_corr)

            avg_conf = df_history.groupby('Decision')['Confidence'].mean().reset_index()
            fig_conf = px.bar(avg_conf, x='Decision', y='Confidence', text=avg_conf['Confidence'].apply(lambda x: f"{x:.2f}"),
                              color='Confidence', color_continuous_scale="Viridis", title="Average confidence per decision / متوسط الثقة لكل قرار")
            st.plotly_chart(fig_conf)

        with st.expander("🚨 Emergency Cases / حالات الطوارئ", expanded=False):
            emergency_cases = df_history[df_history['Decision']=="Emergency"]
            st.write(f"Total: {len(emergency_cases)}")
            if not emergency_cases.empty:
                st.dataframe(emergency_cases[['Time','Drug','Condition','ChronicDiseases','Symptoms','Confidence']])

