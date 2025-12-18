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
# إعداد الصفحة واللغة
# =============================

# إعداد الصفحة
st.set_page_config(page_title="Medical AI CDS", layout="centered")

# اختيار اللغة
lang_choice = st.selectbox("Language / اللغة", ["English", "العربية"])

# تحويل اختيار المستخدم إلى مفتاح القاموس الصحيح
lang = "en" if lang_choice == "English" else "ar"

# نصوص اللغات
texts = {
    "en": {
        "welcome": "Welcome to Medical AI CDS",
        "description": "This system helps you with medicine reminders."
        # أضف باقي النصوص هنا
    },
    "ar": {
        "welcome": "مرحبًا بك في النظام الطبي الذكي",
        "description": "هذا النظام يساعدك في تذكيرك بالأدوية."
        # أضف باقي النصوص هنا
    }
}

# جلب النصوص حسب اللغة المختارة
t = texts[lang]

# مثال على استخدامها
st.title(t["welcome"])
st.write(t["description"])


st.title(t["title"])
st.caption(t["caption"])
st.warning(t["warning"])

if 'history' not in st.session_state:
    st.session_state.history = []

# =============================
# Tabs
# =============================
tab1, tab2, tab3 = st.tabs([t["patient_info"], t["prediction_feedback"], t["history_analysis"]])

# =============================
# Tab 1: Patient Info & Medication
# =============================
with tab1:
    with st.form("patient_form"):
        age = st.number_input(t["age"], 0, 120, 30)
        gender = st.selectbox(t["gender"], ["Male","Female","Other"] if lang=="English" else ["ذكر","أنثى","آخر"])
        weight = st.number_input(t["weight"], 90.0, 200.0, 70.0)
        smoker = st.selectbox(t["smoker"], ["No","Yes"] if lang=="English" else ["لا","نعم"])
        chronic_diseases = st.multiselect(
            t["chronic_diseases"],
            ["Diabetes","Hypertension","Heart Disease","Kidney Disease","None"] if lang=="English" else ["سكري","ضغط الدم","أمراض القلب","أمراض الكلى","لا يوجد"]
        )
        drug = st.selectbox(t["drug"], le_drug.classes_)
        new_drug = st.text_input(t["new_drug"])
        condition = st.selectbox(t["condition"], le_target.classes_)
        side_effects = st.text_area(
            t["side_effects"],
            placeholder="e.g. nausea, dizziness" if lang=="English" else "مثال: غثيان، دوخة"
        )
        submitted = st.form_submit_button(t["get_recommendation"])

# =============================
# Tab 2: Prediction & Feedback with Plotly
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
        EMERGENCY = ['breathing','chest pain','seizure','unconscious','anaphylaxis'] if lang=="English" else ['صعوبة في التنفس','ألم في الصدر','تشنجات','فقدان الوعي','صدمة تحسسية']
        if any(k in side_effects.lower() for k in EMERGENCY):
            st.error(t["emergency"])
            st.stop()

        # Risk score
        risk_score = 0
        if age >= 65: risk_score += 2
        if smoker=="Yes" or smoker=="نعم": risk_score += 1
        if ("Heart Disease" in chronic_diseases) or ("أمراض القلب" in chronic_diseases): risk_score +=2
        if ("Kidney Disease" in chronic_diseases) or ("أمراض الكلى" in chronic_diseases): risk_score +=2
        st.metric(t["risk_score"], risk_score, "/10")

        # Feature encoding
        try:
            drug_enc = le_drug.transform([drug_to_use])[0]
        except ValueError:
            st.warning(t["drug_warning"])
            drug_enc = 0
        chronic_options = ["Diabetes","Hypertension","Heart Disease","Kidney Disease","None"] if lang=="English" else ["سكري","ضغط الدم","أمراض القلب","أمراض الكلى","لا يوجد"]
        chronic_vector = [1 if d in chronic_diseases else 0 for d in chronic_options]
        unknown_chronic = [d for d in chronic_diseases if d not in chronic_options]
        if unknown_chronic:
            st.warning(f"{t['chronic_warning']}: {', '.join(unknown_chronic)}")
        has_chronic = 1 if sum(chronic_vector) > 0 else 0

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

        # =============================
        # Plotly Chart for Prediction Probabilities
        # =============================
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
        if feedback == t["incorrect"]:
            correct_decision = st.selectbox(t["correct_decision"], le_target.classes_, key="correct_decision_select")

        if st.button("💾 Submit Feedback"):
            with open("feedback_log.csv","a",newline="",encoding="utf-8") as f:
                pd.DataFrame([{
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
                }]).to_csv(f, index=False, header=f.tell()==0)
            st.success(t["feedback_saved"])

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

# =============================
# Tab 3: History & Analysis with Plotly
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
