import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")
st.title("🩺 Doctor Review Dashboard")

df = pd.read_csv("feedback_log.csv")
if df.empty:
    st.info("No feedback yet.")
    st.stop()

# ----------------------------
# Low Confidence Cases
# ----------------------------
low_conf = df[df["confidence"]<0.55]
st.subheader("Low Confidence Cases")
st.dataframe(low_conf)

# ----------------------------
# Model Decision Distribution
# ----------------------------
st.subheader("Decision Distribution")
st.bar_chart(df["model_decision"].value_counts())

# ----------------------------
# Real-world Accuracy
# ----------------------------
validated = df[df["human_decision"].notna()]
if not validated.empty:
    acc = (validated["model_decision"]==validated["human_decision"]).mean()*100
    st.metric("Real-world Accuracy",f"{acc:.2f}%")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    st.dataframe(pd.crosstab(
        validated["human_decision"],
        validated["model_decision"]
    ))

    # Accuracy by Language
    st.subheader("Accuracy by Language")
    st.bar_chart(validated.groupby("language").apply(
        lambda x:(x["model_decision"]==x["human_decision"]).mean()*100
    ))

    # Accuracy vs Confidence
    st.subheader("Accuracy vs Confidence")
    validated["conf_bucket"] = pd.cut(validated["confidence"], bins=[0,0.4,0.6,0.8,1.0])
    st.line_chart(validated.groupby("conf_bucket").apply(
        lambda x:(x["model_decision"]==x["human_decision"]).mean()*100
    ))
else:
    st.info("No validated feedback yet.")
# ----------------------------