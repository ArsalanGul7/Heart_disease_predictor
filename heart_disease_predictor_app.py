import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("heart_disease_prediction.jb")

# App Config
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("❤️ Heart Disease Predictor")

st.markdown("""
Use this app to predict the **likelihood of heart disease** based on several health metrics.
Please enter your information below and click **Predict** to see the result.
""")

# --- User Inputs with Descriptions ---
age = st.slider("Age", 20, 100, 50)

sex = st.radio("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")

cp = st.selectbox(
    "Chest Pain Type",
    options=[0, 1, 2, 3],
    format_func=lambda x: [
        "Typical angina",
        "Atypical angina",
        "Non-anginal pain",
        "Asymptomatic"
    ][x]
)

trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)

chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)

fbs = st.radio(
    "Fasting Blood Sugar > 120 mg/dl",
    options=[0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

restecg = st.selectbox(
    "Resting Electrocardiographic Results",
    options=[0, 1, 2],
    format_func=lambda x: [
        "Normal",
        "ST-T wave abnormality",
        "Left ventricular hypertrophy"
    ][x]
)

thalach = st.number_input("Maximum Heart Rate Achieved", 60, 250, 150)

exang = st.radio(
    "Exercise Induced Angina",
    options=[0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

oldpeak = st.slider("ST Depression Induced by Exercise", 0.0, 6.0, 1.0, step=0.1)

slope = st.selectbox(
    "Slope of Peak Exercise ST Segment",
    options=[0, 1, 2],
    format_func=lambda x: [
        "Upsloping",
        "Flat",
        "Downsloping"
    ][x]
)

ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3, 4])

thal = st.selectbox(
    "Thalassemia (Thallium Stress Test Result)",
    options=[1, 2, 3],
    format_func=lambda x: [
        "Normal",
        "Fixed defect",
        "Reversible defect"
    ][x - 1]
)

# Combine inputs
features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                      thalach, exang, oldpeak, slope, ca, thal]])

# Predict Button
if st.button("Predict"):
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    st.markdown("### 🔍 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ High chance of Heart Disease.\n**Probability: {probability:.2f}**")
    else:
        st.success(f"✅ Low chance of Heart Disease.\n**Probability: {probability:.2f}**")
