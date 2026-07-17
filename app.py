import streamlit as st
import numpy as np
import joblib

model = joblib.load("models/diabetes_model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.title("Healthcare Intrusion Detection System (IDS)")
st.subheader("Diabetes Detection Using Machine Learning")

st.write(
    "This system detects abnormal health patterns similar to how an "
    "Intrusion Detection System detects cyber attacks."
)

st.header("Enter Patient Information")

pregnancies = st.number_input("Pregnancies", 0, 20)
glucose = st.number_input("Glucose Level", 0, 200)
blood_pressure = st.number_input("Blood Pressure", 0, 150)
skin_thickness = st.number_input("Skin Thickness", 0, 100)
insulin = st.number_input("Insulin Level", 0, 900)
bmi = st.number_input("BMI", 0.0, 70.0)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
age = st.number_input("Age", 1, 120)

if st.button("Predict Diabetes"):

    input_data = np.array([
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age
    ]).reshape(1, -1)

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    diabetes_prob = probability[0][1] * 100

    st.write(f"Diabetes Probability: {diabetes_prob:.2f}%")

    if prediction[0] == 1:
        st.error("⚠️ Anomaly Detected: Patient likely has Diabetes")
    else:
        st.success("✅ Normal Pattern: Patient likely Non-Diabetic")
