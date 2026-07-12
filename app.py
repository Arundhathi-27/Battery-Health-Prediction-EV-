import streamlit as st
import joblib
import pandas as pd
from src.maintenance_rules import get_recommendation

st.set_page_config(page_title="Battery Health Predictor", layout="centered")

@st.cache_resource
def load_models():
    soh_model = joblib.load("models/soh_model.pkl")
    rul_model = joblib.load("models/rul_model.pkl")
    status_model = joblib.load("models/status_model.pkl")
    label_encoder = joblib.load("models/status_label_encoder.pkl")
    return soh_model, rul_model, status_model, label_encoder

soh_model, rul_model, status_model, label_encoder = load_models()

st.title("🔋 EV Battery Health Prediction")
st.write("Enter battery readings to get SOH, RUL, Status, and Maintenance advice.")

cycle = st.number_input("Cycle Number", min_value=1, value=100)
voltage = st.number_input("Voltage (V)", value=3.7)
current = st.number_input("Current (A)", value=-1.5)
temperature = st.number_input("Temperature (°C)", value=25.0)
discharge_time = st.number_input("Discharge Time (s)", value=3600.0)

if st.button("Predict Battery Health"):
    input_data = pd.DataFrame([{
        "cycle": cycle,
        "voltage": voltage,
        "current": current,
        "temperature": temperature,
        "discharge_time": discharge_time
    }])

    soh_pred = soh_model.predict(input_data)[0]

    input_data_rul = input_data.copy()
    input_data_rul["SOH"] = soh_pred
    rul_pred = rul_model.predict(input_data_rul)[0]

    status_pred_encoded = status_model.predict(input_data_rul)[0]
    status_pred = label_encoder.inverse_transform([status_pred_encoded])[0]

    recommendation = get_recommendation(status_pred, soh_pred)

    st.subheader("Results")
    col1, col2 = st.columns(2)
    col1.metric("SOH (%)", f"{soh_pred:.2f}")
    col2.metric("RUL (cycles)", f"{int(rul_pred)}")

    st.write(f"**Battery Status:** {status_pred}")
    st.info(f"**Maintenance Recommendation:** {recommendation}")