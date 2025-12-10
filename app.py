import streamlit as st
import pandas as pd
import joblib

# Load model
pipeline = joblib.load("model.pkl")

st.title("Hotel Cancellation Prediction App")

# Input fields
hotel = st.selectbox("Hotel", ["City Hotel", "Resort Hotel"])
lead_time = st.number_input("Lead time", 0, 500, 50)
arrival_month = st.selectbox("Arrival Month", 
                             ["January","February","March","April","May","June",
                              "July","August","September","October","November","December"])

# Example input
new_booking = pd.DataFrame([{
    "hotel": hotel,
    "lead_time": lead_time,
    "arrival_date_month": arrival_month,
    "total_nights": 5,
    "adults": 2,
    "children": 0,
    "babies": 0,
    "meal": "BB",
    "country": "PRT",
    "market_segment": "Online TA",
    "distribution_channel": "TA/TO",
    "previous_cancellations": 0,
    "deposit_type": "No Deposit",
    "customer_type": "Transient"
}])

# Predict
if st.button("Predict"):
    pred = pipeline.predict(new_booking)[0]
    st.write("Cancellation :", "Yes" if pred==1 else "No")
