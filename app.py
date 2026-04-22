import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("✈️ Airline Passenger Satisfaction")

# -------------------------------
# LOAD MODEL (FIX: cache)
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load("xgb_pipeline.pkl")

st.write("🔄 Loading model...")
model = load_model()
st.success("✅ Model loaded")

# -------------------------------
# INPUT
# -------------------------------
st.header("Enter Passenger Data")

age = st.slider("Age", 0, 100, 30)
distance = st.number_input("Flight Distance", 0, 10000, 1000)

dep_delay = st.number_input("Departure Delay", 0, 1000, 0)
arr_delay = st.number_input("Arrival Delay", 0, 1000, 0)

gender = st.selectbox("Gender", ["Male", "Female"])
customer_type = st.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
travel_type = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
flight_class = st.selectbox("Class", ["Business", "Eco", "Eco Plus"])

st.subheader("Service Ratings (0–5)")

wifi = st.slider("Inflight wifi service", 0, 5, 3)
time_conv = st.slider("Departure/Arrival time convenient", 0, 5, 3)
booking = st.slider("Ease of Online booking", 0, 5, 3)
gate = st.slider("Gate location", 0, 5, 3)
food = st.slider("Food and drink", 0, 5, 3)
boarding = st.slider("Online boarding", 0, 5, 3)
seat = st.slider("Seat comfort", 0, 5, 3)
ent = st.slider("Inflight entertainment", 0, 5, 3)
onboard = st.slider("On-board service", 0, 5, 3)
leg = st.slider("Leg room service", 0, 5, 3)
baggage = st.slider("Baggage handling", 0, 5, 3)
checkin = st.slider("Checkin service", 0, 5, 3)
inflight = st.slider("Inflight service", 0, 5, 3)
clean = st.slider("Cleanliness", 0, 5, 3)

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------
total_delay = dep_delay + arr_delay
log_distance = np.log1p(distance)

service_avg = np.mean([
    wifi, time_conv, booking, gate, food, boarding,
    seat, ent, onboard, leg, baggage, checkin,
    inflight, clean
])

if age < 25:
    age_group = "Young"
elif age < 60:
    age_group = "Middle"
else:
    age_group = "Senior"

# -------------------------------
# DATAFRAME (ВАЖНО: ВСЕ КОЛОНКИ)
# -------------------------------
input_df = pd.DataFrame([{
    "Age": age,
    "Flight Distance": distance,
    "Inflight wifi service": wifi,
    "Departure/Arrival time convenient": time_conv,
    "Ease of Online booking": booking,
    "Gate location": gate,
    "Food and drink": food,
    "Online boarding": boarding,
    "Seat comfort": seat,
    "Inflight entertainment": ent,
    "On-board service": onboard,
    "Leg room service": leg,
    "Baggage handling": baggage,
    "Checkin service": checkin,
    "Inflight service": inflight,
    "Cleanliness": clean,
    "Total_delay": total_delay,
    "Log_Flight_Distance": log_distance,
    "Service_avg": service_avg,
    "Gender": gender,
    "Customer Type": customer_type,
    "Type of Travel": travel_type,
    "Class": flight_class,
    "Age_group": age_group
}])

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("Predict"):
    try:
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        if pred == 1:
            st.success(f"✅ Satisfied (probability: {prob:.2f})")
        else:
            st.error(f"❌ Not satisfied (probability: {prob:.2f})")

    except Exception as e:
        st.error("⚠️ Error during prediction")
        st.write(e)
