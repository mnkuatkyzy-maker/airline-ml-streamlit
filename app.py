import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# =============================
# LOAD MODEL
# =============================
@st.cache_resource
def load_model():
    return joblib.load("xgb_pipeline.pkl")

model = load_model()

# SHAP explainer (для XGBoost внутри pipeline)
@st.cache_resource
def load_explainer(model):
    return shap.TreeExplainer(model.named_steps['clf'])

explainer = load_explainer(model)

# =============================
# CONFIG
# =============================
st.set_page_config(page_title="Airline Predictor", layout="centered")
st.title("✈️ Airline Satisfaction Predictor")

# =============================
# INPUT
# =============================
st.header("Passenger Info")

age = st.slider("Age", 10, 80, 30)
distance = st.number_input("Flight Distance", 100, 5000, 1000)

delay_dep = st.number_input("Departure Delay", 0, 500, 10)
delay_arr = st.number_input("Arrival Delay", 0, 500, 10)

gender = st.selectbox("Gender", ["Male", "Female"])
customer_type = st.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
travel_type = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
flight_class = st.selectbox("Class", ["Business", "Eco", "Eco Plus"])

# =============================
# SERVICES
# =============================
st.subheader("Service Ratings (0–5)")

SERVICE_COLS = [
    'Inflight wifi service', 'Departure/Arrival time convenient',
    'Ease of Online booking', 'Gate location', 'Food and drink',
    'Online boarding', 'Seat comfort', 'Inflight entertainment',
    'On-board service', 'Leg room service', 'Baggage handling',
    'Checkin service', 'Inflight service', 'Cleanliness'
]

service_values = {}
for col in SERVICE_COLS:
    service_values[col] = st.slider(col, 0, 5, 3)

# =============================
# FEATURE ENGINEERING
# =============================
Total_delay = delay_dep + delay_arr
Log_Flight_Distance = np.log1p(distance)
Service_avg = np.mean(list(service_values.values()))

if age <= 25:
    Age_group = "Young"
elif age <= 60:
    Age_group = "Middle"
else:
    Age_group = "Senior"

# =============================
# CREATE INPUT
# =============================
input_dict = {
    'Age': age,
    'Flight Distance': distance,
    'Total_delay': Total_delay,
    'Log_Flight_Distance': Log_Flight_Distance,
    'Service_avg': Service_avg,
    'Gender': gender,
    'Customer Type': customer_type,
    'Type of Travel': travel_type,
    'Class': flight_class,
    'Age_group': Age_group
}

input_dict.update(service_values)

input_df = pd.DataFrame([input_dict])

# =============================
# PREDICTION + SHAP
# =============================
if st.button("Predict"):

    # Prediction
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if pred == 1:
        st.success(f"✅ Satisfied ({prob:.2%})")
    else:
        st.error(f"❌ Not satisfied ({prob:.2%})")

    # Debug (можно убрать потом)
    st.subheader("Input Data")
    st.write(input_df)

    # =============================
    # SHAP
    # =============================
    st.subheader("🔍 SHAP Explanation")

    try:
        # трансформация через pipeline
        X_transformed = model.named_steps['pre'].transform(input_df)

        # shap values
        shap_values = explainer.shap_values(X_transformed)

        # если возвращает список (иногда бывает)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # Waterfall plot (1 observation)
        fig = plt.figure()
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)

    except Exception as e:
        st.warning("SHAP failed to render")
        st.text(str(e))
