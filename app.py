import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(page_title="Airline Predictor", layout="centered")

# =============================
# LOAD MODEL
# =============================
@st.cache_resource
def load_model():
    return joblib.load("xgb_pipeline.pkl")

model = load_model()

# =============================
# LOAD SHAP (FIXED 🔥)
# =============================
@st.cache_resource
def load_explainer(model):
    return shap.TreeExplainer(model.named_steps['clf'])

explainer = load_explainer(model)

# =============================
# PREPROCESSOR + FEATURE NAMES
# =============================
preprocessor = model.named_steps['pre']

num_features = preprocessor.transformers_[0][2]

cat_features = preprocessor.transformers_[1][1] \
    .named_steps['encoder'] \
    .get_feature_names_out(preprocessor.transformers_[1][2])

feature_names = list(num_features) + list(cat_features)

# =============================
# CACHE TRANSFORM (⚡ speed)
# =============================
@st.cache_data
def transform_input(df):
    return model.named_steps['pre'].transform(df)

# =============================
# TITLE
# =============================
st.title("✈️ Airline Satisfaction Predictor")

st.markdown("""
Predict whether a passenger is **satisfied** based on flight and service data.  
Model: **XGBoost (~96% accuracy)**
""")

# =============================
# SIDEBAR INPUT
# =============================
st.sidebar.header("🧾 Passenger Info")

age = st.sidebar.slider("Age", 10, 80, 30)
distance = st.sidebar.number_input("Flight Distance", 100, 5000, 1000)

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
customer_type = st.sidebar.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
travel_type = st.sidebar.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
flight_class = st.sidebar.selectbox("Class", ["Business", "Eco", "Eco Plus"])

# =============================
# MAIN INPUT (better layout)
# =============================
st.header("✈️ Flight Details")

col1, col2 = st.columns(2)

with col1:
    delay_dep = st.number_input("Departure Delay", 0, 500, 10)

with col2:
    delay_arr = st.number_input("Arrival Delay", 0, 500, 10)

# =============================
# SERVICES (hidden)
# =============================
st.subheader("⭐ Service Ratings")

SERVICE_COLS = [
    'Inflight wifi service', 'Departure/Arrival time convenient',
    'Ease of Online booking', 'Gate location', 'Food and drink',
    'Online boarding', 'Seat comfort', 'Inflight entertainment',
    'On-board service', 'Leg room service', 'Baggage handling',
    'Checkin service', 'Inflight service', 'Cleanliness'
]

service_values = {}

with st.expander("Adjust service ratings (optional)"):
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
# PREDICTION
# =============================
st.divider()

if st.button("🚀 Predict"):

    with st.spinner("Analyzing passenger..."):

        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

    # =============================
    # RESULT UI
    # =============================
    st.subheader("📊 Result")

    st.metric("Satisfaction Probability", f"{prob:.2%}")

    if pred == 1:
        st.success("✅ Passenger is likely satisfied")
    else:
        st.error("❌ Passenger is NOT satisfied")

    # =============================
    # INPUT DATA
    # =============================
    with st.expander("📄 View input data"):
        st.write(input_df)

    # =============================
    # SHAP TOGGLE
    # =============================
    show_shap = st.checkbox("🔍 Show model explanation (SHAP)")

    if show_shap:
        try:
            with st.spinner("Calculating SHAP..."):

                X_transformed = transform_input(input_df)
                shap_values = explainer(X_transformed)

                shap_values.feature_names = feature_names

                # Plot
                fig, ax = plt.subplots()
                shap.plots.waterfall(shap_values[0], show=False)
                st.pyplot(fig)

                # =============================
                # TOP FEATURES
                # =============================
                st.subheader("Top factors")

                values = shap_values.values[0]
                top_idx = np.argsort(np.abs(values))[::-1][:5]

                for i in top_idx:
                    impact = "⬆️ increases" if values[i] > 0 else "⬇️ decreases"
                    st.write(f"{feature_names[i]} {impact} satisfaction ({values[i]:.3f})")

        except Exception as e:
            st.warning("SHAP failed")
            st.text(str(e))
