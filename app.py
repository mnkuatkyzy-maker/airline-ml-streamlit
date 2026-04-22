import streamlit as st
import joblib

st.title("✈️ Airline App")

# 👉 показываем, что приложение стартовало
st.write("🚀 App started")

@st.cache_resource
def load_model():
    st.write("⏳ Loading model...")
    model = joblib.load("xgb_pipeline.pkl")
    st.write("✅ Model loaded")
    return model

model = load_model()
