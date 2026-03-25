import streamlit as st
import joblib
import numpy as np

# 1. Load the model
model = joblib.load('model.pkl')

st.title("ML Prediction App")

# 2. Get user input
val = st.number_input("Enter Input Value", value=0.0)

# 3. Predict
if st.button("Predict"):
    prediction = model.predict([[val]])
    st.success(f"The result is: {prediction[0]}")
