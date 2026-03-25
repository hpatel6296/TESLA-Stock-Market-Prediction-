import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Tesla Stock Predictor", layout="wide")

st.title("🚗 Tesla Stock Price Prediction")
st.write("Comparing SimpleRNN and LSTM models for sequential price forecasting.") [cite: 103, 104]

# Sidebar for Model Selection
model_type = st.sidebar.selectbox("Select Model Architecture", ("LSTM", "SimpleRNN")) [cite: 81]
forecast_days = st.sidebar.radio("Forecast Horizon", (1, 5, 10)) [cite: 21]

# Placeholder for Data Loading
@st.cache_data
def load_data():
    # Use the provided dataset link for the CSV
    df = pd.read_csv("TSLA.csv") # Ensure this file is in your GitHub repo [cite: 68]
    df['Date'] = pd.to_datetime(df['Date']) [cite: 72]
    return df

data = load_data()
st.subheader("Recent Stock Data")
st.dataframe(data.tail(10))

# Visualization
st.subheader(f"Tesla Closing Price Trend") [cite: 19]
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(data['Date'], data['Adj Close'], label='Actual Adj Close') [cite: 71, 96]
plt.legend()
st.pyplot(fig)

st.info("Upload your .h5 model files to the repository to enable active predictions.")
