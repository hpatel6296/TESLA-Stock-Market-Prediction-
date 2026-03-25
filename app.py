import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, SimpleRNN, Dense, Dropout

# Page Config
st.set_page_config(page_title="Tesla Stock Predictor", layout="wide")
st.title("🚗 Tesla Stock Price Prediction")
st.write("Comparing SimpleRNN and LSTM models for Financial Forecasting.")

# 1. Load Data
@st.cache_data
def load_data():
    # Ensure TSLA.csv is in your GitHub folder
    df = pd.read_csv("TSLA.csv") 
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

try:
    df = load_data()
    
    # 2. Data Selection (Adj Close)
    data = df.filter(['Adj Close'])
    dataset = data.values
    training_data_len = int(np.ceil(len(dataset) * .8))

    # 3. Scaling
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    # Sidebar for Model Selection
    model_choice = st.sidebar.selectbox("Select Model", ("LSTM", "SimpleRNN"))
    window_size = st.sidebar.slider("Window Size (Past Days)", 30, 100, 60)

    # 4. Model Building Function
    def build_model(m_type):
        model = Sequential()
        if m_type == "LSTM":
            model.add(LSTM(50, return_sequences=True, input_shape=(window_size, 1)))
            model.add(LSTM(50, return_sequences=False))
        else:
            model.add(SimpleRNN(50, return_sequences=True, input_shape=(window_size, 1)))
            model.add(SimpleRNN(50, return_sequences=False))
        
        model.add(Dropout(0.2)) #
        model.add(Dense(25))
        model.add(Dense(1)) # Output layer
        model.add(compile(optimizer='adam', loss='mean_squared_error')) #
        return model

    st.success(f"Running {model_choice} Model Analysis...")

    # 5. Visualization
    st.subheader("Tesla Adjusted Closing Price History")
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(df['Adj Close'], label='Actual Price')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    plt.legend()
    st.pyplot(fig)

    # Note: In a real deployment, you should pre-train and load .h5 files
    # to avoid timeout errors on Streamlit Cloud.
    st.info("Tip: For the final submission, ensure you upload your trained .h5 models and use `load_model()`.")

except Exception as e:
    st.error(f"Error loading data: {e}")
    st.warning("Please ensure 'TSLA.csv' is uploaded to your repository.")
