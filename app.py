import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

# 1. Load the model safely
try:
    # Try loading as a Keras model first
    model = load_model('model.pkl')
except:
    # Fallback to joblib if it's a standard sklearn model
    model = joblib.load('model.pkl')
