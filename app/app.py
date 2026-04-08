
# Python Standard Libraries and Built-in modules
from pathlib import Path
import os

# Third Party libraries
import joblib
import pandas as pd
import numpy as np


# Hugging Face Libraries
from huggingface_hub import create_repo
from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download
import streamlit as st

# Machine Learning Models 
from xgboost import XGBClassifier

# TITLE
st.title("Engine Condition Predictor")

# Code to load the saved xgboost model from Hugging Face model hub
def load_model():
    try:
        model_path = hf_hub_download(
            repo_id="koustubhsuratran/predictive-engine-maintenance-model",
            filename="models/best_model.joblib"
        )
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

model = load_model()

# get the inputs and save them into a dataframe
def get_input_dataframe(inputs, columns=None):
    if columns:
        df = pd.DataFrame([inputs], columns=columns[:len(inputs)])
    else:
        df = pd.DataFrame([inputs])
    return df

# Input data for prediction
st.header("Enter Engine Parameters")

engine_rpm = st.number_input("Engine RPM", value=0.0)
lub_oil_pressure = st.number_input("Lub Oil Pressure", value=0.0)
fuel_pressure = st.number_input("Fuel Pressure", value=0.0)
coolant_pressure = st.number_input("Coolant Pressure", value=0.0)
lub_oil_temp = st.number_input("Lub Oil Temperature", value=0.0)
coolant_temp = st.number_input("Coolant Temperature", value=0.0)


# Prediction code

if st.button("Predict"):
    if model is None:
        st.warning("Model not loaded")
    else:
        try:
            inputs = [
                engine_rpm,
                lub_oil_pressure,
                fuel_pressure,
                coolant_pressure,
                lub_oil_temp,
                coolant_temp
            ]
            input_data = get_input_dataframe(inputs,columns=[
                "engine_rpm",
                "lub_oil_pressure",
                "fuel_pressure",
                "coolant_pressure",
                "lub_oil_temp",
                "coolant_temp"
            ])
            prediction = model.predict(input_data)
            st.success(f"Prediction: {prediction[0]}")

        except Exception as e:
            st.error(f"Prediction error: {e}")