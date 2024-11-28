import streamlit as st
from datetime import datetime
import pandas as pd
from joblib import load
from prophet.serialize import model_from_json

with open("prophet_model.json", "r") as f:
    model = model_from_json(f.read())

# Streamlit app title
st.title("Page Load Prediction")

# User input for month and year
month = st.selectbox("Select Month", range(1, 13))
year = st.number_input("Enter Year", min_value=2000, max_value=2100, step=1)

if st.button("Predict"):
    # Prepare future dataframe
    date = datetime(year, month, 1)
    future = pd.DataFrame({'ds': [date]})
    
    # Make prediction
    forecast = model.predict(future)
    prediction = forecast.loc[0, 'yhat']
    
    # Display the prediction
    st.write(f"Predicted page loads for {date.strftime('%B %Y')}: {prediction:.2f}")