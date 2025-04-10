import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

import boto3
import os

# Load AWS credentials from secrets
aws_id = st.secrets["AWS_ACCESS_KEY_ID"]
aws_secret = st.secrets["AWS_SECRET_ACCESS_KEY"]
region = st.secrets["AWS_REGION"]
bucket = st.secrets["S3_BUCKET"]
s3_key = st.secrets["S3_KEY"]

# Download the file from S3
s3 = boto3.client("s3", aws_access_key_id=aws_id, aws_secret_access_key=aws_secret, region_name=region)
local_file = "btc_prices_data.csv"

try:
    s3.download_file(bucket, s3_key, local_file)
    st.success("✅ Loaded latest data from S3")
except Exception as e:
    st.error(f"❌ Failed to load data from S3: {e}")
    st.stop()

# Now load the data
df = pd.read_csv(local_file)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values("timestamp")


# App Title
st.title("🪙 Bitcoin Real-Time Dashboard")

# Show latest price
latest = df.iloc[-1]
st.metric(label="Current BTC Price (USD)", value=f"${latest['price_usd']:.2f}", delta=None)

# Plot price history
st.subheader("📈 BTC Price Trend")
fig, ax = plt.subplots()
ax.plot(df['timestamp'], df['price_usd'], label="Price")
df['moving_avg'] = df['price_usd'].rolling(window=5).mean()
ax.plot(df['timestamp'], df['moving_avg'], linestyle='--', label="5-Point Moving Avg")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
st.pyplot(fig)


from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta

st.subheader("🔮 BTC Price Forecast (ARIMA)")

# Fit ARIMA model
model = ARIMA(df['price_usd'], order=(3, 1, 2))
fitted_model = model.fit()

# Forecast next 5 prices
forecast_steps = 5
forecast = fitted_model.forecast(steps=forecast_steps)

# Future timestamps
last_time = df['timestamp'].iloc[-1]
future_times = [last_time + timedelta(minutes=5 * (i + 1)) for i in range(forecast_steps)]

# Plot
fig2, ax2 = plt.subplots()
ax2.plot(df['timestamp'], df['price_usd'], label="Actual")
ax2.plot(future_times, forecast, label="Forecast", linestyle='--', marker='o')
ax2.legend()
ax2.grid(True)
plt.xticks(rotation=45)
st.pyplot(fig2)
