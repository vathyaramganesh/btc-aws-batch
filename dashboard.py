import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Load CSV
try:
    df = pd.read_csv("btc_prices_data.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values("timestamp")
except FileNotFoundError:
    st.error("❌ btc_prices_data.csv not found. Please run ingest.py first.")
    st.stop()

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
