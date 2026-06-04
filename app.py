import streamlit as pd
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from datetime import datetime, timedelta

# 1. Set up the web page title and icon
st.set_page_config(page_title="EcoGuard AIOps Dashboard", page_icon="🛡️", layout="wide")

st.title("🛡️ EcoGuard: Moogsoft Predictive Incident Forecaster")
st.markdown("### Powered by AI/ML • Real-Time Infrastructure Risk Management")
st.write("---")

# 2. Load our trained AI model and historical data
@st.cache_resource
def load_artifacts():
    model = joblib.load("moogsoft_forecaster_model.pkl")
    df = pd.read_csv("moogsoft_historical_data.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return model, df

model, df = load_artifacts()

# 3. Create Sidebar Filters
st.sidebar.header("🎛️ Dashboard Controls")
threshold = st.sidebar.slider("Set Critical Alert Threshold", min_value=30, max_value=100, value=50)

# 4. Generate Future Forecast (Next 24 Hours)
st.subheader("🔮 Next 24-Hour Predictive Breakdown")

last_row = df.iloc[-1]
last_timestamp = last_row['timestamp']
current_lag1 = last_row['alert_count']
current_lag2 = df.iloc[-2]['alert_count']

future_preds = []
for i in range(1, 25):
    future_time = last_timestamp + timedelta(hours=i)
    hour = future_time.hour
    day_of_week = future_time.dayofweek
    
    # Predict using our saved ML model
    features = pd.DataFrame([[hour, day_of_week, current_lag1, current_lag2]], 
                            columns=['hour', 'day_of_week', 'lag_1', 'lag_2'])
    predicted_alerts = model.predict(features)[0]
    
    future_preds.append({
        "Timestamp": future_time,
        "Predicted Alerts": round(predicted_alerts, 1)
    })
    
    # Update lags for the next loop iteration
    current_lag2 = current_lag1
    current_lag1 = predicted_alerts

future_df = pd.DataFrame(future_preds)

# 5. Check if system is going to breach the user-defined threshold
max_predicted = future_df['Predicted Alerts'].max()
peak_time = future_df.loc[future_df['Predicted Alerts'].idxmax(), 'Timestamp']

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Current System Status", value="HEALTHY", delta="0 Active Incidents")
with col2:
    st.metric(label="Predicted Peak Alert Volume", value=f"{max_predicted} / hr")
with col3:
    if max_predicted >= threshold:
        st.error(f"🚨 RISK DETECTED: System expected to breach threshold at {peak_time.strftime('%H:%M')}!")
    else:
        st.success("✅ SYSTEM SAFE: No critical breaches forecasted for the next 24 hours.")

st.write("---")

# 6. Plot the Interactive Interactive Graph
st.subheader("📈 Future Alert Volume Forecast")
fig = px.line(future_df, x="Timestamp", y="Predicted Alerts", 
              title="AI Predicted Incident Velocity Over Time",
              labels={"Predicted Alerts": "Alerts per Hour"})

# Add a dashed line showing our critical threshold limit
fig.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="Critical Threshold Trigger")
st.plotly_chart(fig, use_container_width=True)

print("💻 Dashboard running seamlessly in background!")