import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from datetime import datetime, timedelta

# 1. Premium Theme Config
st.set_page_config(
    page_title="EcoGuard AIOps Dashboard", 
    page_icon="🛡️", 
    layout="wide"
)

# Custom Slate-Dark CSS styling for a professional corporate finish
st.markdown("""
    <style>
        .main { background-color: #0B0F19; }
        .stMetric { background-color: #111827; padding: 20px; border-radius: 8px; border: 1px solid #1E293B; }
        div[data-testid="stSidebarUserContent"] { background-color: #0F172A; }
    </style>
""", unsafe_allow_html=True)

# 2. Header Architecture
st.title("🛡️ EcoGuard: Moogsoft Predictive Incident Forecaster")
st.markdown("<p style='color: #94A3B8; font-size: 1.2rem;'>Powered by AI/ML Engine • Enterprise Infrastructure Risk Mitigation Dashboard</p>", unsafe_allow_html=True)
st.write("---")

# 3. Cache Loader
@st.cache_resource
def load_artifacts():
    model = joblib.load("moogsoft_forecaster_model.pkl")
    df = pd.read_csv("moogsoft_historical_data.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return model, df

model, df = load_artifacts()

# 4. Sidebar Panel Configuration
st.sidebar.markdown("### 🎛️ Control Panel")
threshold = st.sidebar.slider("Critical Alert Threshold (Incident/Hr)", min_value=30, max_value=100, value=50)
st.sidebar.write("---")
st.sidebar.markdown("### 💡 Project Insight")
st.sidebar.info("This system analyzes core telemetry metrics processed by Moogsoft algorithms to forecast server stability matrices 24 hours into the future.")

# 5. Pipeline Logic (Next 24 Hours)
last_row = df.iloc[-1]
last_timestamp = last_row['timestamp']
current_lag1 = last_row['alert_count']
current_lag2 = df.iloc[-2]['alert_count']

future_preds = []
for i in range(1, 25):
    future_time = last_timestamp + timedelta(hours=i)
    hour = future_time.hour
    day_of_week = future_time.dayofweek
    
    features = pd.DataFrame([[hour, day_of_week, current_lag1, current_lag2]], 
                            columns=['hour', 'day_of_week', 'lag_1', 'lag_2'])
    predicted_alerts = model.predict(features)[0]
    
    future_preds.append({
        "Timestamp": future_time,
        "Predicted Alerts": round(predicted_alerts, 1)
    })
    current_lag2 = current_lag1
    current_lag1 = predicted_alerts

future_df = pd.DataFrame(future_preds)

# 6. Executive Metrics Layout
max_predicted = future_df['Predicted Alerts'].max()
peak_time = future_df.loc[future_df['Predicted Alerts'].idxmax(), 'Timestamp']

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="System Operations Status", value="OPTIMAL", delta="0 Latent Faults")
with col2:
    st.metric(label="Predicted Peak Incident Velocity", value=f"{max_predicted} / hr")
with col3:
    if max_predicted >= threshold:
        st.markdown(f"<div style='background-color:#7F1D1D; padding:18px; border-radius:8px; border:1px solid #F87171; color:#FCA5A5;'><strong>🚨 BREACH DETECTED:</strong> High risk expected at {peak_time.strftime('%H:%M')}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='background-color:#064E3B; padding:18px; border-radius:8px; border:1px solid #34D399; color:#A7F3D0;'><strong>✅ STATUS SECURE:</strong> Infrastructure operating within safe boundaries.</div>", unsafe_allow_html=True)

st.write("---")

# 7. Enterprise Visualization
st.subheader("📊 Predictive Operational Horizon (Next 24 Hours)")
fig = px.line(future_df, x="Timestamp", y="Predicted Alerts", 
              labels={"Predicted Alerts": "Alerts Frequency / Hour"},
              template="plotly_dark")

fig.update_traces(line_color='#38BDF8', line_width=3)
fig.add_hline(y=threshold, line_dash="dash", line_color="#EF4444", annotation_text="Critical Operational Limit")

# FIXED PROPERTIES HERE:
fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#1E293B')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#1E293B')

st.plotly_chart(fig, use_container_width=True)
