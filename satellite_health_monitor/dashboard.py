import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time

from data_simulation import generate_telemetry, inject_faults
from feature_engineering import engineer_features
from anomaly_detection import detect_anomalies
from fault_management import classify_fault, compute_health_index

# -------------------------------------------------------
# PAGE CONFIG (Responsive)
# -------------------------------------------------------

st.set_page_config(
    page_title="Orbital Sentinel | Mission Control",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="auto"
)

# -------------------------------------------------------
# AUTO REFRESH (Cloud Safe)
# -------------------------------------------------------

if "refresh" not in st.session_state:
    st.session_state.refresh = 0

st.session_state.refresh += 1

# Refresh every 3 seconds after mission start
def auto_refresh():
    time.sleep(3)
    st.rerun()

# -------------------------------------------------------
# CLEAN HEADER (Hide footer only)
# -------------------------------------------------------

st.markdown("""
<style>
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# RESPONSIVE PREMIUM UI
# -------------------------------------------------------

st.markdown("""
<style>

/* Responsive padding */
.stApp {
    background: radial-gradient(circle at top, #0f172a, #020617);
    color: white;
    padding-left: 1rem;
    padding-right: 1rem;
}

/* Mobile adjustments */
@media (max-width: 768px) {
    .stApp {
        padding-left: 0.5rem;
        padding-right: 0.5rem;
    }
}

/* Metric Cards */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(14px);
    padding: 18px;
    border-radius: 18px;
    border: 1px solid rgba(56,189,248,0.18);
    box-shadow: 0 0 20px rgba(56,189,248,0.08);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#020617,#0f172a);
}

/* Status badges */
.status-badge {
    padding: 8px 16px;
    border-radius: 30px;
    font-weight: 700;
    text-align: center;
}

.nominal {
    background: #10ff8c;
    color: black;
}

.warning {
    background: #ffcc00;
    color: black;
}

.critical {
    background: #ff4d4d;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# HEADER
# -------------------------------------------------------

st.markdown("## 🚀 ORBITAL SENTINEL COMMAND CENTER")
st.caption("AI-Driven Satellite Telemetry Intelligence & Predictive Fault Monitoring System")
st.divider()

# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------

st.sidebar.header("🛰 Mission Configuration")

num_samples = st.sidebar.slider("Telemetry Frames", 200, 3000, 800)
num_faults = st.sidebar.slider("Injected Fault Events", 1, 50, 15)
contamination = st.sidebar.slider("Detection Sensitivity", 0.01, 0.2, 0.05)

run_button = st.sidebar.button("🚀 Initialize Mission")

# -------------------------------------------------------
# SESSION CONTROL
# -------------------------------------------------------

if "mission_started" not in st.session_state:
    st.session_state.mission_started = False

if run_button:
    st.session_state.mission_started = True

if not st.session_state.mission_started:
    st.info("Click **Initialize Mission** to start telemetry.")
    st.stop()

# -------------------------------------------------------
# DATA GENERATION
# -------------------------------------------------------

with st.spinner("Running Telemetry & AI Engine..."):
    data = generate_telemetry(num_samples)
    data = inject_faults(data, num_faults)
    data = engineer_features(data)
    data = detect_anomalies(data, contamination)
    data["fault_type"] = data.apply(classify_fault, axis=1)
    data = compute_health_index(data)

# -------------------------------------------------------
# SYSTEM STATUS
# -------------------------------------------------------

total_anomalies = len(data[data["anomaly"] == -1])
avg_health = round(data["health_index"].mean(), 2)

if avg_health > 85:
    status_text = "SYSTEM NOMINAL"
    status_class = "nominal"
elif avg_health > 60:
    status_text = "SYSTEM WARNING"
    status_class = "warning"
else:
    status_text = "SYSTEM CRITICAL"
    status_class = "critical"

# -------------------------------------------------------
# KPI DASHBOARD (Responsive)
# -------------------------------------------------------

st.subheader("📊 Mission Overview")

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

col1.metric("Telemetry Frames", len(data))
col2.metric("Anomaly Events", total_anomalies)
col3.metric("Health Index", f"{avg_health}%")
col4.markdown(
    f'<div class="status-badge {status_class}">{status_text}</div>',
    unsafe_allow_html=True
)

st.divider()

# -------------------------------------------------------
# HEALTH GAUGE
# -------------------------------------------------------

st.subheader("🧠 Global Health Indicator")

gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=avg_health,
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "#38bdf8"},
        'steps': [
            {'range': [0, 60], 'color': "#7f1d1d"},
            {'range': [60, 85], 'color': "#78350f"},
            {'range': [85, 100], 'color': "#064e3b"}
        ],
    }
))

gauge.update_layout(height=300)
st.plotly_chart(gauge, use_container_width=True)

st.divider()

# -------------------------------------------------------
# TELEMETRY EXPLORER
# -------------------------------------------------------

st.subheader("📡 Subsystem Telemetry")

param_options = {
    "Health Index": "health_index",
    "Temperature": "temperature",
    "Voltage": "voltage",
    "Current": "current"
}

selected_label = st.selectbox("Select Parameter", list(param_options.keys()))
selected_column = param_options[selected_label]

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=data["time"],
    y=data[selected_column],
    mode="lines",
    name=selected_label
))

anomaly_points = data[data["anomaly"] == -1]

fig.add_trace(go.Scatter(
    x=anomaly_points["time"],
    y=anomaly_points[selected_column],
    mode="markers",
    name="Anomaly"
))

fig.update_layout(
    template="plotly_dark",
    height=420,
    margin=dict(l=10, r=10, t=30, b=10)
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# -------------------------------------------------------
# FAULT CONSOLE
# -------------------------------------------------------

st.subheader("🚨 Fault Console")

anomalies = data[data["anomaly"] == -1]

if len(anomalies) > 0:
    with st.expander("View Fault Log", expanded=True):
        st.dataframe(anomalies, use_container_width=True)

    st.download_button(
        "Download Fault Log",
        anomalies.to_csv(index=False),
        file_name="orbital_fault_log.csv"
    )
else:
    st.success("All systems stable.")

# -------------------------------------------------------
# AUTO REFRESH LOOP (Cloud Safe)
# -------------------------------------------------------

auto_refresh()