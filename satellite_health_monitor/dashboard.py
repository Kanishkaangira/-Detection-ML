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
# PAGE CONFIG
# -------------------------------------------------------

st.set_page_config(
    page_title="Orbital Sentinel | Mission Control",
    page_icon="🛰️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------
# PREMIUM NASA STYLE UI
# -------------------------------------------------------

st.markdown("""
<style>

/* ===============================
   CORE BACKGROUND
================================ */

.stApp {
    background: radial-gradient(circle at top, #0f172a, #020617);
    color: white;
    padding-left: 3rem;
    padding-right: 3rem;
}

/* Smooth global animation */
* {
    transition: all 0.35s ease;
}

/* ===============================
   GLASSMORPH DASHBOARD CARDS
================================ */

[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(14px);
    padding: 22px;
    border-radius: 22px;
    border: 1px solid rgba(56,189,248,0.18);
    box-shadow: 0 0 25px rgba(56,189,248,0.08);
}

[data-testid="stMetric"]:hover {
    transform: translateY(-4px);
}

/* ===============================
   SIDEBAR MISSION CONSOLE
================================ */

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#020617,#0f172a);
    border-right: 1px solid rgba(56,189,248,0.15);
}

.sidebar-title {
    font-size: 20px;
    font-weight: 700;
    color: #38bdf8;
}

.sidebar-section {
    margin-top: 22px;
    font-weight: 600;
    color: #00f5ff;
}

/* ===============================
   STATUS BADGES
================================ */

.status-badge {
    padding: 10px 22px;
    border-radius: 40px;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-align: center;
}

.nominal {
    background: linear-gradient(45deg,#064e3b,#10ff8c);
    color: black;
}

.warning {
    background: linear-gradient(45deg,#78350f,#ffcc00);
    color: black;
}

.critical {
    background: linear-gradient(45deg,#7f1d1d,#ff4d4d);
    color: white;
}

/* ===============================
   TELEMETRY GLOW GRAPH
================================ */

.plot-container {
    animation: telemetryGlow 4s infinite ease-in-out;
    border-radius: 18px;
}

@keyframes telemetryGlow {
    0% { box-shadow: 0 0 6px rgba(56,189,248,0.2); }
    50% { box-shadow: 0 0 18px rgba(56,189,248,0.5); }
    100% { box-shadow: 0 0 6px rgba(56,189,248,0.2); }
}

/* ===============================
   SELECT BOX MODERN STYLE
================================ */

div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 14px !important;
    border: 1px solid rgba(56,189,248,0.25) !important;
    backdrop-filter: blur(10px);
}

div[data-baseweb="select"] span {
    color: white !important;
}

/* ===============================
   LIGHT MODE CLEAN UI
================================ */

@media (prefers-color-scheme: light) {

    .stApp {
        background: #f7fafc !important;
        color: #111827 !important;
    }

    h1,h2,h3,h4,h5,h6,p,span,label {
        color: #111827 !important;
    }

    section[data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid #e5e7eb;
    }

    div[data-baseweb="select"] > div {
        background: white !important;
        border: 1px solid #d1d5db !important;
    }

    div[data-baseweb="select"] span {
        color: black !important;
    }
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
# SIDEBAR CONFIGURATION
# -------------------------------------------------------

st.sidebar.markdown('<div class="sidebar-title">🛰 Mission Configuration Console</div>', unsafe_allow_html=True)
st.sidebar.markdown("---")

st.sidebar.markdown('<div class="sidebar-section">📡 Telemetry Simulation</div>', unsafe_allow_html=True)

num_samples = st.sidebar.slider("Telemetry Frames", 200, 3000, 800)
num_faults = st.sidebar.slider("Injected Fault Events", 1, 50, 15)

st.sidebar.markdown('<div class="sidebar-section">🤖 AI Detection Engine</div>', unsafe_allow_html=True)

contamination = st.sidebar.slider("Detection Sensitivity", 0.01, 0.2, 0.05)

run_button = st.sidebar.button("🚀 Initialize Mission Systems")

# -------------------------------------------------------
# SESSION CONTROL
# -------------------------------------------------------

if "mission_started" not in st.session_state:
    st.session_state.mission_started = False

if run_button:
    st.session_state.mission_started = True

if not st.session_state.mission_started:
    st.info("Click **Initialize Mission Systems** to start live telemetry stream.")
    st.stop()

# -------------------------------------------------------
# LIVE STREAMING LOOP
# -------------------------------------------------------

while True:

    with st.spinner("Deploying Telemetry & AI Engine..."):

        data = generate_telemetry(num_samples)
        data = inject_faults(data, num_faults)
        data = engineer_features(data)
        data = detect_anomalies(data, contamination)

        data["fault_type"] = data.apply(classify_fault, axis=1)
        data = compute_health_index(data)

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
    # KPI DASHBOARD
    # -------------------------------------------------------

    st.subheader("📊 Mission Overview")

    col1, col2, col3, col4 = st.columns(4)

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

    st.subheader("📡 Subsystem Telemetry Explorer")

    param_options = {
        "🧠 Health Index": "health_index",
        "🌡 Temperature": "temperature",
        "⚡ Voltage": "voltage",
        "🔌 Current": "current"
    }

    selected_label = st.selectbox(
        "Select Subsystem Parameter",
        list(param_options.keys())
    )

    selected_column = param_options[selected_label]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data["time"],
        y=data[selected_column],
        mode="lines",
        name=selected_label,
        line=dict(width=2),
        opacity=0.85
    ))

    anomaly_points = data[data["anomaly"] == -1]

    fig.add_trace(go.Scatter(
        x=anomaly_points["time"],
        y=anomaly_points[selected_column],
        mode="markers",
        name="Anomaly Event",
        marker=dict(size=7, symbol="circle", opacity=0.9)
    ))

    fig.update_layout(
        template="plotly_dark",
        height=420,
        margin=dict(l=25, r=25, t=40, b=25),
        xaxis_title="Mission Timeline",
        yaxis_title=selected_label,
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # -------------------------------------------------------
    # FAULT CONSOLE
    # -------------------------------------------------------

    st.subheader("🚨 Fault Intelligence Console")

    anomalies = data[data["anomaly"] == -1]

    if len(anomalies) > 0:
        with st.expander("View Detailed Fault Log", expanded=True):
            st.dataframe(anomalies, use_container_width=True)

        st.download_button(
            "⬇ Export Fault Log",
            anomalies.to_csv(index=False),
            file_name="orbital_fault_log.csv"
        )
    else:
        st.success("All subsystems stable. No anomalies detected.")

    time.sleep(3)
    st.rerun()