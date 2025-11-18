import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
import json
import os
import httpx
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

# ---------------------------
# Streamlit Page Config
# ---------------------------
st.set_page_config(
    page_title="IoT Device Dashboard",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("AI Agent for IoT Device Anomaly Explanation")
st.write("This dashboard simulates IoT sensor readings and explains anomalies using an LLM.")

st.sidebar.header("Configuration")

TEMP_THRESHOLD = st.sidebar.slider("Temperature Threshold", 20, 80, 45)
GAS_THRESHOLD = st.sidebar.slider("Gas Threshold", 20, 150, 70)
UPDATE_INTERVAL = st.sidebar.slider("Update Interval (Seconds)", 1, 10, 3)

LOG_FILE = "sensor_data.json"

# ---------------------------
# Initialize Log File
# ---------------------------
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        json.dump([], f)

# ---------------------------
# Initialize Session State
# ---------------------------
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=["Timestamp", "Temperature", "Gas"])

col1, col2 = st.columns(2)
temp_placeholder = col1.empty()
gas_placeholder = col2.empty()

chart_placeholder = st.empty()
alert_placeholder = st.empty()
reason_placeholder = st.empty()

# ---------------------------
# LLM Backend Setup
# ---------------------------
client = httpx.Client(verify=False)

llm = ChatOpenAI(
    base_url="https://genailab.tcs.in",
    model="azure/genailab-mass-gpt-4o",
    api_key="abcdsdfsfsdfd",   # <---- Replace with your real key
    http_client=client,
    temperature=0.2
)

def get_llm_explanation(temp, gas, alert_msg):
    """Invoke GPT model to explain anomaly & give recommendations."""
    prompt = f"""
You are an IoT anomaly analysis assistant.

Sensor Readings:
- Temperature: {temp:.2f} °C
- Gas Level: {gas:.2f} PPM

Alert Triggered:
"{alert_msg}"

Provide a detailed explanation:
- Summary of what is happening
- Root Cause Analysis
- Recommended Actions
- Estimated Impact (Low/Medium/High)
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


# ---------------------------
# Main Simulation Loop
# ---------------------------
run = st.checkbox("Start Sensor Simulation")

if run:
    while True:
        # Generate random sensor values
        temp = np.random.normal(30, 5)
        gas = np.random.normal(50, 10)
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Add to session data
        new_row = pd.DataFrame([[timestamp, temp, gas]], columns=["Timestamp", "Temperature", "Gas"])
        st.session_state.data = pd.concat([st.session_state.data, new_row]).tail(50)

        # Log data to file
        with open(LOG_FILE, "r+") as f:
            data = json.load(f)
            data.append({
                "timestamp": timestamp,
                "temperature": round(float(temp), 2),
                "gas": round(float(gas), 2)
            })

            f.seek(0)
            json.dump(data, f, indent=4)

        # ---------------------------
        # Update UI with metrics
        # ---------------------------
        temp_delta = temp - TEMP_THRESHOLD
        gas_delta = gas - GAS_THRESHOLD

        temp_placeholder.metric(
            label="Temperature (°C)",
            value=f"{temp:.2f}",
            delta=f"{temp_delta:+.2f}",
        )

        gas_placeholder.metric(
            label="Gas (PPM)",
            value=f"{gas:.2f}",
            delta=f"{gas_delta:+.2f}",
        )

        chart_placeholder.line_chart(
            st.session_state.data.set_index("Timestamp"),
            height=300,
            use_container_width=True
        )

        # ---------------------------
        # Alert Logic
        # ---------------------------
        if temp > TEMP_THRESHOLD and gas > GAS_THRESHOLD:
            alert_msg = f"CRITICAL: Temperature {temp:.2f} °C AND Gas {gas:.2f} PPM exceed thresholds!"
        elif temp > TEMP_THRESHOLD:
            alert_msg = f"Temperature Alert: {temp:.2f} °C exceeds {TEMP_THRESHOLD} °C"
        elif gas > GAS_THRESHOLD:
            alert_msg = f"Gas Alert: {gas:.2f} PPM exceeds {GAS_THRESHOLD} PPM"
        else:
            alert_msg = "All readings Normal."

        # Display alerts
        if "CRITICAL" in alert_msg:
            alert_placeholder.error(alert_msg)
        elif "Alert" in alert_msg:
            alert_placeholder.warning(alert_msg)
        else:
            alert_placeholder.success(alert_msg)

        # ---------------------------
        # LLM EXPLANATION (only when anomaly occurs)
        # ---------------------------
        if "Normal" not in alert_msg:
            explanation = get_llm_explanation(temp, gas, alert_msg)
            reason_placeholder.info(explanation)
        else:
            reason_placeholder.empty()

        time.sleep(UPDATE_INTERVAL)

else:
    st.info("Click the checkbox above to start the IoT sensor simulation.")
