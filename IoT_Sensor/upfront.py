import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
import json
import os

st.set_page_config(
	page_title="IoT Device Dashboard",
	layout="centered",
	initial_si,
	debar_state="expanded"
)

st.title("AI Agent for IoT Device Anomaly Explanation")
st.write("This dashboard simulates IoT sensor readings")

st.sidebar.header("Configuration")

TEMP_THRESHOLD = st.sidebar.slider("Temperature Threshold", 20, 80, 45)
GAS_THRESHOLD = st.sidebar.slider("Gas Threshold", 20, 150, 70)
UPDATE_INTERNAL = st.sidebar.slider("Update Interval (Seconds)", 1, 10, 3)

LOG_FILE="sensor_data.json"

if not os.path.exists(LOG_FILE):
	with open(LOG_FILE, "w") as f:
		json.dump([], f)

if "data" not in st.session_state:
	st.session_state.data = pd.DataFrame(columns=["Timestamp", "Temperature", "Gas"])
	
col1, col2 = st.columns(2)
temp_placeholder = col1.empty()
gas_placeholder = col2.empty()

chart_placeholder = st.empty()
alert_placeholder = st.empty()
reason_placeholder = st.empty()

run = st.checkbox("Start Sensor Simulation")

if run:
	while True:
		temp = np.random.normal(30,5)
		gas = np.random.normal(50, 10)
		timestamp = datetime.now().strftime("%H:%M:%S")
		
		new_row = pd.DataFrame([[timestamp, temp, gas]], columns=["Timestamp", "Temperature", "Gas"])
		st.session_state.data = pd.concat([st.session_state.data, new_row]).tail(50)
		
		with open(LOG_FILE, "r+") as f:
			data=json.load(f)
			data.append({
				"timestamp": timestamp,
				"temperatures": round(float(temp), 2),
				"gas": round(float(gas), 2)
			})
			
			f.seek(0)
			json.dump(data, f, indent=4)
			
		temp_delta = temp-TEMP_THRESHOLD
		temp_placeholder.metric(label="Temperature in Cel", 
								value=f"{temp:.2f}, delta=f"{temp_delta:+.2f},
								delta_color="inverse" if temp_delta < TEMP_THRESHOLD else "Normal")
								
		gas_delta = gas-GAS_THRESHOLD
		gas_placeholder.metric(label="Gas in PPM", 
								value=f"{gas:.2f}, delta=f"{gas_delta:+.2f},
								delta_color="inverse" if gas_delta < GAS_THRESHOLD else "Normal")
								
		chart_placeholder.line_chart(
			st.session_state.data.set_index("Timestamp"),
			height=300,
			use_container_width=True
		)
		
		alert_msg = ""
		if temp > TEMP_THRESHOLD and gas > GAS_THRESHOLD:
			alert_msg = f"CRITICAL: Both Temperature ({temp:.2f}deg C) & Gas ({gas:.2f}) PPM"
		elif temp > TEMP_THRESHOLD:
			alert_msg = f"Temperature Alert: ({temp:.2f}deg C) exceed {TEMP_THRESHOLD}"
		elif gas > GAS_THRESHOLD:
			alert_msg = f"Gas Alert: Gas ({gas:.2f}) PPM exceeds {GAS_THRESHOLD} PPM"
		else:
			alert_msg = "All readings Normal!!"
			
		if "CRITICAL" in alert_msg:
			alert_placeholder.error(alert_msg)
		elif "Alert" in alert_msg:
			alert_placeholder.warning(alert_msg)
		else:
			alert_placeholder.success(alert_msg)
			
		time.sleep(UPDATE_INTERNAL)
		
else:
	st.info("Click the checkbox above to start the IoT sensor simulation.")