import pandas as pd
import json
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import httpx
import os


client = httpx.Client(verify=False)
llm = ChatOpenAI(
	base_url="https://genailab.tcs.in",
	model = "azure/genailab-mass-gpt-4o",
	api_key="abcdsdfsfsdfd",
	http_client=client,
	temperature=0.7
)

response = llm.invoke("hi")
print(response)

latest_telemetry = telemetry_df.sort_values("timestamp").iloc[-1].to_dict()
latest_alert = alerts_df.sort_values("timestamp").iloc[-1].to_dict()

"""
telemetry_data = {
	"device_id": "sensor_01",
	"temperature": 90,
	"last_temperature": 65,
	"connectivity": "unstable"
}

alert_log = {
	"alert_type": "temperature Spike",
	"message": "Device heating rapidly"
}
"""

prompt = f"""
	You are an IoT alert explanation and recommendation assistant
	Telemetry:
	{json.dumpts(latest_telemetry, indent=2)}
	Alert:
	{json.dumpts(latest_alert, indent=2)}
	Explain the issue an suggest actions
"""

response = llm.invoke([HumanMessage(content=prompt)])
print("\n IoT Alert Explanation")
print(response.content)