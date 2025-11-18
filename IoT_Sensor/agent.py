import pandas as pd
import json
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import httpx
import os

# Load CSVs
telemetry_df = pd.read_csv("telemetry_data.csv")
alert_df = pd.read_csv("alerts_data.csv")

# Normalize timestamps
telemetry_df['timestamp'] = pd.to_datetime(telemetry_df['timestamp'])
alert_df['timestamp'] = pd.to_datetime(alert_df['timestamp'])

# Merge as-of on timestamp and device_id
merged_data = pd.merge_asof(
    alert_df.sort_values("timestamp"),
    telemetry_df.sort_values("timestamp"),
    on="timestamp",
    by="device_id",
    direction="backward"
)

# Initialize LLM
client = httpx.Client(verify=False)
llm = ChatOpenAI(
    base_url="https://genailab.tcs.in",
    model="azure/genailab-mass-gpt-4o",
    api_key="abcdsdfsfsdfd",
    http_client=client,
    temperature=0.7
)

# Priority assignment function
def assign_priority(row):
    msg = row.get("message", "").lower()
    if "critical" in msg or "overheat" in msg or "disconnect" in msg:
        return "High"
    elif "warning" in msg or "unstable" in msg:
        return "Medium"
    else:
        return "Low"

merged_data["priority"] = merged_data.apply(assign_priority, axis=1)

# Generate explanations
explanations = []

for _, row in merged_data.iterrows():

    telemetry = {col: row[col] for col in telemetry_df.columns if col in row}
    alert = {col: row[col] for col in alert_df.columns if col in row}

    prompt = f"""
You are an IoT alert explanation and recommendation assistant.

Telemetry data:
{json.dumps(telemetry, indent=2, default=str)}

Alert details:
{json.dumps(alert, indent=2, default=str)}

Priority: {row["priority"]}

Provide your analysis in this structure:
- Summary:
- Root Cause Analysis:
- Recommended Actions:
- Estimated Impact (Low/Medium/High):
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    explanations.append({
        "device_id": row.get("device_id", "unknown"),
        "timestamp": row["timestamp"],
        "priority": row["priority"],
        "explanation": response.content
    })

# Save output
results_df = pd.DataFrame(explanations)
results_df.to_csv("alert_explanations.csv", index=False)

print("\nIoT alert explanations generated and saved to alert_explanations.csv")
