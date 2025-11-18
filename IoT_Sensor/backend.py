from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import httpx
import os

def generate_reason(temp, gas, temp_threshold, gas_threshold):
	if temp > temp_threshold:
		temp_status = "high"
	elif abs(temp - temp_threshold) <=3:
		temp_status= "near threshold"
	else:
		temp_status = "Normal"
		
	if gas > gas_threshold:
		gas_status = "high"
	elif abs(gas - gas_threshold) <=5:
		gas_status= "near threshold"
	else:
		gas_status = "Normal"
		
	client = httpx.Client(verify=False)
	
	llm = ChatOpenAI(
		base_url="https://genailab.tcs.in",
		model = "azure/genailab-mass-gpt-4o",
		api_key="abcdsdfsfsdfd",
		http_client=client,
		temperature=0.7
	)
	
	prompt = f"""
		You are an IoT diagnostic assistant monitoring an industrial environment.
		Current sensor reading and thresholds:
		= Temperature: {temp} deg C (Threshold: {temp_threshold} deg C) -> Status: {temp_status}
		= gas: {gas} PPM (Threshold: {gas_threshold} PPM) -> Status: {gas_status}
		
		Analyze the data careflly and produce **two clearly separated paragraph**:
		
		1. Cause/Explanation: Describe the likely technical or environmental reason(s) for these readings.
		Mention possible contributing factors such as equipment heat, poor ventilation, sensor caliberation, or normal operation cycles.
		
		2. Recommendations/Corrective Actions: Suggest practical IoT or operational steps that could be taken to stabilize readings,
		improve safety or prevent future anomalies.
	"""
	
	response = llm.invoke([HumanMessage(content=prompt)])
	return response.content.strip()
	
print(generate_reason(50,40,60,40))

