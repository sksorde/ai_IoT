"""
iot_rag_app.py

Single-file Streamlit app that:
- Builds/loads a Chroma vector store from `knowledge_base/` (PDF/TXT)
- Simulates IoT sensor data (temperature, gas)
- Detects alerts based on thresholds
- Performs Retrieval-Augmented Generation (RAG) using the vector store
- Calls GPT (ChatOpenAI) to explain anomalies and recommend actions
- Shows retrieved chunks in an expandable RAG UI panel

Before running:
- Create `knowledge_base/` and add PDFs/TXT documents to it
- Replace API keys / endpoints below as needed
- Install required packages (streamlit, pandas, numpy, httpx, langchain, langchain-community, chromadb)
"""

import os
import json
import time
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import httpx

# Attempt to import langchain modules; if not present, provide guidance in the UI.
try:
    # embeddings + llm client
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    # document loaders & text splitter & chroma vectorstore
    from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain.schema import HumanMessage

    LANGCHAIN_AVAILABLE = True
except Exception as e:
    LANGCHAIN_AVAILABLE = False
    _IMPORT_ERROR = str(e)

# -------------------------
# Configuration (Edit these)
# -------------------------
# Replace these values with your own LLM endpoint/keys
API_KEY = "abcdsdfsfsdfd"  # <-- Replace with your real api key
BASE_URL = "https://genailab.tcs.in"  # <-- Replace if needed (or keep)
MODEL_NAME = "azure/genailab-mass-gpt-4o"  # <- your model id

# Where your docs live (create this folder and drop PDFs/TXT)
KB_FOLDER = "knowledge_base"
CHROMA_PERSIST_DIR = "chromadb_store"
LOG_FILE = "sensor_log.json"

# -------------------------
# Streamlit UI setup
# -------------------------
st.set_page_config(page_title="IoT RAG Anomaly Dashboard", layout="wide", initial_sidebar_state="expanded")
st.title("🤖 IoT RAG Anomaly Dashboard (Single-file)")
st.write(
    "Simulates IoT sensor data, detects anomalies, retrieves relevant docs from a knowledge base (RAG), "
    "and asks an LLM to explain the anomaly and recommend actions."
)

if not LANGCHAIN_AVAILABLE:
    st.error("LangChain or its components are not available in the environment.")
    st.write("Import error:", _IMPORT_ERROR)
    st.stop()

# Sidebar configuration controls
st.sidebar.header("Configuration")
TEMP_THRESHOLD = st.sidebar.slider("Temperature Threshold (°C)", min_value=20, max_value=120, value=45)
GAS_THRESHOLD = st.sidebar.slider("Gas Threshold (PPM)", min_value=10, max_value=500, value=70)
UPDATE_INTERVAL = st.sidebar.slider("Update Interval (sec)", min_value=1, max_value=10, value=3)

st.sidebar.markdown("---")
st.sidebar.header("Vector DB / RAG")
st.sidebar.write(f"Knowledge base folder: `{KB_FOLDER}`")
rebuild = st.sidebar.button("Rebuild Vector Store (force)")

# Show basic instructions if KB folder missing
if not os.path.exists(KB_FOLDER):
    st.sidebar.warning(f"Folder `{KB_FOLDER}` not found. Create it and add PDF/TXT docs before building the vector store.")

# -------------------------
# Utility: Build vector store
# -------------------------
@st.cache_data(show_spinner=False)
def build_vector_store(kb_folder=KB_FOLDER, persist_dir=CHROMA_PERSIST_DIR, api_key=API_KEY, base_url=BASE_URL):
    """
    Loads PDF/TXT docs from kb_folder, splits into chunks, creates embeddings, and stores in Chroma.
    This function is cached so it's not re-run unnecessarily by Streamlit.
    """
    # Validate folder
    if not os.path.exists(kb_folder):
        raise FileNotFoundError(f"Knowledge base folder '{kb_folder}' does not exist.")

    # Loaders: PDFs + txt
    pdf_loader = DirectoryLoader(kb_folder, glob="**/*.pdf", loader_cls=PyPDFLoader)
    txt_loader = DirectoryLoader(kb_folder, glob="**/*.txt", loader_cls=TextLoader)

    st.info("Loading documents from knowledge base...")
    documents = []
    documents += pdf_loader.load()
    documents += txt_loader.load()

    if len(documents) == 0:
        raise RuntimeError("No documents found in the knowledge_base folder. Add PDFs or TXT files.")

    st.info(f"Loaded {len(documents)} documents. Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    st.info(f"Created {len(chunks)} chunks.")

    st.info("Creating embeddings (this may take a while)...")
    embeddings = OpenAIEmbeddings(base_url=base_url, api_key=api_key)

    st.info("Building Chroma vector store and persisting to disk...")
    db = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_dir)
    db.persist()
    st.success("Vector store built and persisted.")
    return db

# -------------------------
# Load or Build Vector DB
# -------------------------
vector_db = None
needs_build = rebuild or (not os.path.exists(CHROMA_PERSIST_DIR))

# If persist dir exists, try loading; else build
if needs_build:
    # Build vector store (shows progress in UI)
    try:
        with st.spinner("Building vector store (this runs once)..."):
            vector_db = build_vector_store()
    except Exception as e:
        st.error("Failed to build vector store: " + str(e))
        st.stop()
else:
    # Load embeddings + chroma from disk
    try:
        embeddings = OpenAIEmbeddings(base_url=BASE_URL, api_key=API_KEY)
        vector_db = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
        st.sidebar.success("Loaded existing vector store.")
    except Exception as e:
        st.sidebar.error("Failed to load vector store: " + str(e))
        # Offer to rebuild
        if st.sidebar.button("Try to rebuild now"):
            try:
                vector_db = build_vector_store()
            except Exception as e2:
                st.error("Rebuild failed: " + str(e2))
                st.stop()

# -------------------------
# Initialize LLM client
# -------------------------
# Using httpx client (verify=False used earlier; keep consistent with your environment)
httpx_client = httpx.Client(verify=False)

llm = ChatOpenAI(
    base_url=BASE_URL,
    model=MODEL_NAME,
    api_key=API_KEY,
    http_client=httpx_client,
    temperature=0.15
)

# -------------------------
# Helper: RAG-enabled LLM explanation function
# -------------------------
def get_rag_explanation(temp: float, gas: float, alert_msg: str, k: int = 3):
    """
    1) Builds a short query from the sensor data
    2) retrieves top-k chunks from the vector DB
    3) composes a prompt that includes the retrieved chunks + sensor readings
    4) calls the LLM and returns (explanation_text, retrieved_docs)
    """
    # Make sure vector_db is ready
    if vector_db is None:
        raise RuntimeError("Vector DB is not initialized.")

    query = f"Temperature={temp:.2f} °C, Gas={gas:.2f} PPM, Alert={alert_msg}"
    docs = vector_db.similarity_search(query, k=k)

    retrieved_text = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
You are an IoT anomaly diagnosis assistant with access to technical documentation.

Relevant retrieved documentation (from the company's knowledge base):
{retrieved_text}

Current sensor readings:
- Temperature: {temp:.2f} °C
- Gas Level: {gas:.2f} PPM

Alert Triggered:
"{alert_msg}"

Using the retrieved documentation and sensor readings, produce a clear and actionable analysis:
1) Summary (1-2 sentences)
2) Root Cause Analysis (link to the retrieved docs when relevant)
3) Recommended Corrective Actions (prioritized)
4) Estimated Impact (Low / Medium / High)
5) Short citations: for each major claim include the chunk index (1..{len(docs)}) that supports it.

Be concise but factual.
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content, docs

# -------------------------
# Sensor Simulation & UI components
# -------------------------
# Session state for retaining data across reruns
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=["Timestamp", "Temperature", "Gas"])

# Ensure log file exists
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        json.dump([], f)

# Top row: metrics + controls
colA, colB, colC = st.columns([2, 2, 1])
with colA:
    temp_metric = st.empty()
with colB:
    gas_metric = st.empty()
with colC:
    run_sim = st.checkbox("Start Simulation", value=False)

chart_area = st.container()
alert_area = st.empty()
rag_area = st.container()
explain_area = st.container()

# Control: allow manual single-step update (useful when not running continuous loop)
do_step = st.button("Generate Single Reading")

# Main simulation loop (or single-step)
def simulate_once():
    # generate simulated readings
    temp = float(np.random.normal(30, 5))
    gas = float(np.random.normal(50, 10))
    timestamp = datetime.now().strftime("%H:%M:%S")

    # update session data (keep last 100)
    new_row = pd.DataFrame([[timestamp, temp, gas]], columns=["Timestamp", "Temperature", "Gas"])
    st.session_state.data = pd.concat([st.session_state.data, new_row], ignore_index=True).tail(100)

    # append to log file
    try:
        with open(LOG_FILE, "r+") as f:
            try:
                data = json.load(f)
            except Exception:
                data = []
            data.append({"timestamp": timestamp, "temperature": round(temp, 2), "gas": round(gas, 2)})
            f.seek(0)
            json.dump(data, f, indent=2)
    except Exception as e:
        st.warning(f"Failed to write to log: {e}")

    # update metrics UI
    temp_metric.metric(label="🌡 Temperature (°C)", value=f"{temp:.2f}")
    gas_metric.metric(label="🧪 Gas (PPM)", value=f"{gas:.2f}")

    # chart
    with chart_area:
        st.subheader("Sensor Readings (last 100)")
        st.line_chart(st.session_state.data.set_index("Timestamp"))

    # alerting
    if (temp > TEMP_THRESHOLD) and (gas > GAS_THRESHOLD):
        alert_msg = f"CRITICAL: Temperature {temp:.2f} °C AND Gas {gas:.2f} PPM exceed thresholds!"
    elif temp > TEMP_THRESHOLD:
        alert_msg = f"Temperature Alert: {temp:.2f} °C exceeds {TEMP_THRESHOLD} °C"
    elif gas > GAS_THRESHOLD:
        alert_msg = f"Gas Alert: {gas:.2f} PPM exceeds {GAS_THRESHOLD} PPM"
    else:
        alert_msg = "All readings normal."

    # show alert
    if "CRITICAL" in alert_msg:
        alert_area.error(alert_msg)
    elif "Alert" in alert_msg:
        alert_area.warning(alert_msg)
    else:
        alert_area.success(alert_msg)

    # If alert, run RAG + LLM
    if "normal" not in alert_msg.lower():
        with st.spinner("Retrieving relevant docs and getting LLM explanation..."):
            try:
                explanation, docs = get_rag_explanation(temp, gas, alert_msg, k=3)
            except Exception as e:
                explanation = f"LLM/RAG failed: {e}"
                docs = []

        # Show explanation
        with explain_area:
            st.subheader("LLM Explanation (RAG-backed)")
            st.info(explanation)

        # Show retrieved chunks in RAG panel
        with rag_area:
            with st.expander("📚 Retrieved Knowledge Base Chunks (click to expand)"):
                if len(docs) == 0:
                    st.write("No documents retrieved or vector DB empty.")
                else:
                    for i, doc in enumerate(docs, start=1):
                        st.markdown(f"**Chunk {i}**")
                        # show first 600 characters to keep UI neat, with option to expand
                        st.write(doc.page_content[:1200] + ("..." if len(doc.page_content) > 1200 else ""))
                        st.markdown("---")
    else:
        # Clear explanation and RAG panels if everything normal
        explain_area.empty()
        rag_area.empty()

    return temp, gas, alert_msg

# If continuous run checkbox selected, loop until unchecked.
if run_sim:
    # Use a loop with Streamlit rerun-friendly approach.
    # We'll repeatedly call simulate_once() and then sleep. Because Streamlit runs the script top-to-bottom on each user interaction,
    # continuous while True loops can freeze the UI. We keep a simple loop but break gracefully if user unchecks the checkbox.
    st.info("Simulation running. Uncheck 'Start Simulation' to stop.")
    try:
        # We'll run forever until user unchecks — Streamlit reruns this code quickly, so provide a simple loop with sleep.
        while st.session_state.get("run_loop", True):
            simulate_once()
            time.sleep(UPDATE_INTERVAL)
            # check the widget state (re-run will re-evaluate run_sim)
            if not st.session_state.get("run_loop", True):
                break
            # Re-read the run_sim checkbox (can't directly read earlier variable after rerun)
            # Use st.session_state to allow user to stop by toggling the checkbox
            if "Start Simulation" in st.session_state:
                # no-op
                pass
            # We rely on the user unchecking the checkbox to stop the rerun; Streamlit's execution model means this will re-run appropriately.
    except KeyboardInterrupt:
        st.warning("Simulation interrupted.")
else:
    # Single-step mode (button)
    if do_step:
        simulate_once()
    else:
        st.info("Press 'Generate Single Reading' to produce one simulated reading, or check 'Start Simulation' to run continuously.")

# -------------------------
# Helpful notes in footer
# -------------------------
st.markdown("---")
st.markdown("**Notes & Next steps**")
st.markdown(
    "- Put device manuals and SOPs into the `knowledge_base/` folder and rebuild the vector store for RAG to use them.\n"
    "- Replace `API_KEY` and `BASE_URL` with your real credentials.\n"
    "- For production, secure API keys (do NOT hardcode) and use an authenticated vector DB or hosted solution."
)
