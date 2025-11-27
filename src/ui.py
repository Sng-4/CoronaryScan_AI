import streamlit as st
import requests
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# --- CONFIGURATION & SECRETS ---
st.set_page_config(
    page_title="ARCADE AI Dashboard",
    page_icon="🫀",
    layout="wide"
)

# Robust logic to find the API URL
# 1. Check Streamlit Cloud Secrets (Production)
# 2. Check Environment Variables (Docker)
# 3. Default to Localhost (Local Testing)
try:
    if "API_URL" in st.secrets:
        API_URL = st.secrets["API_URL"]
    else:
        API_URL = os.getenv("API_URL", "http://localhost:8000")
except FileNotFoundError:
    # Happens locally if no .streamlit/secrets.toml exists
    API_URL = os.getenv("API_URL", "http://localhost:8000")

# Strip trailing slash to prevent double-slash errors (e.g. .com//predict)
API_URL = API_URL.rstrip("/")

# --- SIDEBAR & DIAGNOSTICS ---
st.sidebar.title("Navigation")
st.sidebar.info(f"🔌 Connected to: `{API_URL}`") # Debug info so you can SEE where it's pointing
page = st.sidebar.radio("Go to", ["Dashboard & Predictions", "System Health", "Retraining Pipeline"])

# --- HELPER FUNCTIONS ---
def check_api_health():
    """Pings the API to check up-time and model status."""
    try:
        start_time = time.time()
        response = requests.get(f"{API_URL}/")
        latency = (time.time() - start_time) * 1000 # ms
        if response.status_code == 200:
            return True, response.json(), latency
    except requests.exceptions.ConnectionError:
        return False, None, 0
    except Exception as e:
        st.sidebar.error(f"Health Check Error: {e}")
        return False, None, 0
    return False, None, 0

# --- PAGE 1: DASHBOARD & PREDICTIONS ---
if page == "Dashboard & Predictions":
    st.title("🫀 Coronary Stenosis Detection")
    st.markdown("Upload X-ray Angiography images to detect arterial blockages.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Upload X-Ray")
        uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

        if uploaded_file is not None:
            # Display Image
            image = Image.open(uploaded_file).convert('L')
            # FIX: Replaced deprecated 'use_column_width' with 'use_container_width'
            st.image(image, caption="Input Angiogram", use_container_width=True)
            
            # Prediction Button
            if st.button("Analyze Image", type="primary"):
                with st.spinner("Running EfficientNet Model..."):
                    try:
                        # Reset pointer to start of file
                        uploaded_file.seek(0)
                        files = {"file": uploaded_file.getvalue()}
                        
                        response = requests.post(f"{API_URL}/predict", files=files)
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            # Store result in session state
                            st.session_state['last_result'] = result
                            st.session_state['last_image'] = image
                        else:
                            st.error(f"Prediction Failed: {response.text}")
                            
                    except requests.exceptions.ConnectionError:
                        st.error(f"❌ Connection Error. Tried connecting to: {API_URL}")
                        st.warning("Check your API_URL secret in Streamlit Cloud settings.")

    with col2:
        st.subheader("2. Diagnostics & Visualization")
        
        if 'last_result' in st.session_state and uploaded_file is not None:
            res = st.session_state['last_result']
            
            # --- Result Metrics ---
            diag_color = "red" if "Stenosis" in res['diagnosis'] else "green"
            st.markdown(f"### Diagnosis: :{diag_color}[{res['diagnosis']}]")
            
            m1, m2 = st.columns(2)
            m1.metric("Confidence Score", f"{res['confidence']:.2%}")
            m2.metric("Raw Output", f"{res['raw_score']:.4f}")
            
            # Show TTA details if available
            if 'tta_scores' in res:
                st.caption(f"TTA Analysis: Original ({res['tta_scores']['original']:.2f}) | Flipped ({res['tta_scores']['flipped']:.2f})")
            
            st.divider()
            
            # --- Requirement: Data Visualizations ---
            st.markdown("#### Feature Interpretability")
            
            tab1, tab2 = st.tabs(["Pixel Intensity", "Contrast Map"])
            
            img_array = np.array(st.session_state['last_image'])
            
            with tab1:
                # Pixel Histogram
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.hist(img_array.ravel(), bins=50, color='#4B4B4B', range=[0, 255])
                ax.set_title("Vessel Density Distribution")
                ax.set_xlabel("Pixel Intensity")
                ax.set_ylabel("Frequency")
                st.pyplot(fig)
                st.caption("Lower intensity peaks (left) typically represent dye-filled vessels.")
                
            with tab2:
                # Heatmap
                fig2, ax2 = plt.subplots(figsize=(5, 3))
                im = ax2.imshow(img_array, cmap='inferno')
                plt.colorbar(im)
                ax2.axis('off')
                st.pyplot(fig2)
                st.caption("High contrast regions highlighting vessel boundaries.")

        else:
            st.info("Upload an image and click 'Analyze' to see results.")

# --- PAGE 2: SYSTEM HEALTH (Model Up-time) ---
elif page == "System Health":
    st.title("🖥️ System Status Monitor")
    
    # Live Ping
    is_online, data, latency = check_api_health()
    
    st.subheader("Real-time Metrics")
    kpi1, kpi2, kpi3 = st.columns(3)
    
    if is_online:
        kpi1.metric("API Status", "ONLINE", delta="Active")
        kpi2.metric("Model Loaded", str(data.get("model_status", "Unknown")))
        kpi3.metric("Latency", f"{latency:.2f} ms")
        st.success(f"Connected to ARCADE Service: {data.get('service')}")
    else:
        kpi1.metric("API Status", "OFFLINE", delta="-Down")
        kpi2.metric("Model Loaded", "N/A")
        kpi3.metric("Latency", "∞")
        st.error(f"Cannot connect to API at `{API_URL}`.")

# --- PAGE 3: RETRAINING ---
elif page == "Retraining Pipeline":
    st.title("🔄 Model Retraining")
    st.markdown("Trigger the MLOps pipeline to update the model with new data.")
    
    with st.form("retrain_form"):
        batch_files = st.file_uploader("Upload New Dataset Batch (Images)", accept_multiple_files=True)
        notes = st.text_area("Version Notes", placeholder="e.g. Added 50 new Stenosis cases")
        
        submitted = st.form_submit_button("Trigger Retraining")
        
        if submitted:
            try:
                res = requests.post(f"{API_URL}/retrain")
                if res.status_code == 200:
                    st.success(f"Pipeline Triggered! {res.json()['message']}")
                else:
                    st.error("Failed to trigger pipeline.")
            except:
                 st.error("Failed to connect to API.")