import streamlit as st
import pandas as pd
import time
import smtplib
from email.mime.text import MIMEText
import joblib
from datetime import datetime
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from geoip2 import database

# NEW: Import Google Generative AI
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# --- FIRST STREAMLIT COMMAND MUST BE HERE ---
st.set_page_config(page_title="Real-time Intrusion Detection System", layout="wide")
# --- END FIRST STREAMLIT COMMAND ---

# --- Configuration ---
# Paths to your model files and data
MODEL_DIR = r'C:\Users\maryn\Downloads\final\models' # Base directory for models
GEOIP_DIR = r'C:\Users\maryn\Downloads\final\geoip' # Base directory for GeoIP

MODEL_PATH = os.path.join(MODEL_DIR, 'attack_detector_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'attack_scaler.pkl')
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'attack_label_encoder.pkl')
ANOMALY_MODEL_PATH = os.path.join(MODEL_DIR, 'anomaly_detector_model.pkl')
GEOIP_DB_PATH = os.path.join(GEOIP_DIR, 'GeoLite2-City.mmdb')

DATA_PATH = r'C:\Users\maryn\Downloads\final\NF-UNSW-NB15-v3.csv' # Used for column names if needed


# Email Configuration - IMPORTANT: Using Streamlit Secrets
# If you don't use secrets, replace st.secrets.get(...) with your actual values
SENDER_EMAIL = st.secrets.get("sender_email", "your.sender@example.com")
RECEIVER_EMAIL = st.secrets.get("receiver_email", "your.receiver@example.com")
EMAIL_PASSWORD = st.secrets.get("email_password", "YOUR_GMAIL_APP_PASSWORD")

# NEW: LLM API Key from Streamlit Secrets
LLM_API_KEY = st.secrets.get("llm_api_key", "YOUR_GEMINI_API_KEY_HERE")

# Configure Gemini API
if LLM_API_KEY and LLM_API_KEY != "YOUR_GEMINI_API_KEY_HERE":
    genai.configure(api_key=LLM_API_KEY)
    LLM_AVAILABLE = True
else:
    LLM_AVAILABLE = False


# --- File existence checks (Delayed error reporting) ---
error_messages = []
if not os.path.exists(MODEL_PATH):
    error_messages.append(f"Model file not found: `{MODEL_PATH}`. Please ensure it's in the correct directory.")
if not os.path.exists(SCALER_PATH):
    error_messages.append(f"Scaler file not found: `{SCALER_PATH}`. Please ensure it's in the correct directory.")
if not os.path.exists(LABEL_ENCODER_PATH):
    error_messages.append(f"Label Encoder file not found: `{LABEL_ENCODER_PATH}`. Please ensure it's in the correct directory.")
if not os.path.exists(DATA_PATH):
    error_messages.append(f"Data file not found: `{DATA_PATH}`. Please ensure it's in the correct directory or adjust the path.")
if not os.path.exists(ANOMALY_MODEL_PATH):
    error_messages.append(f"Anomaly Detector Model not found: `{ANOMALY_MODEL_PATH}`. Please ensure it's in the correct directory and you've created it.")
if not os.path.exists(GEOIP_DB_PATH):
    error_messages.append(f"GeoLite2 City database not found: `{GEOIP_DB_PATH}`. Please download 'GeoLite2-City.mmdb' and place it in the 'geoip' subfolder.")

# Display all collected errors *after* set_page_config
if error_messages:
    for msg in error_messages:
        st.error(msg)
    st.stop() # Halts the script execution

# Display LLM warning now that set_page_config is done
if not LLM_AVAILABLE:
    st.warning("Gemini API Key not configured. LLM features will be disabled.")


# --- Load AI Model and Tools ---
@st.cache_resource
def load_ml_components():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        anomaly_model = joblib.load(ANOMALY_MODEL_PATH)
        return model, scaler, label_encoder, anomaly_model
    except Exception as e:
        st.error(f"Error loading ML components: {e}. Make sure paths are correct and files are accessible.")
        st.stop()

model, scaler, label_encoder, anomaly_model = load_ml_components()


# Define columns to drop (Global definition for consistency)
columns_to_drop_for_features = [
    'IPV4_SRC_ADDR',
    'IPV4_DST_ADDR',
    'SRC_TO_DST_SECOND_BYTES',
    'DST_TO_SRC_SECOND_BYTES',
    'Label',
    'Attack',
    # Uncomment and add any other columns you explicitly dropped from your original dataset
    # e.g., 'sport', 'dsport', 'service' etc., if they were not used in the model
    # 'sport',
    # 'dsport',
    # 'service',
    # 'proto',
    # 'state'
]

# --- Get All Model Feature Columns (Crucial for consistent ordering) ---
@st.cache_data
def get_model_feature_columns(data_path, cols_to_drop):
    try:
        df_temp = pd.read_csv(data_path)
        features_df = df_temp.drop(columns=cols_to_drop, errors='ignore')
        return features_df.columns.tolist()
    except Exception as e:
        st.error(f"Error determining model feature columns: {e}. Check DATA_PATH and `columns_to_drop`.")
        st.stop()
model_feature_cols = get_model_feature_columns(DATA_PATH, columns_to_drop_for_features)


# --- Determine Top Features and Default Values (Outside UI function) ---
@st.cache_data
def get_top_features_and_defaults(_model, data_path, cols_to_drop, all_feature_names, num_features=15):
    try:
        df_full_for_defaults = pd.read_csv(data_path)
        features_df_original = df_full_for_defaults.drop(
            columns=cols_to_drop,
            errors='ignore'
        )

        missing_in_data = set(all_feature_names) - set(features_df_original.columns)
        if missing_in_data:
            st.error(f"Critical Error: The loaded data from '{data_path}' is missing features that your model expects: {missing_in_data}. Please check your CSV and `columns_to_drop_for_features`.")
            st.stop()

        # Ensure features_df_original only contains columns that are in all_feature_names
        # This step is crucial for consistent input to the model/scaler
        features_df_original = features_df_original[all_feature_names]

        feature_defaults = features_df_original.mean().to_dict()

        if hasattr(_model, 'feature_importances_'):
            importances = _model.feature_importances_
            feature_importance_df = pd.DataFrame({'feature': all_feature_names, 'importance': importances})
            feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
            top_features = feature_importance_df.head(num_features)['feature'].tolist()
            return top_features, feature_defaults
        else:
            st.warning("Model does not have 'feature_importances_'. Using all features for manual test inputs.")
            return all_feature_names, feature_defaults
    except Exception as e:
        st.error(f"Error determining top features and defaults: {e}. Check DATA_PATH and `columns_to_drop_for_features`.")
        st.stop()


# Call the function here, outside any try block, and ensure it's not indented unexpectedly
top_N_features, all_feature_defaults = get_top_features_and_defaults(model, DATA_PATH, columns_to_drop_for_features, model_feature_cols, num_features=15)

# --- Define Attack Templates (Crucial: Populate with realistic values from your dataset) ---
# IMPORTANT: Replace 'Feature_X' with your ACTUAL column names.
# For example, if 'Dur' is your first feature, change 'Feature_1': 0.0001 to 'Dur': 0.0001
# Populate with actual mean/median from YOUR training data for each attack type for best results.
# Ensure ALL features in `model_feature_cols` are present in each template for consistency.
# If a feature is not explicitly in a template, it will default to `all_feature_defaults`.


# Helper to get a feature name safely, falling back to a placeholder if model_feature_cols is too short
def get_safe_feature_name(index, default_prefix="Feature_"):
    # This helper needs to be careful if model_feature_cols is empty or not yet fully determined
    if model_feature_cols and index < len(model_feature_cols):
        return model_feature_cols[index]
    return f"{default_prefix}{index + 1}"

# Updated Attack Templates with more specific scenarios and placeholders for values
attack_templates = {
    "Benign": {
        # Populate with actual mean/median values for Benign flows from your dataset
        # Example: Replace get_safe_feature_name(0) with 'Dur' if that's your 1st feature
        get_safe_feature_name(0): 0.0001,  # Dur
        get_safe_feature_name(1): 2.0,     # Spkts
        get_safe_feature_name(2): 2.0,     # Dpkts
        get_safe_feature_name(3): 100.0,   # Sbytes
        get_safe_feature_name(4): 100.0,   # Dbytes
        get_safe_feature_name(5): 10000.0, # Rate
        # Ensure you replace these with your actual one-hot encoded feature names and values if applicable
        # Example for Proto_tcp and State_FIN being features at index 39 and 50 respectively in your model_feature_cols
        get_safe_feature_name(39): 1.0, # Proto_tcp (example)
        get_safe_feature_name(50): 1.0, # State_FIN (example)
        # Add ALL other features from model_feature_cols here with their benign values.
        # If a feature is not explicitly listed, it will use all_feature_defaults later.
        # This is where the ACTION REQUIRED message is primarily aimed.
    },
    "DoS": {
        get_safe_feature_name(0): 1.0,     # Dur
        get_safe_feature_name(1): 5000.0,  # Spkts
        get_safe_feature_name(2): 5000.0,  # Dpkts
        get_safe_feature_name(3): 1000000.0, # Sbytes
        get_safe_feature_name(4): 1000000.0, # Dbytes
        get_safe_feature_name(5): 500000.0, # Rate
        get_safe_feature_name(40): 1.0, # Proto_udp (example for common DoS)
        get_safe_feature_name(51): 1.0, # State_INT (example for common DoS)
        # Add ALL other features from model_feature_cols here with their DoS values.
    },
    "Exploits": {
        get_safe_feature_name(0): 0.01,    # Dur
        get_safe_feature_name(1): 10.0,    # Spkts
        get_safe_feature_name(2): 10.0,    # Dpkts
        get_safe_feature_name(3): 500.0,   # Sbytes
        get_safe_feature_name(4): 500.0,   # Dbytes
        get_safe_feature_name(5): 5000.0,  # Rate
        get_safe_feature_name(39): 1.0, # Proto_tcp
        get_safe_feature_name(44): 1.0, # State_CON (example for exploits setting up connections)
        # Add ALL other features from model_feature_cols here with their Exploits values.
    },
    "Reconnaissance": {
        get_safe_feature_name(0): 0.005,   # Dur
        get_safe_feature_name(1): 1.0,     # Spkts
        get_safe_feature_name(2): 1.0,     # Dpkts
        get_safe_feature_name(3): 60.0,    # Sbytes
        get_safe_feature_name(4): 0.0,     # Dbytes (e.g., port scan no response)
        get_safe_feature_name(5): 200.0,   # Rate
        get_safe_feature_name(39): 1.0, # Proto_tcp (common for port scans)
        get_safe_feature_name(51): 1.0, # State_INT (often for scans that don't complete connection)
        # Add ALL other features from model_feature_cols here with their Reconnaissance values.
    },
    # --- New Egypt-Specific Scenarios (Placeholder Values - Refine with your data!) ---
    "SQL Injection Attempt (Web App)": {
        get_safe_feature_name(0): 0.0002,
        get_safe_feature_name(1): 2.0,
        get_safe_feature_name(2): 2.0,
        get_safe_feature_name(3): 150.0, # Larger than normal HTTP GET
        get_safe_feature_name(4): 150.0,
        get_safe_feature_name(5): 500.0,
        get_safe_feature_name(39): 1.0, # Proto_tcp (for HTTP/S)
        get_safe_feature_name(50): 1.0, # State_FIN or State_CON
        # Add ALL other features from model_feature_cols here.
    },
    "Banking Trojan Phishing Payload": {
        get_safe_feature_name(0): 0.05,
        get_safe_feature_name(1): 5.0,
        get_safe_feature_name(2): 5.0,
        get_safe_feature_name(3): 300.0, # Initial small payload
        get_safe_feature_name(4): 300.0,
        get_safe_feature_name(5): 100.0,
        get_safe_feature_name(39): 1.0, # Proto_tcp
        get_safe_feature_name(44): 1.0, # State_CON
        # Add ALL other features from model_feature_cols here.
    },
    "Ransomware Initial Access": {
        get_safe_feature_name(0): 0.1,
        get_safe_feature_name(1): 8.0,
        get_safe_feature_name(2): 8.0,
        get_safe_feature_name(3): 600.0, # Larger payload delivery
        get_safe_feature_name(4): 600.0,
        get_safe_feature_name(5): 80.0,
        get_safe_feature_name(39): 1.0, # Proto_tcp
        get_safe_feature_name(44): 1.0, # State_CON
        # Add ALL other features from model_feature_cols here.
    },
}

# Static warning, will disappear once user updates the template names
# This check is still valid and important.
if 'Feature_1' in attack_templates.get("Benign", {}):
    st.warning("""
        **ACTION REQUIRED:**
        You must update the `attack_templates` dictionary in the code
        with the **EXACT FEATURE NAMES** from your `NF-UNSW-NB15-v3.csv` dataset,
        after dropping the specified columns (`IPV4_SRC_ADDR`, `Attack`, etc.).

        Currently, they are placeholders like `Feature_1`, `Feature_2`.
        Refer to the previous explanation for how to get the correct names from your CSV.
    """)


# --- Geo-IP Lookup Function ---
@st.cache_data
def get_geo_location(ip_address):
    try:
        with database.Reader(GEOIP_DB_PATH) as reader:
            response = reader.city(ip_address)
            lat = response.location.latitude
            lon = response.location.longitude
            city = response.city.name if response.city.name else "Unknown City"
            country = response.country.name if response.country.name else "Unknown Country"
            return lat, lon, city, country
    except Exception: # Catch any error, including IP not found in DB
        return None, None, "Unknown", "Unknown"

# --- LLM Context Generation Function ---
@st.cache_data(show_spinner=False)
def get_llm_attack_context(attack_type, ip_address, confidence, city="Unknown", country="Unknown"):
    if not LLM_AVAILABLE:
        return "LLM not available. Please configure API key."

    location_info = ""
    if city != "Unknown" and country != "Unknown":
        location_info = f"This attack originated from {city}, {country} (IP: {ip_address})."
        # Explicitly mention Egypt relevance
        if country == "Egypt" and datetime.now().year >= 2023: # Add a current context
            location_info += " **This is particularly relevant for an Egyptian company's cybersecurity posture, given recent regional threat trends.**"
    elif ip_address != "Unknown":
         location_info = f"This attack originated from IP address {ip_address}."

    confidence_hint = ""
    if confidence is not None:
        if confidence >= 90:
            confidence_hint = "The system detected this with high confidence."
        elif confidence < 70:
            confidence_hint = "The system detected this with lower confidence, indicating potential ambiguity."

    # Enhanced prompt for Egypt context
    prompt = f"""
    You are a highly knowledgeable cybersecurity expert specializing in the threat landscape for companies in Egypt.
    Provide a concise explanation (2-3 sentences) of the attack type '{attack_type}' in the context of network security, specifically highlighting any relevance or commonality within the Egyptian cybersecurity environment if applicable.
    {location_info} {confidence_hint}
    Then, suggest one or two immediate, practical mitigation steps that a security analyst working for an Egyptian company might consider for an attack of this type, keeping local IT infrastructure and common practices in mind.
    Focus on practical, high-level advice, and consider Egyptian compliance or common technical stacks.
    """

    try:
        model_llm = genai.GenerativeModel('gemini-2.0-flash')
        response = model_llm.generate_content(
            prompt,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        return response.text
    except Exception as e:
        return f"Error getting LLM context: {e}. (API issue or content blocked. Ensure LLM_API_KEY is correct.)"

# --- Email Function ---
def send_email(pred_label, ip, confidence=None, anomaly_status="", llm_context=""):
    if pred_label.lower() == "benign":
        return

    conf_str = f" (Confidence: {confidence:.2f}%)" if confidence is not None else ""
    anomaly_str = f" (Anomaly: {anomaly_status})" if anomaly_status else ""
    # Truncate LLM context for email body to prevent excessively long emails
    context_str = f"\nLLM Context (Truncated):\n{llm_context[:500]}..." if llm_context else ""
    msg = MIMEText(f"🚨 Intrusion Detected: {pred_label} from {ip}{conf_str}{anomaly_str}{context_str}")
    msg["Subject"] = "Cyber Security Alert"
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, EMAIL_PASSWORD)
            server.send_message(msg)
            st.success(f"Email sent for {pred_label} from {ip}")
    except Exception as e:
        st.error(f"Email error: {e}. Check your Gmail App Password and sender/receiver settings. Ensure your Gmail account allows less secure apps or has an App Password generated.")

# --- Defense Action Implementations (simulated) ---
def block_ip(ip):
    st.warning(f"**[DEFENSE ACTION]** Simulated: Blocked IP `{ip}` (iptables would be executed here).")

def rate_limit_ip(ip):
    st.warning(f"**[DEFENSE ACTION]** Simulated: Rate limiting applied to `{ip}`.")

def isolate_host(ip):
    st.warning(f"**[DEFENSE ACTION]** Simulated: Isolating host `{ip}` - block all outgoing and incoming traffic.")

def log_and_monitor(ip):
    st.info(f"**[DEFENSE ACTION]** Simulated: Monitoring IP `{ip}` closely, logging deep packet data...")

def kill_suspicious_process(ip):
    st.warning(f"**[DEFENSE ACTION]** Simulated: Kill suspicious process from `{ip}` (manual or scripted response needed).")

def enable_captcha(ip):
    st.warning(f"**[DEFENSE ACTION]** Simulated: Enable CAPTCHA or lock user account from IP `{ip}` (web layer defense).")

def enhanced_monitoring(ip):
    st.info(f"**[DEFENSE ACTION]** Simulated: Enhanced monitoring/logging for `{ip}` due to low confidence detection or anomaly.")


# --- Triggering Alert and Response (Enhanced with Confidence and Anomaly) ---
def trigger_alert_and_defense(pred_label, ip, confidence, anomaly_status, city, country, status_placeholder):
    llm_context = ""
    # Only get LLM context if it's not benign or if it's an anomaly
    if pred_label.lower() != "benign" or anomaly_status == "Anomaly":
        with st.spinner(f"Getting LLM context for {pred_label} from {ip}..."):
            llm_context = get_llm_attack_context(pred_label, ip, confidence, city, country)

    status_placeholder.info(f"**[PREDICTION]** Detected: `{pred_label}` (Conf: {confidence:.2f}%) | Anomaly: `{anomaly_status}` from `{ip}` ({city}, {country})")
    if llm_context:
        status_placeholder.markdown(f"**LLM Context:**\n> {llm_context}")

    action_taken = ""

    send_email(pred_label, ip, confidence, anomaly_status, llm_context)

    with status_placeholder:
        HIGH_CONFIDENCE_THRESHOLD = 85.0 # Use 85% directly as it's already %

        if pred_label.lower() == "benign" and anomaly_status == "Normal":
            status_placeholder.success("[ACTION] Traffic allowed (benign)")
            action_taken = "Allow"
        elif pred_label.lower() != "benign" and confidence >= HIGH_CONFIDENCE_THRESHOLD:
            # High confidence attack
            match pred_label:
                case "Exploits" | "SQL Injection Attempt (Web App)" | "Ransomware Initial Access": # Grouping similar actions
                    block_ip(ip)
                    isolate_host(ip)
                    action_taken = "Block + Isolate (High Conf.)"
                case "Fuzzers":
                    rate_limit_ip(ip)
                    action_taken = "Rate Limit (High Conf.)"
                case "Generic":
                    enable_captcha(ip)
                    rate_limit_ip(ip)
                    action_taken = "CAPTCHA + Rate Limit (High Conf.)"
                case "Reconnaissance":
                    block_ip(ip)
                    action_taken = "Block (High Conf.)"
                case "DoS":
                    rate_limit_ip(ip)
                    action_taken = "Rate Limit (High Conf.)"
                case "Backdoor" | "Shellcode" | "Banking Trojan Phishing Payload": # Grouping similar actions
                    kill_suspicious_process(ip)
                    isolate_host(ip) # Consider isolating for trojans/backdoors
                    action_taken = "Kill Process + Isolate (High Conf.)"
                case "Analysis":
                    log_and_monitor(ip)
                    action_taken = "Log and Monitor (High Conf.)"
                case "Worms":
                    isolate_host(ip)
                    action_taken = "Isolate (High Conf.)"
                case _:
                    st.warning(f"**[WARNING]** Unknown attack type: `{pred_label}` (High Conf.). Applying default strong action.")
                    block_ip(ip) # Default strong action for unknown high conf
                    action_taken = f"Unknown ({pred_label}) - Block (High Conf.)"
        elif (pred_label.lower() != "benign" and confidence < HIGH_CONFIDENCE_THRESHOLD) or anomaly_status == "Anomaly":
            # Low confidence attack or general anomaly
            enhanced_monitoring(ip)
            log_and_monitor(ip)
            action_taken = "Enhanced Monitoring (Low Conf./Anomaly)"
        else:
            st.warning(f"**[WARNING]** Unhandled detection scenario: `{pred_label}` | Anomaly: `{anomaly_status}`")
            action_taken = "Unhandled"

    return action_taken, llm_context


# --- Streamlit UI ---
st.title("🛡️ Real-time Intrusion Detection and Defense (Local Demo)")
st.markdown("This application simulates a real-time AI-powered system for detecting cyber intrusions and performing automated defense actions.")

# Removed tab5 (Manual Test)
tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Report & Summary", "Defense Actions", "LLM Analyst Assistant"])


with tab1: # Dashboard Tab
    st.header("🔍 Detection Log")
    log_container = st.empty()
    st.header("⚡ Current Action Status")
    status_container = st.empty()

    st.sidebar.header("System Control")
    start_button = st.sidebar.button("Start Simulation", type="primary")
    stop_button = st.sidebar.button("Stop Simulation")

    st.sidebar.markdown("---")
    st.sidebar.header("Configuration & Status")
    st.sidebar.write(f"Model Path: `{MODEL_PATH}`")
    st.sidebar.write(f"Anomaly Model Path: `{ANOMALY_MODEL_PATH}`")
    st.sidebar.write(f"Data Path: `{DATA_PATH}`")
    st.sidebar.write(f"GeoIP DB Path: `{GEOIP_DB_PATH}`")
    st.sidebar.write(f"Sender: `{SENDER_EMAIL}`")
    st.sidebar.write(f"Receiver: `{RECEIVER_EMAIL}`")
    st.sidebar.info("Defense actions are simulated for local execution.")
    if LLM_AVAILABLE:
        st.sidebar.success("Gemini LLM is active!")
    else:
        st.sidebar.error("Gemini LLM is NOT active. Check API key.")


    if 'running' not in st.session_state:
        st.session_state['running'] = False
    if 'simulation_results' not in st.session_state:
        st.session_state['simulation_results'] = []
    # Removed 'precomputed_data' from session_state as we'll load new each time

    if start_button:
        st.session_state['running'] = True
        st.session_state['simulation_results'] = [] # Clear previous results
        st.info("Loading and pre-processing data for simulation...")

        try:
            # Load data directly when start is pressed
            df_full = pd.read_csv(DATA_PATH)
            # You can change the slice here if you want to simulate more/less data each run
            # For demonstration, let's just shuffle and take the first 100 rows
            df_demo = df_full.sample(n=min(100, len(df_full)), random_state=int(time.time())).copy() # Shuffle for different data each run
            # If you want a different random subset each time, use random_state=None or remove it.
            # Using int(time.time()) will give a different shuffle each time you press start.

            features_df = df_demo.drop(
                columns=columns_to_drop_for_features,
                errors='ignore'
            )
            # Ensure the order of columns matches the model's training order
            features_df = features_df[model_feature_cols]

            if features_df.empty:
                raise ValueError("No data left after dropping columns for feature processing. Check `DATA_PATH` and `columns_to_drop_for_features`.")
            if features_df.shape[1] == 0:
                raise ValueError("No features remaining after dropping columns. Adjust `columns_to_drop_for_features`.")
            if features_df.shape[1] != len(model_feature_cols):
                 raise ValueError(f"Mismatch in feature count: {features_df.shape[1]} features in demo data, but model expects {len(model_feature_cols)} features. Check `columns_to_drop_for_features` and `model_feature_cols` definition.")


            features_scaled = scaler.transform(features_df.values)

            # Store computed flows for the current run
            current_precomputed_flows = []
            for i, row_data in df_demo.iterrows():
                ip = row_data['IPV4_SRC_ADDR']
                # The index 'i' here refers to the original DataFrame's index.
                # When using .sample(), the new DataFrame `df_demo` will retain original indices.
                # Use df_demo.index.get_loc(i) to get the row index in the potentially reordered df_demo if needed for features_scaled.
                # More robust: get features_scaled row by iterating on features_df directly
                input_scaled_row = scaler.transform(features_df.loc[[row_data.name]]) # Pass a single row DF to scaler

                prediction_value = model.predict(input_scaled_row)[0]
                pred_label = label_encoder.inverse_transform([prediction_value])[0]
                confidence = model.predict_proba(input_scaled_row).max() * 100

                anomaly_score = anomaly_model.decision_function(input_scaled_row)[0]
                anomaly_status = "Anomaly" if anomaly_model.predict(input_scaled_row)[0] == -1 else "Normal"

                lat, lon, city, country = get_geo_location(ip)

                current_precomputed_flows.append({
                    "ip": ip,
                    "pred_label": pred_label,
                    "confidence": confidence,
                    "anomaly_score": anomaly_score,
                    "anomaly_status": anomaly_status,
                    "latitude": lat,
                    "longitude": lon,
                    "city": city,
                    "country": country
                })
            st.session_state['current_run_flows'] = current_precomputed_flows # Store for current run only
            st.success(f"Data pre-processing complete. Ready to simulate {len(current_precomputed_flows)} flows.")

        except Exception as e:
            st.error(f"Error loading or pre-processing data: {e}. Check DATA_PATH and model paths.")
            st.session_state['running'] = False
            st.info("Simulation halted due to pre-processing error. Please check configurations.")


    if st.session_state.get('running', False) and 'current_run_flows' in st.session_state and st.session_state['current_run_flows']:
        st.success("Simulation started. Processing traffic flows...")
        progress_bar = st.progress(0)
        counter = 0

        log_messages = []

        # Iterate over the precomputed flows for the current run
        while st.session_state.get('running', False) and counter < len(st.session_state['current_run_flows']):
            try:
                flow_data = st.session_state['current_run_flows'][counter]
                ip = flow_data['ip']
                pred_label = flow_data['pred_label']
                confidence = flow_data['confidence']
                anomaly_status = flow_data['anomaly_status']
                lat = flow_data['latitude']
                lon = flow_data['longitude']
                city = flow_data['city']
                country = flow_data['country']

                action_taken, llm_context = trigger_alert_and_defense(pred_label, ip, confidence, anomaly_status, city, country, status_container)

                log_messages.append(f"**Flow {counter + 1}**: IP `{ip}` ({city}, {country}) - Detected: `{pred_label}` (Conf: {confidence:.2f}%) - Anomaly: `{anomaly_status}` - Action: `{action_taken}`")
                # Display only the most recent logs
                log_container.markdown("\n".join(log_messages[-10:])) # Show last 10 messages

                st.session_state['simulation_results'].append({
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "IP Address": ip,
                    "City": city,
                    "Country": country,
                    "Detected Attack": pred_label,
                    "Confidence (%)": f"{confidence:.2f}",
                    "Anomaly Status": anomaly_status,
                    "Anomaly Score": f"{flow_data['anomaly_score']:.2f}",
                    "Action Taken": action_taken,
                    "LLM Context": llm_context,
                    "latitude": lat,
                    "longitude": lon
                })

                counter += 1
                progress_bar.progress(counter / len(st.session_state['current_run_flows']))
                time.sleep(0.1) # Simulate real-time delay

            except Exception as e:
                st.error(f"[ERROR in row {counter}] {e}. Simulation halted.")
                st.session_state['running'] = False
                break

        if st.session_state.get('running', False): # If loop finishes without error
            st.session_state['running'] = False
            st.success("Simulation finished.")

    elif stop_button:
        st.session_state['running'] = False
        st.warning("Simulation stopped by user.")
    else:
        st.info("Click 'Start Simulation' to begin detecting intrusions.")

with tab2: # Report & Summary Tab
    st.header("📊 Simulation Report & Summary")
    st.markdown("This section provides an overview of the detected attacks, anomaly detections, and actions taken during the simulation.")

    if not st.session_state['simulation_results']:
        st.info("No simulation data available yet. Please run the simulation on the 'Dashboard' tab.")
    else:
        report_df = pd.DataFrame(st.session_state['simulation_results'])
        report_df['Confidence (%)'] = pd.to_numeric(report_df['Confidence (%)'], errors='coerce')
        report_df['Anomaly Score'] = pd.to_numeric(report_df['Anomaly Score'], errors='coerce')
        report_df.dropna(subset=['Confidence (%)', 'Anomaly Score'], inplace=True)


        st.subheader("Summary Statistics")
        col_summary1, col_summary2, col_summary3 = st.columns(3)
        with col_summary1:
            total_flows = len(report_df)
            st.metric("Total Flows Processed", total_flows)
            attack_flows = report_df[report_df['Detected Attack'].str.lower() != 'benign']
            st.metric("Total Attacks Detected", len(attack_flows))
        with col_summary2:
            anomaly_detections = report_df[report_df['Anomaly Status'] == 'Anomaly']
            st.metric("Total Anomalies Detected", len(anomaly_detections))
            unique_ips = report_df['IP Address'].nunique()
            st.metric("Unique IPs Involved", unique_ips)
        with col_summary3:
            avg_attack_confidence = attack_flows['Confidence (%)'].mean() if not attack_flows.empty else 0
            st.metric("Avg. Attack Confidence", f"{avg_attack_confidence:.2f}%")
            avg_anomaly_score = anomaly_detections['Anomaly Score'].mean() if not anomaly_detections.empty else 0
            st.metric("Avg. Anomaly Score", f"{avg_anomaly_score:.2f}")


        st.subheader("Geographical Distribution of Detections")
        map_df = report_df[(report_df['Detected Attack'].str.lower() != 'benign') | (report_df['Anomaly Status'] == 'Anomaly')]
        map_df = map_df.dropna(subset=['latitude', 'longitude'])
        if not map_df.empty:
            st.map(map_df[['latitude', 'longitude']], zoom=1)
            st.info(f"Showing {len(map_df)} attack/anomaly locations on the map. Note: Geo-IP resolution is approximate.")
        else:
            st.info("No attack or anomaly detections with valid geographical data to display on the map.")


        st.subheader("Attack Type Distribution (Excluding Benign)")
        if not attack_flows.empty:
            attack_counts = attack_flows['Detected Attack'].value_counts()
            fig_attacks, ax_attacks = plt.subplots(figsize=(8, 6))
            sns.barplot(x=attack_counts.index, y=attack_counts.values, ax=ax_attacks, palette='viridis')
            ax_attacks.set_title('Distribution of Detected Attack Types')
            ax_attacks.set_xlabel('Attack Type')
            ax_attacks.set_ylabel('Number of Detections')
            ax_attacks.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            st.pyplot(fig_attacks)
        else:
            st.info("No attacks detected in the simulation to plot.")

        st.subheader("Anomaly vs. Normal Distribution")
        anomaly_counts = report_df['Anomaly Status'].value_counts()
        fig_anomaly, ax_anomaly = plt.subplots(figsize=(6, 6))
        ax_anomaly.pie(anomaly_counts, labels=anomaly_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
        ax_anomaly.set_title('Distribution of Anomaly Status')
        st.pyplot(fig_anomaly)


        st.subheader("Actions Taken Distribution")
        if not report_df.empty:
            action_counts = report_df['Action Taken'].value_counts()
            fig_actions, ax_actions = plt.subplots(figsize=(8, 6))
            sns.barplot(x=action_counts.index, y=action_counts.values, ax=ax_actions, palette='magma')
            ax_actions.set_title('Distribution of Defense Actions Taken')
            ax_actions.set_xlabel('Action Taken')
            ax_actions.set_ylabel('Count')
            plt.xticks(rotation=45, ha='right') # Rotate for better readability
            plt.tight_layout()
            st.pyplot(fig_actions)


        st.subheader("Raw Simulation Data (Last 20 Flows with LLM Context)")
        display_cols = ["Timestamp", "IP Address", "City", "Country", "Detected Attack", "Confidence (%)", "Anomaly Status", "Action Taken", "LLM Context"]
        st.dataframe(report_df[display_cols].tail(20))

with tab3: # Defense Actions Tab
    st.header("🛡️ Automated Defense Actions Explained")
    st.markdown("""
    This system implements a set of automated defense actions based on the type of detected intrusion and the system's confidence/anomaly assessment.
    These actions are simulated in this demonstration environment.
    """)

    st.subheader("Specific Defense Strategies:")

    st.markdown("""
    ---
    #### 🚫 **Block IP Address (Simulated: `iptables -j DROP`)**
    This action aims to immediately prevent further communication from a malicious source IP address by dropping all its incoming and outgoing packets.
    * **Triggered by:** `Exploits`, `Reconnaissance`, `SQL Injection Attempt (Web App)`, `Ransomware Initial Access` (especially with high confidence)
    ---
    #### ⏱️ **Rate Limit IP (Simulated: `iptables -m limit --limit 10/min`)**
    This action restricts the number of connections or packets allowed from a specific IP address within a time window, often used against Distributed Denial of Service (DDoS) or brute-force attacks to reduce their impact without outright blocking.
    * **Triggered by:** `Fuzzers`, `Generic`, `DoS` (especially with high confidence)
    ---
    #### 🔗 **Isolate Host (Simulated: Comprehensive Block)**
    This is a more severe measure that effectively cuts off a compromised host or a highly malicious IP from the network to prevent lateral movement or further damage.
    * **Triggered by:** `Exploits`, `Backdoor`, `Worms`, `Banking Trojan Phishing Payload`, `Ransomware Initial Access` (especially with high confidence)
    ---
    #### 📝 **Log and Monitor (Simulated: Enhanced Logging)**
    For certain types of less aggressive or more subtle attacks (e.g., advanced persistent threats), the system might opt for enhanced logging and close monitoring rather than immediate blocking, to gather more intelligence.
    * **Triggered by:** `Analysis` (with high confidence), or as a component of other actions.
    ---
    #### 🔪 **Kill Suspicious Process (Simulated: Manual Intervention Required)**
    This action represents the need to terminate processes associated with malware or unauthorized activity on a compromised host. In a real-world scenario, this often involves integration with Endpoint Detection and Response (EDR) tools.
    * **Triggered by:** `Backdoor`, `Shellcode`, `Banking Trojan Phishing Payload` (especially with high confidence)
    ---
    #### 🔒 **Enable CAPTCHA / Account Lock (Simulated: Web Application Layer)**
    For web-based attacks or unauthorized access attempts, the system might trigger defenses at the application layer, such as forcing a CAPTCHA challenge or temporarily locking user accounts.
    * **Triggered by:** `Generic` (especially with high confidence)
    ---
    #### 👁️ **Enhanced Monitoring (New Intelligent Action)**
    This action is taken when the system detects an anomaly, or when a specific attack type is detected with lower confidence. It signifies a need for increased vigilance and data collection without immediate, aggressive blocking.
    * **Triggered by:** Any attack type detected with **lower confidence**, or when an IP is flagged as an **"Anomaly"** by the unsupervised model, regardless of primary classification.
    ---
    #### 🧠 **AI-Powered Contextualization (New LLM Feature)**
    For every detected attack or anomaly, the system leverages a Large Language Model (LLM) to generate a concise explanation of the attack type and suggest immediate, general mitigation steps. This provides security analysts with instant, human-readable context, reducing investigation time and improving response efficiency.
    * **Triggered by:** Any non-benign detection or an anomaly detection.
    ---
    """)
    st.info("Note: All `iptables` and system-level commands are simulated in this demo as direct execution requires root privileges and specific server environments.")
    st.info("Note: LLM calls require an active internet connection and a configured API key. Delays may occur during LLM responses.")

with tab4: # LLM Analyst Assistant Tab
    st.header("🤖 LLM Analyst Assistant")
    st.markdown("Ask the AI about cybersecurity threats or get more detailed insights on the **current simulation detections**.")

    def get_simulation_summary_for_llm():
        if not st.session_state.get('simulation_results'):
            return "No simulation data available yet."

        report_df = pd.DataFrame(st.session_state['simulation_results'])

        report_df['Confidence (%)'] = pd.to_numeric(report_df['Confidence (%)'], errors='coerce')
        report_df['Anomaly Score'] = pd.to_numeric(report_df['Anomaly Score'], errors='coerce')
        report_df.dropna(subset=['Confidence (%)', 'Anomaly Score'], inplace=True)

        summary_lines = []

        total_flows = len(report_df)
        summary_lines.append(f"Total network flows processed: {total_flows}.")

        attack_flows = report_df[report_df['Detected Attack'].str.lower() != 'benign']
        if not attack_flows.empty:
            total_attacks = len(attack_flows)
            summary_lines.append(f"Total attacks detected: {total_attacks}.")
            top_attacks = attack_flows['Detected Attack'].value_counts().head(3).to_dict()
            summary_lines.append(f"Top detected attack types and their counts: {json.dumps(top_attacks)}.")
            avg_attack_conf = attack_flows['Confidence (%)'].mean()
            summary_lines.append(f"Average confidence for attacks: {avg_attack_conf:.2f}%.")
        else:
            summary_lines.append("No specific attacks (non-benign) were detected.")

        anomaly_detections = report_df[report_df['Anomaly Status'] == 'Anomaly']
        if not anomaly_detections.empty:
            total_anomalies = len(anomaly_detections)
            summary_lines.append(f"Total anomalies detected: {total_anomalies}.")
            avg_anomaly_score = anomaly_detections['Anomaly Score'].mean()
            summary_lines.append(f"Average anomaly score for detected anomalies: {avg_anomaly_score:.2f}.")
        else:
            summary_lines.append("No anomalies were detected.")

        unique_ips = report_df['IP Address'].nunique()
        summary_lines.append(f"Number of unique IP addresses involved: {unique_ips}.")

        top_source_ips = report_df['IP Address'].value_counts().head(3).to_dict()
        summary_lines.append(f"Top 3 most frequent source IPs and their counts: {json.dumps(top_source_ips)}.")

        top_countries = report_df['Country'].value_counts().head(3).to_dict()
        summary_lines.append(f"Top 3 countries of origin for traffic: {json.dumps(top_countries)}.")

        return "\n".join(summary_lines)

    st.info("You can ask questions like: \n- 'What was the most common attack type detected?'\n- 'How many anomalies were there?'\n- 'Which countries were most involved?'\n- 'What was the average confidence of detected attacks?'\n\nOr any general cybersecurity question, especially regarding the Egyptian context.")

    user_query = st.text_area("Your Question:", height=100, key="llm_analyst_query_input")

    if st.button("Get Insight", key="get_llm_insight_button"):
        if not LLM_AVAILABLE:
            st.warning("Gemini API Key not configured. LLM features are disabled.")
        elif not user_query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Generating LLM insight..."):
                try:
                    sim_summary = get_simulation_summary_for_llm()

                    full_prompt = f"""
                    You are a highly knowledgeable cybersecurity analyst.
                    I am running an intrusion detection system simulation. Here is a summary of the events that occurred during the recent simulation run:
                    ---
                    {sim_summary}
                    ---
                    Based on the provided simulation summary and your general cybersecurity knowledge, please answer the following question thoroughly and clearly, providing practical advice if applicable.
                    If the question can be directly answered by the simulation summary, prioritize that information.
                    If not, use your general knowledge, specifically considering the context of cybersecurity challenges for companies in Egypt.

                    Question: {user_query}
                    """

                    model_llm_qa = genai.GenerativeModel('gemini-2.0-flash')
                    qa_response = model_llm_qa.generate_content(
                        full_prompt,
                        safety_settings={
                            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                        }
                    )
                    st.markdown("---")
                    st.subheader("LLM Response:")
                    st.info(qa_response.text)
                except Exception as e:
                    st.error(f"Error getting LLM insight: {e}. (API issue or content blocked)")