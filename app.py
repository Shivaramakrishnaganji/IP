# app.py

import streamlit as st
import pandas as pd
import pickle
import time  # <-- 1. IMPORT TIME
import os
from openai import OpenAI
from dotenv import load_dotenv

# --- Load environment variables from .env file ---
load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="Solar Power Predictor",
    page_icon="☀️",
    layout="wide" 
)

# --- Constants ---
STANDARD_INVERTER_DC_CAPACITY_KW = 1450.0

# --- Model Loading ---
@st.cache_resource
def load_model():
    """Loads the pre-trained model and feature columns from disk."""
    try:
        with open('solar_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('model_columns.pkl', 'rb') as f:
            model_columns = pickle.load(f)
        return model, model_columns
    except FileNotFoundError:
        st.error("Error: Model file 'solar_model.pkl' or 'model_columns.pkl' not found.")
        st.error("Please run 'train_model.py' first to create the model files.")
        return None, None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None, None

model, model_columns = load_model()

# --- State Management ---
def clear_prediction_state():
    st.session_state.show_prediction = False
    if 'last_prediction' in st.session_state:
        del st.session_state.last_prediction
    if 'messages' in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": "System prompt (will be updated)"}, 
            {"role": "assistant", "content": "Hi! Run a prediction, then ask me about it. Your context will be cleared if you change the inputs."}
        ]

if 'show_prediction' not in st.session_state:
    st.session_state.show_prediction = False


# --- LAYOUT: Title and Chatbot Button in Columns ---
col1, col2 = st.columns([3, 1]) 

with col1:
    st.title("☀️ Solar Power Generation Predictor")

# --- Chatbot Popover (the "layer") ---
with col2:
    st.write("") 
    st.write("")
    with st.popover("🤖 AI Explainer Assistant", use_container_width=True):
        
        st.markdown("Ask me questions about this project!")
        api_key = os.environ.get("OPENAI_API_KEY")

        if not api_key:
            st.warning("Could not find OpenAI API Key. Please make sure you have a `.env` file with your `OPENAI_API_KEY`.")
        else:
            try:
                client = OpenAI(api_key=api_key)
            except Exception as e:
                st.error(f"Error initializing OpenAI client: {e}")
                client = None
                
            model_name = "gpt-4o-mini" 
            
            system_prompt = f"""
            You are an AI expert assistant for a college student's 'Solar Power Predictor' project expo...
            (Your full system prompt text)
            ...
            **Prediction Inputs:** {st.session_state.get('last_prediction', {}).get('inputs', 'No prediction yet')}
            **Prediction Calculation:** {st.session_state.get('last_prediction', {}).get('calculation', 'No prediction yet')}
            ...
            """

            # Initialize chat history
            if "messages" not in st.session_state:
                st.session_state.messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "assistant", "content": "Hi! Run a prediction, then ask me about it. (e.g., 'Why is my prediction low?')."}
                ]

            # (The rest of your chatbot logic: for loop, if prompt, etc...)
            for message in st.session_state.messages:
                if message["role"] != "system": 
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

            if prompt := st.chat_input("Ask about your prediction..."):
                if not client:
                    st.error("OpenAI client not initialized. Please check your API key.")
                else:
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    with st.chat_message("assistant"):
                        with st.spinner("AI is thinking..."):
                            try:
                                updated_system_prompt = f"""
                                You are an AI expert assistant...
                                ...
                                **Prediction Inputs:** {st.session_state.get('last_prediction', {}).get('inputs', 'No prediction yet')}
                                **Prediction Calculation:** {st.session_state.get('last_prediction', {}).get('calculation', 'No prediction yet')}
                                ...
                                """
                                api_messages = [msg for msg in st.session_state.messages if msg['role'] != 'system']
                                api_messages.insert(0, {"role": "system", "content": updated_system_prompt}) 

                                response = client.chat.completions.create(
                                    model=model_name,
                                    messages=api_messages,
                                    max_tokens=250 
                                )
                                bot_response = response.choices[0].message.content
                                st.markdown(bot_response)
                                st.session_state.messages.append({"role": "assistant", "content": bot_response})
                            
                            except Exception as e:
                                st.error(f"An error occurred with the OpenAI API. {e}")


# --- Sidebar for User Input ---
st.sidebar.header("Input Conditions")
st.sidebar.markdown("### 1. Plant DC Capacity")

total_panels = st.sidebar.number_input(
    label="Total Number of Solar Panels", min_value=1, value=30000, step=100,
    on_change=clear_prediction_state
)
panel_capacity_W = st.sidebar.number_input(
    label="Single Panel Capacity (in Watts)", min_value=1, value=330, step=5,
    help="The wattage (e.g., 330W, 450W) of one solar panel.",
    on_change=clear_prediction_state
)

st.sidebar.markdown("### 2. Weather & Time")

ambient_temp = st.sidebar.slider("Ambient Temperature (°C)", -10.0, 50.0, 25.0, 0.1, on_change=clear_prediction_state)
module_temp = st.sidebar.slider("Module Temperature (°C)", -10.0, 80.0, 35.0, 0.1, on_change=clear_prediction_state)
irradiation = st.sidebar.slider("Irradiation (Sunlight Intensity)", 0.0, 1.2, 0.5, 0.01, on_change=clear_prediction_state)
wind_speed = st.sidebar.slider("Wind Speed (m/s)", 0.0, 25.0, 5.0, 0.1,
                               help="This is a simulation. Higher wind will apply a small efficiency bonus to simulate panel cooling.", on_change=clear_prediction_state)
hour = st.sidebar.slider("Hour of the Day (0-23)", 0, 23, 12, 1, on_change=clear_prediction_state)
st.sidebar.markdown("*(0-12 = 12am-12pm; 13-23 = 1pm-11pm)*")
month = st.sidebar.slider("Month of the Year (1-12)", 1, 12, 6, 1, on_change=clear_prediction_state)


# --- MAIN PAGE: The Predictor ---
st.header("Predict Total Power Output")
st.markdown("""
    Use the sidebar to input your plant's **DC capacity** (panels) and the **weather conditions**, then click 'Predict'.
""")

# --- 2. THIS IS THE MAIN CHANGE ---
# We create a placeholder for the results.
results_placeholder = st.empty()


if model and model_columns:
    if st.sidebar.button("Predict Power Output"):
        
        # --- 3. WRAP YOUR LOGIC IN A SPINNER ---
        with st.spinner("Analyzing weather and calculating power..."):
            
            # 1. Run all calculations
            input_data = {
                'AMBIENT_TEMPERATURE': ambient_temp,
                'MODULE_TEMPERATURE': module_temp,
                'IRRADIATION': irradiation,
                'Hour': hour,
                'Month': month
            }
            input_df = pd.DataFrame([input_data])
            input_df = input_df[model_columns] 
            prediction = model.predict(input_df)
            single_inverter_ac_power = prediction[0]
            
            if irradiation == 0.0 or hour < 5 or hour > 19:
                single_inverter_ac_power = 0.0
            
            total_plant_dc_capacity_kW = (total_panels * panel_capacity_W) / 1000.0
            
            if STANDARD_INVERTER_DC_CAPACITY_KW == 0:
                 scaling_factor = 0.0
            else:
                 scaling_factor = total_plant_dc_capacity_kW / STANDARD_INVERTER_DC_CAPACITY_KW
            
            ml_predicted_power = single_inverter_ac_power * scaling_factor
            cooling_bonus_factor = 1.0 + (wind_speed * 0.002) 
            total_predicted_power = ml_predicted_power * cooling_bonus_factor

            # --- 4. ADD ARTIFICIAL DELAY ---
            time.sleep(2) # Wait 2 seconds to make it feel like a real calculation

            # 5. Save all results to session_state
            st.session_state.show_prediction = True
            st.session_state.total_predicted_power = total_predicted_power
            st.session_state.total_plant_dc_capacity_kW = total_plant_dc_capacity_kW
            st.session_state.ml_predicted_power = ml_predicted_power
            st.session_state.cooling_bonus_factor = cooling_bonus_factor

            st.session_state.last_prediction = {
                "inputs": {
                    "Total Panels": total_panels,
                    "Panel Capacity (W)": panel_capacity_W,
                    "Ambient Temp (°C)": ambient_temp,
                    "Module Temp (°C)": module_temp,
                    "Irradiation": irradiation,
                    "Wind Speed (m/s)": wind_speed,
                    "Hour": hour
                },
                "calculation": {
                    "Total Plant DC Capacity (kW)": total_plant_dc_capacity_kW,
                    "Base ML Prediction (kW)": ml_predicted_power,
                    "Wind Cooling Factor": cooling_bonus_factor,
                    "Final Predicted Power (kW)": total_predicted_power
                }
            }
            
        # The spinner automatically disappears here
        
        # 6. Rerun to show results
        st.rerun()

    # --- 7. DISPLAY LOGIC (Unchanged, but now runs *after* the spinner) ---
    if st.session_state.get('show_prediction', False):
        # We write the results into the placeholder
        with results_placeholder.container():
            st.subheader("Prediction Result")
            st.success(f"Predicted TOTAL Power Output: **{st.session_state.total_predicted_power:.2f} kW**")
            
            st.markdown("---")
            st.subheader("Calculation Breakdown")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("1. Your Plant's DC Capacity", f"{st.session_state.total_plant_dc_capacity_kW:.0f} kW")
            col2.metric("2. Base ML Prediction", f"{st.session_state.ml_predicted_power:.2f} kW")
            col3.metric("3. Wind Cooling Factor", f"{st.session_state.cooling_bonus_factor:.3f}x")
            col4.metric("4. Final Prediction (2 * 3)", f"{st.session_state.total_predicted_power:.2f} kW")

            st.info("Prediction context saved! You can now click the '🤖' button at the top to ask the AI about this result.")
        
    else:
        # Default message
        results_placeholder.info("Adjust the inputs in the sidebar and click 'Predict' to see the result.")