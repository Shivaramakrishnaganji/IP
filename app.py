# app.py

import streamlit as st
import pandas as pd
import pickle
import time 
import os
import google.generativeai as genai
from dotenv import load_dotenv  # Import dotenv

# --- Load environment variables from .env file ---
load_dotenv()  # Load the .env file

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


# --- LAYOUT: Title and Chatbot Button in Columns ---
col1, col2 = st.columns([3, 1]) # Give 3/4 space to title, 1/4 to button

with col1:
    st.title("☀️ Solar Power Generation Predictor")

# --- Chatbot Popover (the "layer") ---
with col2:
    # We add a little space to vertically center it
    st.write("") 
    st.write("")
    with st.popover("🤖 AI Explainer Assistant", use_container_width=True):
        
        st.markdown("Ask me questions about this project!")

        # --- Load the API key from os.environ ---
        api_key = os.environ.get("GEMINI_API_KEY")

        if not api_key:
            st.warning("Could not find Google AI Studio API Key. Please make sure you have a `.env` file with your `GEMINI_API_KEY`.")
        else:
            try:
                genai.configure(api_key=api_key)
            except Exception as e:
                st.error(f"Error configuring Google AI: {e}")
                
            # --- *** THIS IS THE ERROR FIX *** ---
            # The correct model name is 'gemini-pro'
            model_name = "gemini-pro" 
            
            # Set up the system prompt
            system_prompt = f"""
            You are an AI expert assistant for a college student's 'Solar Power Predictor' project expo.
            Your **ONLY** purpose is to answer questions about this project.
            
            The project themes are:
            - Solar power generation
            - The user's prediction results
            - Machine Learning (Random Forest)
            - The technologies used (Python, Streamlit, Scikit-learn, Google Gemini)
            - Electrical engineering concepts related to solar energy.

            **YOUR RULES:**
            1.  **STAY ON TOPIC:** Only answer questions related to the themes above.
            2.  **REFUSE OTHER TOPICS:** If the user asks about anything else (e.g., "write me a poem," "what is the capital of France"), you MUST politely refuse. Say: "I am an AI assistant for this solar project. I can only answer questions related to solar power, machine learning, or this application."
            3.  **USE CONTEXT:** If the user asks about their prediction, use this data to inform your answer:
                - **Prediction Inputs:** {st.session_state.get('last_prediction', {}).get('inputs', 'No prediction yet')}
                - **Prediction Calculation:** {st.session_state.get('last_prediction', {}).get('calculation', 'No prediction yet')}
            4.  **Be concise and clear.**
            """

            # Initialize the chat model in session state
            if "chat_model" not in st.session_state:
                try:
                    generative_model = genai.GenerativeModel(
                        model_name=model_name,
                        system_instruction=system_prompt
                    )
                    st.session_state.chat_model = generative_model.start_chat(history=[])
                except Exception as e:
                    st.error(f"Error starting chat model: {e}")
                    st.session_state.chat_model = None
            
            # Initialize chat history
            if "messages" not in st.session_state:
                st.session_state.messages = [
                    {"role": "assistant", "content": "Hi! Run a prediction, then ask me about it. (e.g., 'Why is my prediction low?')"}
                ]

            # Display prior chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Get user input
            if prompt := st.chat_input("Ask about your prediction..."):
                if not st.session_state.chat_model:
                    st.error("Chat model is not initialized. Please check your API key.")
                else:
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    with st.chat_message("assistant"):
                        with st.spinner("AI is thinking..."):
                            try:
                                response = st.session_state.chat_model.send_message(prompt)
                                bot_response = response.text
                                st.markdown(bot_response)
                                st.session_state.messages.append({"role": "assistant", "content": bot_response})
                            
                            except Exception as e:
                                st.error(f"An error occurred with the Gemini API. {e}")


# --- Sidebar for User Input ---
# This part is all the same
st.sidebar.header("Input Conditions")
st.sidebar.markdown("### 1. Plant DC Capacity")

total_panels = st.sidebar.number_input(
    label="Total Number of Solar Panels",
    min_value=1,
    value=30000,
    step=100
)
panel_capacity_W = st.sidebar.number_input(
    label="Single Panel Capacity (in Watts)",
    min_value=1,
    value=330,
    step=5,
    help="The wattage (e.g., 330W, 450W) of one solar panel."
)

st.sidebar.markdown("### 2. Weather & Time")

ambient_temp = st.sidebar.slider("Ambient Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0, step=0.1)
module_temp = st.sidebar.slider("Module Temperature (°C)", min_value=-10.0, max_value=80.0, value=35.0, step=0.1)
irradiation = st.sidebar.slider("Irradiation (Sunlight Intensity)", min_value=0.0, max_value=1.2, value=0.5, step=0.01)
wind_speed = st.sidebar.slider("Wind Speed (m/s)", min_value=0.0, max_value=25.0, value=5.0, step=0.1,
                               help="This is a simulation. Higher wind will apply a small efficiency bonus to simulate panel cooling.")
hour = st.sidebar.slider("Hour of the Day (0-23)", min_value=0, max_value=23, value=12, step=1)
st.sidebar.markdown("*(0-12 = 12am-12pm; 13-23 = 1pm-11pm)*")
month = st.sidebar.slider("Month of the Year (1-12)", min_value=1, max_value=12, value=6, step=1)


# --- MAIN PAGE: The Predictor ---
# This is no longer inside a tab
st.header("Predict Total Power Output")
st.markdown("""
    Use the sidebar to input your plant's **DC capacity** (panels) and the **weather conditions**, then click 'Predict'.
""")

if model and model_columns:
    if st.sidebar.button("Predict Power Output"):
        
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

        st.subheader("Prediction Result")
        st.success(f"Predicted TOTAL Power Output: **{total_predicted_power:.2f} kW**")
        
        st.markdown("---")
        st.subheader("Calculation Breakdown")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("1. Your Plant's DC Capacity", f"{total_plant_dc_capacity_kW:.0f} kW")
        col2.metric("2. Base ML Prediction", f"{ml_predicted_power:.2f} kW")
        col3.metric("3. Wind Cooling Factor", f"{cooling_bonus_factor:.3f}x")
        col4.metric("4. Final Prediction (2 * 3)", f"{total_predicted_power:.2f} kW")

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
        st.info("Prediction context saved! You can now click the '🤖' button at the top to ask the AI about this result.")
        
    else:
        st.info("Adjust the inputs in the sidebar and click 'Predict' to see the result.")