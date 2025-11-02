# app.py

import streamlit as st
import pandas as pd
import pickle
import time 

# --- Page Configuration ---
st.set_page_config(
    page_title="Solar Power Predictor",
    page_icon="☀️",
    layout="wide" 
)

# --- Constants ---
# This is our "EEE" baseline.
# From the training data, one inverter handles a max of ~1450 kW of DC power.
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

# --- Chatbot Response Function ---
def get_bot_response(user_question):
    """Generates a simple, rule-based response for the chatbot."""
    text = user_question.lower().strip()
    
    if 'hello' in text or 'hi' in text:
        return "Hello! I'm a bot. How can I help you understand this project?"
    if 'what is this' in text or 'project' in text:
        return "This is a machine learning app that predicts solar power generation. You can input the plant's DC capacity (panels) and weather conditions to forecast the total AC power output."
    if 'model' in text or 'how' in text:
        return "It uses a Random Forest Regressor, a popular and reliable machine learning model. It was trained in Python using Scikit-learn on historical data from a real solar plant."
    if 'features' in text or 'data' in text:
        return "The model was trained on 5 key weather/time features: Ambient Temperature, Module Temperature, Sunlight (Irradiation), Hour of the Day, and Month."
    if 'capacity' in text or 'plant' or 'calculation' in text:
        return f"This is the 'EEE' part! The model predicts AC power for a 'standard inverter' with a {STANDARD_INVERTER_DC_CAPACITY_KW} kW DC capacity. We first calculate your plant's total DC capacity (Panels x Panel Watts). Then, we find a scaling factor (Your_Plant_DC / {STANDARD_INVERTER_DC_CAPACITY_KW}). The final prediction is the model's output multiplied by this scaling factor. This scales the prediction to your exact plant size."
    if 'thank' in text:
        return "You're welcome! Feel free to ask more questions."
        
    return "Sorry, I'm not sure how to answer that. Try asking about the 'project', 'model', 'features', or 'calculation'."


# --- Page Title ---
st.title("☀️ Solar Power Generation Predictor")

# --- Page Layout (Tabs) ---
tab1, tab2 = st.tabs(["Power Predictor", "🤖 Explainer Chatbot"])


# --- Sidebar for User Input ---
st.sidebar.header("Input Conditions")
st.sidebar.markdown("### 1. Plant DC Capacity")

# Plant Capacity Inputs
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

# Weather/Time sliders
ambient_temp = st.sidebar.slider("Ambient Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0, step=0.1)
module_temp = st.sidebar.slider("Module Temperature (°C)", min_value=-10.0, max_value=80.0, value=35.0, step=0.1)
irradiation = st.sidebar.slider("Irradiation (Sunlight Intensity)", min_value=0.0, max_value=1.2, value=0.5, step=0.01)

hour = st.sidebar.slider("Hour of the Day (0-23)", min_value=0, max_value=23, value=12, step=1)
# <-- NEW: Added description for the hour slider -->
st.sidebar.markdown("*(0-12 = 12am-12pm; 13-23 = 1pm-11pm)*")

month = st.sidebar.slider("Month of the Year (1-12)", min_value=1, max_value=12, value=6, step=1)


# --- Tab 1: The Predictor ---
with tab1:
    st.header("Predict Total Power Output")
    st.markdown("""
        Use the sidebar to input your plant's **DC capacity** (panels) and the **weather conditions**, then click 'Predict'.
    """)

    if model and model_columns:
        # --- Prediction Logic ---
        if st.sidebar.button("Predict Power Output"):
            
            # 1. Create a dictionary for the model
            input_data = {
                'AMBIENT_TEMPERATURE': ambient_temp,
                'MODULE_TEMPERATURE': module_temp,
                'IRRADIATION': irradiation,
                'Hour': hour,
                'Month': month
            }
            
            # 2. Convert to a DataFrame for the model
            input_df = pd.DataFrame([input_data])
            input_df = input_df[model_columns] 

            # 3. Make the base prediction (for one standard inverter)
            prediction = model.predict(input_df)
            single_inverter_ac_power = prediction[0]
            
            # 4. Handle night-time predictions
            if irradiation == 0.0 or hour < 5 or hour > 19:
                single_inverter_ac_power = 0.0
            
            # 5. Perform the "EEE Calculation"
            
            # Calculate total DC capacity of the user's plant in kW
            total_plant_dc_capacity_kW = (total_panels * panel_capacity_W) / 1000.0
            
            # Calculate the scaling factor
            if STANDARD_INVERTER_DC_CAPACITY_KW == 0:
                 scaling_factor = 0.0
            else:
                 scaling_factor = total_plant_dc_capacity_kW / STANDARD_INVERTER_DC_CAPACITY_KW
            
            # 6. Calculate the final total power
            total_predicted_power = single_inverter_ac_power * scaling_factor

            # --- Display the Result ---
            st.subheader("Prediction Result")
            
            with st.spinner('Calculating...'):
                time.sleep(1) 

            # Display the main result
            st.success(f"Predicted TOTAL Power Output: **{total_predicted_power:.2f} kW**")
            
            st.markdown("---")
            
            # Show the calculation breakdown
            st.subheader("Calculation Breakdown")
            col1, col2, col3 = st.columns(3)
            col1.metric("Your Plant's Total DC Capacity", f"{total_plant_dc_capacity_kW:.0f} kW")
            col2.metric("Base Model Prediction", f"{single_inverter_ac_power:.2f} kW", help="Predicted AC power for one standard inverter block.")
            col3.metric("Plant Scaling Factor", f"{scaling_factor:.2f}x", help="Your Plant's DC Capacity / Standard Inverter DC Capacity")

            st.info(f"""
                **How this was calculated:**
                1.  Your total plant size is **{total_plant_dc_capacity_kW:.0f} kW** (DC) (from {total_panels} panels $\times$ {panel_capacity_W}W).
                2.  Based on the weather, the model predicted **{single_inverter_ac_power:.2f} kW** (AC) for a standard **{STANDARD_INVERTER_DC_CAPACITY_KW} kW** (DC) block.
                3.  Your plant is **{scaling_factor:.2f} times** the size of the standard block.
                4.  **Final Prediction:** {single_inverter_ac_power:.2f} kW $\times$ {scaling_factor:.2f} = **{total_predicted_power:.2f} kW**
            """)
            
        else:
            st.info("Adjust the inputs in the sidebar and click 'Predict' to see the result.")


# --- Tab 2: The Chatbot ---
with tab2:
    st.header("🤖 Project Explainer Bot")
    st.markdown("Ask me questions about this project!")

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! Ask me about the 'project', 'model', 'features', or 'calculation'."}
        ]

    # Display prior chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get user input
    if prompt := st.chat_input("What would you like to know?"):
        
        # Add user message to history and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get and display bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                time.sleep(0.5) # Simulate thinking
                response = get_bot_response(prompt)
                st.markdown(response)
        
        # Add bot response to history
        st.session_state.messages.append({"role": "assistant", "content": response})