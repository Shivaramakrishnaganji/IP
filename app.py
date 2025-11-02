# app.py

import streamlit as st
import pandas as pd
import pickle
import time # To simulate a 'thinking' process

# --- Page Configuration ---
# Set the page title and a fun emoji
st.set_page_config(
    page_title="Solar Power Predictor",
    page_icon="☀️"
)

# --- Model Loading ---
# We use st.cache_resource to load the model only once
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

# --- Page Title and Description ---
if model:
    st.title("☀️ Solar Power Generation Predictor")
    st.markdown("""
        Welcome to the Solar Power Predictor! This app uses a Machine Learning model 
        (Random Forest) to predict the **AC power output (in kWh)** of a solar plant 
        based on the current weather conditions.
        
        Use the sliders in the sidebar to input the conditions and click 'Predict'.
    """)

    # --- Sidebar for User Input ---
    st.sidebar.header("Input Weather Conditions")

    # Create sliders for user input.
    # We use realistic min/max/default values based on a typical dataset.
    ambient_temp = st.sidebar.slider("Ambient Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0, step=0.1)
    module_temp = st.sidebar.slider("Module Temperature (°C)", min_value=-10.0, max_value=80.0, value=35.0, step=0.1)
    irradiation = st.sidebar.slider("Irradiation (Sunlight Intensity)", min_value=0.0, max_value=1.2, value=0.5, step=0.01)
    hour = st.sidebar.slider("Hour of the Day (0-23)", min_value=0, max_value=23, value=12, step=1)
    month = st.sidebar.slider("Month of the Year (1-12)", min_value=1, max_value=12, value=6, step=1)

    # --- Prediction Logic ---
    if st.sidebar.button("Predict Power Output"):
        
        # 1. Create a dictionary of the inputs
        input_data = {
            'AMBIENT_TEMPERATURE': ambient_temp,
            'MODULE_TEMPERATURE': module_temp,
            'IRRADIATION': irradiation,
            'Hour': hour,
            'Month': month
        }
        
        # 2. Convert to a DataFrame, ensuring the column order matches the model's training
        input_df = pd.DataFrame([input_data])
        input_df = input_df[model_columns] # This is the critical step!

        # 3. Make the prediction
        prediction = model.predict(input_df)
        
        # Get the single prediction value
        predicted_power = prediction[0]
        
        # Handle night-time predictions (power can't be negative or high at night)
        if irradiation == 0.0 or hour < 5 or hour > 19:
            predicted_power = 0.0

        # --- Display the Result ---
        st.subheader("Prediction Result")
        
        # Add a little "thinking" spinner
        with st.spinner('Calculating...'):
            time.sleep(1) # Simulate model thinking

        # Display the result in a nice "success" box
        st.success(f"Predicted AC Power Output: **{predicted_power:.2f} kWh**")
        
        # Add a simple chart for visual appeal
        st.bar_chart(pd.DataFrame({'Prediction': [predicted_power]}, index=['kWh']))

        st.info(f"**How to interpret this:** Based on your inputs, the model predicts the solar plant will generate approximately {predicted_power:.2f} kWh at this moment.")
        
    else:
        st.info("Adjust the sliders in the sidebar and click 'Predict' to see the result.")