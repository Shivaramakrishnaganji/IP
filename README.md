<div align="center">

# ☀️ Solar Power Generation Predictor

**An Intelligent Machine Learning Web Application for Predicting Solar Energy Output with AI Assistant**

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=flat-square&logo=openai&logoColor=white)](https://openai.com/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)

</div>

---

## ✨ Features

- **Accurate Power Prediction** — Predicts AC Power output based on dynamic environmental inputs using a trained Random Forest Regressor model.
- **🤖 AI Explainer Assistant** — Integrated ChatGPT (GPT-4o-mini) bot that explains predictions, answers questions about the calculations, and acts as a project expert.
- **Dynamic Environmental Controls** — Granular sliders for Ambient Temperature, Module Temperature, Irradiation, Wind Speed, Time of Day, and Month.
- **Simulated Cooling Factor** — Incorporates wind speed data to simulate panel cooling efficiency bonuses dynamically.
- **Interactive Web Interface** — Built with Streamlit, featuring a modern, responsive layout with sidebars, popovers, and animated calculation spinners.
- **Customizable Plant Configuration** — Adjust total number of solar panels and single panel wattage capacity to simulate any plant size.
- **Comprehensive Data Breakdown** — Detailed metrics view showing Plant DC Capacity, Base ML Prediction, Wind Cooling Factor, and the Final Prediction.

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend/UI** | Streamlit |
| **Machine Learning** | Scikit-Learn (RandomForestRegressor), Pandas, NumPy |
| **AI Integration** | OpenAI API (gpt-4o-mini) |
| **Environment Management**| python-dotenv |
| **Data Serialization** | Pickle |

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have Python 3.8+ installed. You also need the historical dataset files (`Plant_1_Weather_Sensor_Data.csv` and `Plant_1_Generation_Data.csv`) in the root directory.

### 2. Installation
```bash
# Clone the repository
git clone <repository-url>
cd <repository-directory>

# Install the required dependencies
pip install -r requirements.txt
```

### 3. Environment Setup
Create a `.env` file in the root directory and add your OpenAI API key for the Explainer Assistant:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Train the Model
Before running the app, you must train the Random Forest model and generate the `.pkl` files.
```bash
python train_model.py
```
*This script will output model evaluation metrics (R², MAE, RMSE) and save `solar_model.pkl` and `model_columns.pkl`.*

### 5. Run the Application
```bash
streamlit run app.py
```
*The app will automatically open in your default browser at `http://localhost:8501`.*

## 🧠 Model Details

The predictive model uses a **Random Forest Regressor** trained on historical solar plant generation and weather sensor data. 
- **Features Used**: Ambient Temperature, Module Temperature, Irradiation, Hour, and Month.
- **Target Variable**: AC Power Output.
- **Data Preprocessing**: Automatically filters out nighttime/zero-irradiation entries and drops missing values to optimize training accuracy.

## 📄 License

MIT License
