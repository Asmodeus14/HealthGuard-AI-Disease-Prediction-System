import streamlit as st
import pickle
from streamlit_option_menu import option_menu
import pandas as pd

# Configuration
st.set_page_config(page_title="HealthGuard AI", page_icon="🏥", layout="wide")

# Model Paths (Update these paths)
MODEL_PATHS = {
    'diabetes': r'D:\AICTE-Internship\Disease-prediction-model\Source\Model\diabetes_model.pkl',
    'heart': r'D:\AICTE-Internship\Disease-prediction-model\Source\Model\heart_model.pkl',
    'parkinson': r'D:\AICTE-Internship\Disease-prediction-model\Source\Model\parkinsons_model.pkl'
}
# Load models with caching
@st.cache_resource
def load_models():
    models = {}
    try:
        for disease, path in MODEL_PATHS.items():
            with open(path, 'rb') as f:
                models[disease] = pickle.load(f)
        return models
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

models = load_models()

# Sidebar Navigation
with st.sidebar:
    st.title("HealthGuard AI")
    selected = option_menu("Disease Prediction", 
                         ["Diabetes", "Heart Disease", "Parkinson's"],
                         icons=["droplet", "heart-pulse", "person"],
                         menu_icon="clipboard-pulse",
                         default_index=0)



# Helper Functions
def display_prediction(prediction, disease):
    if prediction == 1:
        st.error(f"⚠️ High probability of {disease}. Please consult a healthcare professional.")
    else:
        st.success(f"✅ No significant indication of {disease}. Maintain healthy habits!")

# Diabetes Prediction Page
if selected == "Diabetes":
    st.title("Diabetes Risk Assessment")
    with st.form("diabetes_form"):
        col1, col2 = st.columns(2)
        with col1:
            pregnancies = st.number_input("Pregnancies", 0, 17, 0)
            glucose = st.number_input("Glucose Level (mg/dL)", 70, 200, 90)
            bp = st.number_input("Blood Pressure (mmHg)", 60, 130, 70)
            skin_thickness = st.number_input("Skin Thickness (mm)", 0, 99, 20)
        
        with col2:
            insulin = st.number_input("Insulin Level (μU/ml)", 0, 846, 80)
            bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
            dpf = st.number_input("Diabetes Pedigree Function", 0.08, 2.5, 0.25)
            age = st.number_input("Age", 21, 120, 30)
        
        if st.form_submit_button("Assess Diabetes Risk"):
            input_data = [[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]]
            prediction = models['diabetes'].predict(input_data)[0]
            display_prediction(prediction, "diabetes")

# Heart Disease Prediction Page
elif selected == "Heart Disease":
    st.title("Cardiovascular Health Evaluation")
    with st.form("heart_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", 20, 100, 50)
            trestbps = st.number_input("Resting BP (mmHg)", 90, 200, 120)
            chol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
            thalach = st.number_input("Max Heart Rate", 60, 220, 150)
            oldpeak = st.number_input("ST Depression", 0.0, 6.2, 0.0)
        
        with col2:
            sex = st.selectbox("Gender", ["Male", "Female"])
            cp = st.selectbox("Chest Pain Type", [
                "Typical Angina", 
                "Atypical Angina", 
                "Non-anginal Pain", 
                "Asymptomatic"
            ])
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
            exang = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
            slope = st.selectbox("ST Segment Slope", [
                "Upsloping", 
                "Flat", 
                "Downsloping"
            ])

        if st.form_submit_button("Evaluate Heart Health"):
            # Convert categorical inputs to numerical
            sex = 1 if sex == "Male" else 0
            cp_mapping = {"Typical Angina": 0, "Atypical Angina": 1, 
                        "Non-anginal Pain": 2, "Asymptomatic": 3}
            cp = cp_mapping[cp]
            fbs = 1 if fbs == "Yes" else 0
            exang = 1 if exang == "Yes" else 0
            slope_mapping = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
            slope = slope_mapping[slope]
            
            input_data = [[age, sex, cp, trestbps, chol, fbs, 0, thalach, exang, oldpeak, slope, 0, 1]]
            prediction = models['heart'].predict(input_data)[0]
            display_prediction(prediction, "heart disease")

# Parkinson's Prediction Page
else:
    st.title("Parkinson's Disease Screening")

    with st.form("parkinson_form"):
        st.write("Voice Feature Analysis Parameters")
        col1, col2 = st.columns(2)

        with col1:
            mdvp_fo = st.number_input("Fundamental Frequency (Hz)", 80.0, 260.0, 150.0)
            mdvp_fhi = st.number_input("Maximum Frequency (Hz)", 100.0, 300.0, 200.0)
            mdvp_flo = st.number_input("Minimum Frequency (Hz)", 60.0, 150.0, 90.0)
            mdvp_jitter = st.number_input("Jitter (%)", 0.0, 0.1, 0.005, format="%.4f")
            mdvp_jitter_abs = st.number_input("Jitter (Abs)", 0.0, 0.01, 0.0005, format="%.5f")
            mdvp_rap = st.number_input("RAP", 0.0, 0.05, 0.002, format="%.4f")
            mdvp_ppq = st.number_input("PPQ", 0.0, 0.05, 0.002, format="%.4f")
            jitter_ddp = st.number_input("Jitter DDP", 0.0, 0.15, 0.006, format="%.4f")
            mdvp_shimmer = st.number_input("Shimmer", 0.0, 0.3, 0.01, format="%.3f")
            mdvp_shimmer_db = st.number_input("Shimmer (dB)", 0.0, 3.0, 0.1, format="%.3f")
            d2 = st.number_input("D2", 0.0, 4.0, 2.0)
        with col2:
            shimmer_apq3 = st.number_input("Shimmer APQ3", 0.0, 0.1, 0.01, format="%.4f")
            shimmer_apq5 = st.number_input("Shimmer APQ5", 0.0, 0.2, 0.02, format="%.4f")
            mdvp_apq = st.number_input("MDVP APQ", 0.0, 0.3, 0.03, format="%.4f")
            shimmer_dda = st.number_input("Shimmer DDA", 0.0, 0.1, 0.01, format="%.4f")
            nhr = st.number_input("Noise-to-Harmonics Ratio", 0.0, 1.0, 0.01)
            hnr = st.number_input("Harmonics-to-Noise Ratio", 0.0, 40.0, 20.0)
            rpde = st.number_input("RPDE (Nonlinear Measure)", 0.0, 1.0, 0.5)
            dfa = st.number_input("DFA (Signal Complexity)", 0.5, 0.9, 0.7)
            spread1 = st.number_input("Spread1 (Nonlinear Prop)", -8.0, 0.0, -5.0)
            spread2 = st.number_input("Spread2 (Nonlinear Prop)", 0.0, 0.5, 0.2)
            
            ppe = st.number_input("PPE (Nonlinear Dynamic)", 0.0, 0.5, 0.1)

        if st.form_submit_button("Analyze Parkinson's Risk"):
            # Define feature names
            feature_names = [
                "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP",
                "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3",
                "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "spread1",
                "spread2", "D2", "PPE"
            ]

            # Convert input values to a DataFrame (fixes feature name issue)
            input_df = pd.DataFrame([[
                mdvp_fo, mdvp_fhi, mdvp_flo, mdvp_jitter, mdvp_jitter_abs, mdvp_rap,
                mdvp_ppq, jitter_ddp, mdvp_shimmer, mdvp_shimmer_db, shimmer_apq3,
                shimmer_apq5, mdvp_apq, shimmer_dda, nhr, hnr, rpde, dfa, spread1,
                spread2, d2, ppe
            ]], columns=feature_names)

            # Make prediction
            prediction = models['parkinson'].predict(input_df)[0]

            # Display result
            display_prediction(prediction, "Parkinson's disease")

# Footer
st.markdown("---")
st.markdown("ℹ️ This tool provides risk assessments and should not replace professional medical advice.")