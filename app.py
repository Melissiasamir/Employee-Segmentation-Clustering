import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib  # use joblib for model

# Page configuration
st.set_page_config(
    page_title="Employee Segmentation Predictor",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme and styling
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stAlert { background-color: #1e3a2e; border-left: 5px solid #4caf50; }
    .result-box { padding: 20px; border-radius: 10px; margin: 10px 0; }
    .success-box { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .info-box { background-color: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
</style>
""", unsafe_allow_html=True)

# Load model function using joblib
@st.cache_resource
def load_model():
    try:
        model_path = os.path.join('Model', 'final_clustering_model.pkl')
        model = joblib.load(model_path)
        return model, None
    except Exception as e:
        return None, str(e)

# Load model
model, error = load_model()

# ============================================================================#
# SIDEBAR
# ============================================================================#

# Model status
if model is not None:
    st.sidebar.success("✅ Model loaded successfully!")
else:
    st.sidebar.error(f"❌ Model loading failed")
    if error:
        st.sidebar.caption(f"Error: {error}")

st.sidebar.markdown("---")
st.sidebar.title("👥 Employee Segmentation")
st.sidebar.markdown("---")

# How to Use
with st.sidebar.expander("📖 How to Use", expanded=True):
    st.markdown("""
    1. **Enter employee information** in the form fields
    2. **Click 'Predict Segment'** button to analyze
    3. **View the prediction** and segment details
    """)

st.sidebar.markdown("---")

# Model Info
with st.sidebar.expander("ℹ️ Model Info"):
    st.markdown("""
    **Framework:** scikit-learn
    
    **Algorithm:** K-Means Clustering (k=4)
    
    **Pipeline Components:**
    - MinMaxScaler
    - One-Hot Encoding
    - PCA (2 components)
    - K-Means clustering
    
    **Features Used for Prediction:**
    - Age, Education Level, Department, Length of Service
    - Number of Trainings, Average Training Score
    - Gender, Region, Recruitment Channel
    """)

# ============================================================================#
# MAIN CONTENT
# ============================================================================#

st.title("🎯 Employee Segmentation System")
st.markdown("*Advanced ML-powered employee classification for HR analytics*")
st.markdown("---")

if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False

# BEFORE PREDICTION: Show input form
if not st.session_state.prediction_made:
    st.subheader("📝 Enter Employee Information")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("employee_form"):
            st.markdown("### Employee Details")
            
            # Personal Information
            age = st.number_input("Age", min_value=18, max_value=70, value=30)
            education = st.selectbox("Education Level", ["Bachelor's", "Master's & above", "Below Secondary"])
            department = st.selectbox(
                "Department",
                ["Sales & Marketing", "Operations", "Technology", "Analytics", 
                 "R&D", "Procurement", "Finance", "HR", "Legal"]
            )
            length_of_service = st.number_input("Length of Service (years)", min_value=0, max_value=40, value=3)
            
            # New fields to match model
            gender = st.selectbox("Gender", ["Male", "Female"])
            region = st.selectbox("Region", ["Region_A", "Region_B", "Region_C"])
            recruitment_channel = st.selectbox("Recruitment Channel", ["Sourcing", "Referral", "Other"])
            
            st.markdown("### Performance Metrics (For Display Only)")
            previous_year_rating = st.slider("Previous Year Rating", min_value=1, max_value=5, value=3)
            no_of_trainings = st.number_input("Number of Trainings Completed", min_value=0, max_value=20, value=2)
            avg_training_score = st.slider("Average Training Score", min_value=0, max_value=100, value=70)
            KPIs_met = st.selectbox("KPIs Met More Than 80%", ["Yes", "No"])
            awards_won = st.selectbox("Awards Won", ["Yes", "No"])
            
            st.markdown("---")
            submitted = st.form_submit_button("🔍 Predict Segment", use_container_width=True, type="primary")
            
            if submitted:
                if model is None:
                    st.error("⚠️ Model not loaded. Cannot make prediction.")
                else:
                    st.session_state.inputs = {
                        'age': age, 'education': education, 'department': department,
                        'length_of_service': length_of_service, 'previous_year_rating': previous_year_rating,
                        'no_of_trainings': no_of_trainings, 'avg_training_score': avg_training_score,
                        'KPIs_met': KPIs_met, 'awards_won': awards_won,
                        'gender': gender, 'region': region, 'recruitment_channel': recruitment_channel
                    }
                    
                    # Prediction input matches model features
                    input_data = pd.DataFrame({
                        'no_of_trainings': [no_of_trainings],
                        'age': [age],
                        'length_of_service': [length_of_service],
                        'avg_training_score': [avg_training_score],
                        'education': [education],
                        'department': [department],
                        'gender': [gender],
                        'region': [region],
                        'recruitment_channel': [recruitment_channel]
                    })
                    
                    try:
                        predicted_cluster = model.predict(input_data)[0]
                        st.session_state.prediction = predicted_cluster
                        st.session_state.prediction_made = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Prediction error: {str(e)}")
                        st.info("💡 Please check that all input values are correct.")
                        st.code(f"Debug info: {str(e)}", language="text")

# AFTER PREDICTION: Show results
else:
    cluster_info = {
        0: {"name": "High Performers & Rising Stars", "description": "Exceptional performance", "risk":"Low"},
        1: {"name": "Solid Contributors", "description": "Steady reliable employees", "risk":"Low"},
        2: {"name": "Development Focus Group", "description": "Need targeted support", "risk":"Medium"},
        3: {"name": "At-Risk & Requires Intervention", "description": "Performance concerns", "risk":"High"}
    }
    predicted_cluster = st.session_state.prediction
    cluster = cluster_info.get(predicted_cluster, cluster_info[0])
    inputs = st.session_state.inputs
    
    col1, col2 = st.columns([1,1])
    with col1:
        st.subheader("📋 Employee Profile")
        profile_data = {
            "Age": inputs['age'], "Education": inputs['education'], "Department": inputs['department'],
            "Service Length": f"{inputs['length_of_service']} years", "Previous Rating": inputs['previous_year_rating'],
            "Trainings": inputs['no_of_trainings'], "Training Score": f"{inputs['avg_training_score']}%",
            "KPIs Met": inputs['KPIs_met'], "Awards Won": inputs['awards_won'],
            "Gender": inputs['gender'], "Region": inputs['region'], "Recruitment Channel": inputs['recruitment_channel']
        }
        df_profile = pd.DataFrame(list(profile_data.items()), columns=['Attribute', 'Value'])
        st.dataframe(df_profile, use_container_width=True, hide_index=True)
        
        if st.button("🔄 Analyze Another Employee", use_container_width=True):
            st.session_state.prediction_made = False
            st.rerun()
    
    with col2:
        st.subheader("🎯 Analysis Results")
        st.markdown(f"<div class='result-box success-box'><h2 style='margin:0;'>✅ {cluster['name']}</h2><p>{cluster['description']}</p></div>", unsafe_allow_html=True)
        st.markdown(f"**Retention Risk Level:** {cluster['risk']}")

st.markdown("---")
st.markdown("*Built with Streamlit and scikit-learn | Employee Segmentation ML Project 2024*")
st.caption("Model uses K-Means clustering with PCA dimensionality reduction for employee segmentation.")
