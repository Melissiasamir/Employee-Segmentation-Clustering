import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# 1. Set page config (MUST be the first streamlit command)
st.set_page_config(
    page_title="Employee Segmentation Predictor",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. CSS Styling
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stAlert { background-color: #1e3a2e; border-left: 5px solid #4caf50; }
    .result-box { padding: 20px; border-radius: 10px; margin: 10px 0; }
    .success-box { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
</style>
""", unsafe_allow_html=True)

# 3. Model Loading Function
@st.cache_resource
def load_model():
    # We check for both .joblib and .pkl for safety
    joblib_path = os.path.join('Model', 'final_clustering_model.joblib')
    pickle_path = os.path.join('Model', 'final_clustering_model.pkl')
    
    try:
        if os.path.exists(joblib_path):
            model = joblib.load(joblib_path)
            return model, None
        elif os.path.exists(pickle_path):
            import pickle
            with open(pickle_path, 'rb') as f:
                model = pickle.load(f)
            return model, None
        else:
            return None, "Model file not found in 'Model/' folder."
    except Exception as e:
        return None, str(e)

# 4. Initialize the model
model, error = load_model()
# ============================================================================
# SIDEBAR
# ============================================================================

# Model status
if model is not None:
    st.sidebar.success("✅ Model loaded successfully!")
else:
    st.sidebar.error(f"❌ Model loading failed")
    if error:
        st.sidebar.caption(f"Error: {error}")

st.sidebar.markdown("---")

# Main section
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
    - StandardScaler (feature scaling)
    - PCA (dimensionality reduction)
    - K-Means clustering
    
    **Features Used:**
    - Age
    - Education Level
    - Department
    - Length of Service
    - Number of Trainings
    - Previous Year Rating
    - KPIs Met (>80%)
    - Awards Won
    - Average Training Score
    """)

# ============================================================================
# MAIN CONTENT
# ============================================================================

st.title("🎯 Employee Segmentation System")
st.markdown("*Advanced ML-powered employee classification for HR analytics*")
st.markdown("---")

# Check if form has been submitted (stored in session state)
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False

# BEFORE PREDICTION: Show input form
if not st.session_state.prediction_made:
    st.subheader("📝 Enter Employee Information")
    
    # Center the form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.form("employee_form"):
            st.markdown("### Employee Details")
            
            # Personal Information
            age = st.number_input("Age", min_value=18, max_value=70, value=30)
            
            education = st.selectbox(
                "Education Level",
                ["Bachelor's", "Master's & above", "Below Secondary"]
            )
            
            department = st.selectbox(
                "Department",
                ["Sales & Marketing", "Operations", "Technology", "Analytics", 
                 "R&D", "Procurement", "Finance", "HR", "Legal"]
            )
            
            length_of_service = st.number_input(
                "Length of Service (years)",
                min_value=0, max_value=40, value=3
            )
            
            st.markdown("### Performance Metrics")
            
            previous_year_rating = st.slider(
                "Previous Year Rating",
                min_value=1, max_value=5, value=3
            )
            
            no_of_trainings = st.number_input(
                "Number of Trainings Completed",
                min_value=0, max_value=20, value=2
            )
            
            avg_training_score = st.slider(
                "Average Training Score",
                min_value=0, max_value=100, value=70
            )
            
            KPIs_met = st.selectbox(
                "KPIs Met More Than 80%",
                ["Yes", "No"]
            )
            
            awards_won = st.selectbox(
                "Awards Won",
                ["Yes", "No"]
            )
            
            st.markdown("---")
            
            # Submit button
            submitted = st.form_submit_button(
                "🔍 Predict Segment",
                use_container_width=True,
                type="primary"
            )
            
            if submitted:
                if model is None:
                    st.error("⚠️ Model not loaded. Cannot make prediction.")
                else:
                    # Store inputs in session state
                    st.session_state.inputs = {
                        'age': age,
                        'education': education,
                        'department': department,
                        'length_of_service': length_of_service,
                        'previous_year_rating': previous_year_rating,
                        'no_of_trainings': no_of_trainings,
                        'avg_training_score': avg_training_score,
                        'KPIs_met': KPIs_met,
                        'awards_won': awards_won
                    }
                    
                    # Make prediction
                    try:
                        # Create input dataframe matching the exact features the model expects
                        # The model expects these columns in this order
                        input_data = pd.DataFrame({
                            'age': [age],
                            'education': [education],
                            'department': [department],
                            'length_of_service': [length_of_service],
                            'no_of_trainings': [no_of_trainings],
                            'previous_year_rating': [previous_year_rating],
                            'KPIs_met_more_than_80': [1 if KPIs_met == "Yes" else 0],
                            'awards_won': [1 if awards_won == "Yes" else 0],
                            'avg_training_score': [avg_training_score]
                        })
                        
                        # Predict using the pipeline (it handles all preprocessing)
                        predicted_cluster = model.predict(input_data)[0]
                        
                        # Store prediction
                        st.session_state.prediction = predicted_cluster
                        st.session_state.prediction_made = True
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Prediction error: {str(e)}")
                        st.info("💡 Please check that all input values are correct.")
                        st.code(f"Debug info: {str(e)}", language="text")

# AFTER PREDICTION: Show results
else:
    # Cluster definitions (based on typical HR segmentation patterns)
    cluster_info = {
        0: {
            "name": "High Performers & Rising Stars",
            "description": "Employees with exceptional performance metrics and strong development trajectory",
            "characteristics": [
                "High average training scores (typically 75+)",
                "Consistently meets or exceeds KPIs (>80%)",
                "Multiple awards and recognitions",
                "Strong previous year ratings (4-5)",
                "Active in training and development programs"
            ],
            "risk": "Low",
            "color": "#4caf50",
            "recommendations": [
                "Fast-track for leadership development programs",
                "Provide challenging stretch assignments and projects",
                "Ensure competitive compensation and benefits package",
                "Assign as mentors for junior employees",
                "Regular career progression discussions",
                "Consider for critical business initiatives"
            ]
        },
        1: {
            "name": "Solid Contributors",
            "description": "Reliable employees with steady, consistent performance",
            "characteristics": [
                "Moderate to good training scores (60-75)",
                "Generally meets KPIs with occasional variations",
                "Stable tenure and reliable work output",
                "Average previous year ratings (3-4)",
                "Regular but moderate training participation"
            ],
            "risk": "Low",
            "color": "#2196f3",
            "recommendations": [
                "Maintain current engagement and recognition practices",
                "Provide skill enhancement opportunities for career growth",
                "Encourage cross-functional collaboration",
                "Recognize and appreciate long-term contributions",
                "Offer lateral movement opportunities",
                "Support work-life balance initiatives"
            ]
        },
        2: {
            "name": "Development Focus Group",
            "description": "Employees requiring targeted support and performance improvement",
            "characteristics": [
                "Below-average training scores (<60)",
                "Inconsistent KPI achievement",
                "Limited awards or recognition",
                "Lower previous year ratings (2-3)",
                "May show capability but need guidance"
            ],
            "risk": "Medium",
            "color": "#ff9800",
            "recommendations": [
                "Implement structured performance improvement plans",
                "Assign dedicated mentors or coaches",
                "Provide targeted skill development training",
                "Set clear, achievable short-term goals",
                "Conduct bi-weekly check-ins with direct managers",
                "Address any barriers to performance (workload, resources, clarity)",
                "Offer support resources (training budget, time allocation)"
            ]
        },
        3: {
            "name": "At-Risk & Requires Intervention",
            "description": "Employees showing significant performance concerns or disengagement",
            "characteristics": [
                "Very low training scores and participation",
                "Consistently fails to meet KPIs",
                "No recent awards or recognitions",
                "Poor previous year ratings (1-2)",
                "Potential signs of disengagement or capability mismatch"
            ],
            "risk": "High",
            "color": "#f44336",
            "recommendations": [
                "Immediate manager intervention required",
                "Schedule urgent one-on-one meetings to understand issues",
                "Create formal Performance Improvement Plan (PIP) with timeline",
                "Assess role fit - consider reassignment if skill mismatch",
                "Investigate workplace issues (conflict, burnout, personal challenges)",
                "Provide access to employee assistance programs",
                "Document all interventions and progress",
                "Consider if role expectations are realistic and achievable",
                "If no improvement after intervention period, initiate exit process"
            ]
        }
    }
    
    # Get prediction result
    predicted_cluster = st.session_state.prediction
    cluster = cluster_info.get(predicted_cluster, cluster_info[0])
    inputs = st.session_state.inputs
    
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    # LEFT COLUMN: Input Summary
    with col1:
        st.subheader("📋 Employee Profile")
        
        # Display input summary in a nice box
        profile_data = {
            "Age": inputs['age'],
            "Education": inputs['education'],
            "Department": inputs['department'],
            "Service Length": f"{inputs['length_of_service']} years",
            "Previous Rating": inputs['previous_year_rating'],
            "Trainings": inputs['no_of_trainings'],
            "Training Score": f"{inputs['avg_training_score']}%",
            "KPIs Met": inputs['KPIs_met'],
            "Awards Won": inputs['awards_won']
        }
        
        # Create a styled dataframe
        df_profile = pd.DataFrame(list(profile_data.items()), columns=['Attribute', 'Value'])
        st.dataframe(df_profile, use_container_width=True, hide_index=True)
        
        # Reset button
        if st.button("🔄 Analyze Another Employee", use_container_width=True):
            st.session_state.prediction_made = False
            st.rerun()
    
    # RIGHT COLUMN: Prediction Results
    with col2:
        st.subheader("🎯 Analysis Results")
        
        # Main prediction result
        st.markdown(f"""
        <div class="result-box success-box">
            <h2 style='margin:0;'>✅ {cluster['name']}</h2>
            <p style='margin:5px 0 0 0; font-size: 1.1em;'>{cluster['description']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Risk level indicator
        risk_colors = {
            "Low": "#4caf50",
            "Medium": "#ff9800",
            "High": "#f44336"
        }
        
        st.markdown("### 📊 Retention Risk Level")
        risk_color = risk_colors.get(cluster['risk'], "#gray")
        st.markdown(f"""
        <div style='text-align: center; padding: 20px; background-color: {risk_color}20; border-radius: 10px; border: 2px solid {risk_color};'>
            <h1 style='color: {risk_color}; margin: 0;'>{cluster['risk']}</h1>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Detailed Analysis (expandable)
        with st.expander("📈 View Detailed Analysis", expanded=True):
            st.markdown(f"**Predicted Cluster:** {predicted_cluster}")
            st.markdown("**Model:** K-Means with PCA preprocessing")
            
            st.markdown("#### Key Characteristics:")
            for char in cluster['characteristics']:
                st.markdown(f"• {char}")
            
            st.markdown("#### 📋 HR Action Items:")
            for idx, rec in enumerate(cluster['recommendations'], 1):
                st.markdown(f"{idx}. {rec}")
        
        # Additional insights section
        st.markdown("---")
        st.markdown("### 💡 Quick Insights")
        
        # Calculate some basic insights
        performance_score = (
            inputs['previous_year_rating'] * 20 + 
            inputs['avg_training_score'] * 0.5 + 
            (20 if inputs['KPIs_met'] == "Yes" else 0) +
            (15 if inputs['awards_won'] == "Yes" else 0)
        ) / 1.55
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Performance Score", f"{performance_score:.1f}%")
        with col_b:
            st.metric("Experience", f"{inputs['length_of_service']} yrs")
        with col_c:
            st.metric("Training Activity", f"{inputs['no_of_trainings']} courses")

st.markdown("---")
st.markdown("*Built with Streamlit and scikit-learn | Employee Segmentation ML Project 2024*")
st.caption("Model uses K-Means clustering with PCA dimensionality reduction for employee segmentation.")