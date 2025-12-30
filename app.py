import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ===========================
# Page Config
# ===========================
st.set_page_config(
    page_title="Employee Segmentation Predictor",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================
# CSS Styling
# ===========================
st.markdown("""
<style>
    :root {
        --primary: #2563eb;
        --primary-light: #e9f0ff;
        --card: #0f172a;
        --muted: #94a3b8;
        --border: #1e293b;
        --shadow: 0 14px 40px rgba(15, 23, 42, 0.25);
    }
    .main { background-color: #0b1220; }
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    .card {
        background-color: var(--card);
        padding: 22px;
        border-radius: 16px;
        box-shadow: var(--shadow);
        border: 1px solid var(--border);
    }
    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 0.4rem;
    }
    .muted { color: var(--muted); }
    .section-card {
        background-color: #0d1628;
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 16px 16px 12px;
        margin-bottom: 14px;
        box-shadow: 0 10px 28px rgba(0,0,0,0.25);
    }
    .section-card h3, .section-card h4 {
        margin-bottom: 0.35rem;
    }
    .stButton>button {
        background: linear-gradient(120deg, #2563eb, #1d4ed8);
        color: white;
        border-radius: 10px;
        padding: 0.65rem 1.2rem;
        border: none;
        box-shadow: 0 10px 25px rgba(37, 99, 235, 0.2);
    }
    .stButton>button:hover { background: #1d4ed8; }
    .stSelectbox > div[data-baseweb="select"] { height: 46px; }
    label { color: #e2e8f0 !important; font-weight: 600; }
    .stNumberInput, .stSelectbox, .stTextInput, .stSlider { margin-bottom: 8px; }
    .stDataFrame { border-radius: 12px; overflow: hidden; }
    .badge {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        font-size: 12px;
        background: var(--primary-light);
        color: #1d4ed8;
        border: 1px solid #cbd5f5;
    }
    /* Slider polish */
    input[type="range"]::-webkit-slider-thumb { background: #2563eb; border: 2px solid #0b1220; }
    input[type="range"]::-moz-range-thumb { background: #2563eb; border: 2px solid #0b1220; }
    input[type="range"]::-webkit-slider-runnable-track { background: #1e293b; }
    input[type="range"]::-moz-range-track { background: #1e293b; }
</style>
""", unsafe_allow_html=True)

# ===========================
# Load Model
# ===========================
@st.cache_resource
def load_model():
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

model, error = load_model()

if model:
    st.sidebar.success("✅ Model loaded successfully!")
else:
    st.sidebar.error("❌ Model loading failed")
    if error:
        st.sidebar.caption(f"Error: {error}")

st.title("Employee Segmentation Predictor")

# ===========================
# Sidebar Info
# ===========================
with st.sidebar.expander("📖 How to Use", expanded=True):
    st.markdown("""
    1. Enter employee information in the form fields  
    2. Click 'Predict Segment' button  
    3. View the prediction and segment details in the 'Prediction Results' tab
    """)

with st.sidebar.expander("ℹ️ Model Info", expanded=True):
    st.markdown("""
    **Framework:** scikit-learn  
    **Algorithm:** K-Means Clustering  
    **Pipeline Components:**  
    - Preprocessing (StandardScaler / OneHotEncoder)  
    - PCA (dimensionality reduction)  
    - K-Means clustering  
    **Features:** age, gender, region, education, department, length_of_service, previous_year_rating, no_of_trainings, avg_training_score, KPIs_met, awards_won, recruitment_channel
    """)

# ===========================
# Tabs
# ===========================
tabs = st.tabs(["📝 Input Employee Info", "🎯 Prediction Results"])

# ===========================
# Tab 1: Input
# ===========================
with tabs[0]:
    st.header("Employee Information & Performance Metrics")
    with st.form("employee_form"):
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        # Employee Info
        st.subheader("👤 Employee Info")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=70, value=30)
            gender = st.selectbox("Gender", ["Male", "Female"], index=0)
            region = st.selectbox("Region", ["North", "South", "East", "West"], index=0)
            recruitment_channel = st.selectbox("Recruitment Channel", ["HR Referral", "LinkedIn", "Career Portal"], index=0)
        with col2:
            education = st.selectbox("Education Level", ["Bachelor's", "Master's & above", "Below Secondary"], index=0)
            department = st.selectbox("Department", ["Sales & Marketing", "Operations", "Technology", "Analytics", "R&D", "Procurement", "Finance", "HR", "Legal"], index=0)
            length_of_service = st.number_input("Length of Service (years)", min_value=0, max_value=40, value=3)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("📊 Performance Metrics")
        col3, col4 = st.columns(2)
        with col3:
            previous_year_rating = st.slider("Previous Year Rating", min_value=1, max_value=5, value=3)
            avg_training_score = st.slider("Average Training Score", min_value=0, max_value=100, value=70)
        with col4:
            no_of_trainings = st.number_input("Number of Trainings Completed", min_value=0, max_value=20, value=2)
            KPIs_met = st.selectbox("KPIs Met More Than 80%", ["Yes", "No"], index=0)
            awards_won = st.selectbox("Awards Won", ["Yes", "No"], index=0)
        st.markdown("</div>", unsafe_allow_html=True)

        submitted = st.form_submit_button("🔍 Predict Segment")
        if submitted:
            if model is None:
                st.error("⚠️ Model not loaded. Cannot make prediction.")
            else:
                input_data = pd.DataFrame({
                    'age': [age],
                    'education': [education],
                    'department': [department],
                    'length_of_service': [length_of_service],
                    'no_of_trainings': [no_of_trainings],
                    'previous_year_rating': [previous_year_rating],
                    'KPIs_met_more_than_80': [1 if KPIs_met=="Yes" else 0],
                    'awards_won': [1 if awards_won=="Yes" else 0],
                    'avg_training_score': [avg_training_score],
                    'gender': [gender],
                    'region': [region],
                    'recruitment_channel': [recruitment_channel]
                })
                try:
                    predicted_cluster = model.predict(input_data)[0]
                    st.session_state.prediction_made = True
                    st.session_state.prediction = predicted_cluster
                    st.session_state.inputs = input_data
                    st.session_state.inputs_dict = input_data.iloc[0].to_dict()
                    st.success("✅ Prediction made! Switch to 'Prediction Results' tab to view details.")
                except Exception as e:
                    st.error(f"❌ Prediction error: {str(e)}")

# ===========================
# Tab 2: Prediction Results
# ===========================
with tabs[1]:
    st.header("Prediction Results")
    if 'prediction_made' in st.session_state and st.session_state.prediction_made:
        predicted_cluster = st.session_state.prediction
        inputs = st.session_state.inputs
        inputs_row = inputs.iloc[0]

        # Cluster details
        cluster_info = {
            0: {"name":"High Performers & Rising Stars","description":"Exceptional performance, strong growth trajectory","color":"#4caf50"},
            1: {"name":"Solid Contributors","description":"Steady and reliable performance","color":"#2196f3"},
            2: {"name":"Development Focus Group","description":"Needs guidance and performance improvement","color":"#ff9800"},
            3: {"name":"At-Risk & Requires Intervention","description":"Significant performance concerns","color":"#f44336"}
        }
        cluster = cluster_info.get(predicted_cluster, cluster_info[0])

        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        # Prediction box
        st.markdown(f"<div style='background-color:{cluster['color']}20; padding:20px; border-radius:10px;'>"
                    f"<h2 style='color:{cluster['color']}; margin:0'>{cluster['name']}</h2>"
                    f"<p>{cluster['description']}</p></div>", unsafe_allow_html=True)

        # Cluster legend
        st.caption("Segment color key")
        legend_cols = st.columns(len(cluster_info))
        for idx, (c_id, c) in enumerate(cluster_info.items()):
            with legend_cols[idx]:
                st.markdown(
                    f"<div style='border:1px solid #1e293b; border-radius:12px; padding:10px; background-color:{c['color']}10'>"
                    f"<strong>Cluster {c_id}</strong><br><span style='color:{c['color']}'>{c['name']}</span>"
                    f"</div>", unsafe_allow_html=True
                )
        st.markdown("</div>", unsafe_allow_html=True)

        # Inputs displayed alongside results
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Employee Profile")
        feature_labels = {
            "age": "Age",
            "gender": "Gender",
            "region": "Region",
            "education": "Education Level",
            "department": "Department",
            "recruitment_channel": "Recruitment Channel",
            "length_of_service": "Length of Service (years)",
            "previous_year_rating": "Previous Year Rating",
            "no_of_trainings": "Number of Trainings Completed",
            "avg_training_score": "Average Training Score",
            "KPIs_met_more_than_80": "KPIs Met More Than 80%",
            "awards_won": "Awards Won",
        }

        profile_rows = []
        for key, label in feature_labels.items():
            val = inputs_row[key]
            if key in ["KPIs_met_more_than_80", "awards_won"]:
                val = "Yes" if val == 1 else "No"
            profile_rows.append({"Feature": label, "Value": val})

        profile_df = pd.DataFrame(profile_rows)
        st.dataframe(profile_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Quick Metrics
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Quick Metrics")
        col1, col2, col3 = st.columns(3)
        performance_score = (
            inputs_row['previous_year_rating']*20
            + inputs_row['avg_training_score']*0.5
            + (20 if inputs_row['KPIs_met_more_than_80']==1 else 0)
            + (15 if inputs_row['awards_won']==1 else 0)
        )/1.55
        with col1:
            st.metric("Performance Score", f"{performance_score:.1f}%")
        with col2:
            st.metric("Experience", f"{inputs_row['length_of_service']} yrs")
        with col3:
            st.metric("Training Activity", f"{inputs_row['no_of_trainings']} courses")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("ℹ️ Please submit employee data in the 'Input Employee Info' tab first.")
