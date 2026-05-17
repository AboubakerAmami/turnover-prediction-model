import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# ══════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Turnover Prediction System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════
# CUSTOM CSS - IMPROVED READABILITY
# ══════════════════════════════════════════════════════════════

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .risk-high {
        background-color: #fff5f5;
        border-left-color: #dc3545;
        color: #2d2d2d;
    }
    .risk-high h2 {
        color: #dc3545;
        font-weight: bold;
    }
    .risk-high h3 {
        color: #721c24;
        font-weight: bold;
    }
    .risk-high p, .risk-high ul, .risk-high li {
        color: #2d2d2d;
        font-size: 1rem;
    }
    .risk-low {
        background-color: #f0fff4;
        border-left-color: #28a745;
        color: #2d2d2d;
    }
    .risk-low h2 {
        color: #28a745;
        font-weight: bold;
    }
    .risk-low h3 {
        color: #155724;
        font-weight: bold;
    }
    .risk-low p, .risk-low ul, .risk-low li {
        color: #2d2d2d;
        font-size: 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 1.2rem;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #155a8a;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# LOAD MODEL, CONFIG, AND ENCODERS
# ══════════════════════════════════════════════════════════════

@st.cache_resource
def load_model():
    """Load the trained model, config, and label encoders"""
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('config.pkl', 'rb') as f:
            config = pickle.load(f)
        with open('label_encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        return model, config, encoders
    except FileNotFoundError:
        st.warning("⚠️ Model files not found! Upload model.pkl, config.pkl, and label_encoders.pkl")
        return None, None, None

model, config, encoders = load_model()

# Get actual direction and subdept names
if encoders:
    direction_names = encoders['le_direction'].classes_.tolist()
    subdept_names = encoders['le_subdept'].classes_.tolist()
else:
    direction_names = ['Direction A', 'Direction B', 'Direction C', 'Direction D', 'Direction E']
    subdept_names = ['Dept 1', 'Dept 2', 'Dept 3', 'Dept 4', 'Dept 5']

# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════

st.markdown('<h1 class="main-header">🎯 Employee Turnover Prediction System</h1>', unsafe_allow_html=True)
st.markdown("---")

# Create tabs
tab1, tab2, tab3 = st.tabs(["📝 Prediction", "📊 Model Performance", "ℹ️ About"])

# ══════════════════════════════════════════════════════════════
# TAB 1: PREDICTION FORM
# ══════════════════════════════════════════════════════════════

with tab1:
    st.header("Employee Information Form")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📊 Performance & Evaluation")
        
        epns = st.selectbox(
            "EPNS Rate 2025",
            options=['Promoters', 'Passives', 'Detractors'],
            help="Employee Net Promoter Score - Measures employee satisfaction and likelihood to recommend the company"
        )
        
        perf_2025 = st.selectbox(
            "Performance 2025",
            options=['Outstanding', 'Exceed Expectations', 'Solid Performance',
                    'Marginally below expectations', 'No Performance'],
            help="Employee performance rating for 2025"
        )
        
        perf_2024 = st.selectbox(
            "Performance 2024",
            options=['Outstanding', 'Exceed Expectations', 'Solid Performance',
                    'Marginally below expectations', 'No Performance'],
            help="Employee performance rating for 2024"
        )
        
        perf_2023 = st.selectbox(
            "Performance 2023",
            options=['Outstanding', 'Exceed Expectations', 'Solid Performance',
                    'Marginally below expectations', 'No Performance'],
            help="Employee performance rating for 2023"
        )
        
        promotion = st.selectbox(
            "Promotion in Last 2 Years",
            options=['No', 'Yes'],
            help="Has the employee been promoted in the last 2 years?"
        )
        
        salary = st.selectbox(
            "Salary Placement 2025",
            options=['Marginally Below Placement', 'Within the company Placement', 'High'],
            help="Employee's salary position relative to market standards"
        )
    
    with col2:
        st.subheader("👤 Personal Information")
        
        age = st.selectbox(
            "Age Range",
            options=['20 & 25', '26 & 30', '31 & 40', '40 & 50', '50+'],
            help="Employee's age range"
        )
        
        gender = st.selectbox(
            "Gender",
            options=['H', 'F'],
            format_func=lambda x: 'Male' if x == 'H' else 'Female',
            help="Employee's gender"
        )
        
        manager = st.selectbox(
            "Position Level",
            options=['No Manager', 'Manager'],
            help="Is the employee in a managerial position?"
        )
        
        education = st.selectbox(
            "Education Level",
            options=['BAC', 'Bachelor', 'Engineer', 'Master', 'PhD'],
            help="Highest level of education completed"
        )
        
        seniority = st.slider(
            "Seniority (years)",
            min_value=0.0,
            max_value=40.0,
            value=3.0,
            step=0.5,
            help="Number of years with the company"
        )
        
        direction = st.selectbox(
            "Direction",
            options=direction_names,
            help="Employee's department/direction"
        )
        
        subdept = st.selectbox(
            "Sub Department",
            options=subdept_names,
            help="Employee's sub-department"
        )
    
    with col3:
        st.subheader("⚙️ Work Conditions")
        
        training_hours = st.slider(
            "Training Hours (per year)",
            min_value=0,
            max_value=100,
            value=25,
            step=1,
            help="Total hours of training received in the past year"
        )
        
        stress = st.selectbox(
            "Stress Level",
            options=['Low Level', 'Acceptable Level', 'High Level'],
            help="Employee's reported stress level"
        )
        
        overload = st.selectbox(
            "Work Overload",
            options=['No', 'Yes'],
            help="Is the employee experiencing work overload?"
        )
        
        transport = st.selectbox(
            "Transportation",
            options=['Company Transportation', 'Private Transportation'],
            help="How does the employee commute to work?"
        )
        
        scheme = st.selectbox(
            "Working Scheme",
            options=['FT', 'PT', 'TH'],
            format_func=lambda x: {'FT': 'Full Time', 'PT': 'Part Time', 'TH': 'Temporary Hire'}[x],
            help="Employee's work schedule type"
        )
    
    st.markdown("---")
    
    # Predict button
    if st.button("🎯 PREDICT TURNOVER RISK", type="primary"):
        
        if model is None or config is None:
            st.error("Model not loaded. Please upload model files.")
        else:
            # Get mappings from config
            mappings = config['mappings']
            
            # Encode direction and subdept using label encoders
            if encoders:
                direction_encoded = encoders['le_direction'].transform([direction])[0]
                subdept_encoded = encoders['le_subdept'].transform([subdept])[0]
            else:
                direction_encoded = 0
                subdept_encoded = 0
            
            # Create input dataframe
            input_data = pd.DataFrame({
                'Direction': [direction_encoded],
                'Sub Dept': [subdept_encoded],
                'EPNS Rate 2025': [mappings['epns'][epns]],
                'Performance rate 2025': [mappings['performance'][perf_2025]],
                'Performance rate 2024': [mappings['performance'][perf_2024]],
                'Performance rate 2023': [mappings['performance'][perf_2023]],
                'Promotion Last 2 Years': [mappings['binary'][promotion]],
                'Salary Placement 2025': [mappings['salary'][salary]],
                'Age Range': [mappings['age'][age]],
                'Gender': [mappings['gender'][gender]],
                'M/NM': [mappings['manager'][manager]],
                'Traning Hours': [training_hours],
                'Stress Level': [mappings['stress'][stress]],
                'Seniority': [seniority],
                'Work overload': [mappings['binary'][overload]],
                'Education level': [mappings['education'][education]],
                'Transportation Way': [mappings['transport'][transport]],
                'Working Scheme': [mappings['scheme'][scheme]]
            })
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]
            
            # Display results
            st.markdown("---")
            st.header("📊 Prediction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                prob_leave = probability[1] * 100
                prob_stay = probability[0] * 100
                
                st.metric("Turnover Risk", f"{prob_leave:.1f}%", 
                         delta=f"{prob_leave - 50:.1f}% vs baseline",
                         delta_color="inverse")
                st.progress(prob_leave / 100)
            
            with col2:
                st.metric("Retention Probability", f"{prob_stay:.1f}%",
                         delta=f"{prob_stay - 50:.1f}% vs baseline")
                st.progress(prob_stay / 100)
            
            # Bar chart
            chart_data = pd.DataFrame({
                'Category': ['Will Stay', 'Will Leave'],
                'Probability': [prob_stay, prob_leave]
            })
            st.bar_chart(chart_data.set_index('Category'))
            
            # Risk assessment
            st.markdown("---")
            
            if prediction == 1:
                st.markdown(f"""
                <div class="metric-card risk-high">
                    <h2>⚠️ HIGH RISK - Employee Likely to Leave</h2>
                    <h3>Risk Score: {prob_leave:.1f}%</h3>
                    <p><strong>Recommended Actions:</strong></p>
                    <ul>
                        <li><strong>🎯 Immediate HR intervention required</strong></li>
                        <li><strong>💬 Schedule one-on-one meeting to discuss concerns</strong></li>
                        <li><strong>📊 Review workload and stress levels</strong></li>
                        <li><strong>🎓 Consider additional training opportunities</strong></li>
                        <li><strong>💰 Evaluate compensation and benefits package</strong></li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-card risk-low">
                    <h2>✅ LOW RISK - Employee Likely to Stay</h2>
                    <h3>Retention Score: {prob_stay:.1f}%</h3>
                    <p><strong>Recommended Actions:</strong></p>
                    <ul>
                        <li><strong>✅ Continue regular monitoring and check-ins</strong></li>
                        <li><strong>🎯 Maintain current engagement strategies</strong></li>
                        <li><strong>📈 Focus on career development opportunities</strong></li>
                        <li><strong>🏆 Recognize and reward good performance</strong></li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 2: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════

with tab2:
    st.header("📊 Model Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", "99.40%", "↑ Excellent")
    with col2:
        st.metric("AUC-ROC", "99.96%", "↑ Near Perfect")
    with col3:
        st.metric("Sensitivity", "100%", "↑ All Caught")
    with col4:
        st.metric("Specificity", "99.23%", "↑ Minimal FP")
    
    st.markdown("---")
    
    st.subheader("🔍 Top Features Driving Turnover")
    
    # Feature importance
    features_data = pd.DataFrame({
        'Feature': ['Work overload', 'Training Hours', 'Stress Level', 
                   'Performance 2025', 'Promotion Last 2 Years', 
                   'EPNS Rate 2025', 'Performance 2023'],
        'Importance': [94.19, 56.55, 54.24, 49.05, 46.12, 43.22, 40.70]
    })
    features_data = features_data.sort_values('Importance', ascending=True)
    st.bar_chart(features_data.set_index('Feature'))
    
    st.info("""
    **Key Insights:**
    - **Work Overload** is the strongest predictor (94% importance)
    - **Training Hours** significantly impact retention (57%)
    - **Stress Level** is critical (54%)
    - Focus HR interventions on these areas for maximum impact
    """)

# ══════════════════════════════════════════════════════════════
# TAB 3: ABOUT
# ══════════════════════════════════════════════════════════════

with tab3:
    st.header("ℹ️ About This System")
    
    st.markdown(f"""
    ### 🎯 Purpose
    Employee Turnover Prediction System using **Machine Learning** to identify 
    at-risk employees for proactive HR intervention.
    
    ### 🤖 Model Details
    - **Algorithm:** Random Forest Classifier
    - **Training Data:** 1,661 employee records
    - **Features:** 18 employee characteristics
    - **Performance:** 99.96% AUC-ROC
    
    ### 📊 Key Metrics
    - **Accuracy:** 99.40%
    - **Sensitivity:** 100% (catches all departures)
    - **Specificity:** 99.23% (only 2 false alarms)
    
    ### 🔍 Top Drivers
    1. Work Overload (94%)
    2. Training Hours (57%)
    3. Stress Level (54%)
    
    ### 📈 Version
    **v1.0** - {datetime.now().strftime("%B %Y")}
    
    ---
    *Built with Python, Scikit-learn & Streamlit*
    """)

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("📊 Turnover Predictor")
    st.markdown("---")
    
    st.subheader("📊 Quick Stats")
    st.metric("Model Accuracy", "99.40%")
    st.metric("Employees Analyzed", "1,661")
    st.metric("Turnover Rate", "22.34%")
    
    st.markdown("---")
    st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d')}")
    st.caption("Powered by Random Forest ML")