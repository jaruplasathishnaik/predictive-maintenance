import streamlit as st
import joblib
import numpy as np

# Page Config
st.set_page_config(
    page_title="NexGen Predictive Maintenance",
    page_icon="🤖",
    layout="wide"
)

# --- ADVANCED UI/UX CUSTOMIZATION ---
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }

    /* Glassmorphism Card */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }

    /* Gradient Title */
    .main-title {
        background: -webkit-linear-gradient(45deg, #00f2fe, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0;
    }

    /* Metric Styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #38bdf8 !important;
    }

    /* Custom Button */
    .stButton>button {
        width: 100%;
        background: linear-gradient(45deg, #3b82f6, #2563eb);
        color: white;
        border: none;
        padding: 0.75rem;
        border-radius: 12px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(59, 130, 246, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    model = joblib.load("rf_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error("Model files not found. Please ensure 'rf_model.pkl' and 'scaler.pkl' are in the directory.")
    st.stop()

# --- SIDEBAR INPUTS ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2092/2092130.png", width=100)
    st.header("⚙️ Configuration")
    
    machine_type = st.selectbox("Machine Grade", ["L (Low)", "M (Medium)", "H (High)"])
    air_temp = st.slider("Air Temperature [K]", 280, 320, 300)
    process_temp = st.slider("Process Temperature [K]", 290, 340, 310)
    rpm = st.slider("Rotational Speed [rpm]", 1000, 3000, 1500)
    torque = st.slider("Torque [Nm]", 10, 80, 40)
    tool_wear = st.slider("Tool Wear [min]", 0, 300, 100)
    
    predict_btn = st.button("Run")

# --- MAIN CONTENT ---
st.markdown('<h1 class="main-title">NEXGEN PREDICTIVE AI</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; opacity: 0.7;'>Real-time Industry 4.0 Failure Analysis</p>", unsafe_allow_html=True)

# Metric Display Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Thermal Delta", f"{process_temp - air_temp} K")
with col2:
    st.metric("Stress Level", f"{torque * rpm / 9550:.1f} kW") # Calculated Power
with col3:
    st.metric("Efficiency", f"{100 - (tool_wear/3):.1f}%")
with col4:
    st.metric("Grade", machine_type[0])

st.markdown('---')

# --- PREDICTION LOGIC ---
if predict_btn:
    # Preprocessing
    type_mapping = {"L (Low)": 0, "M (Medium)": 1, "H (High)": 2}
    type_encoded = type_mapping[machine_type]
    temp_diff = process_temp - air_temp

    input_data = np.array([[type_encoded, air_temp, process_temp, rpm, torque, tool_wear, temp_diff]])
    input_scaled = scaler.transform(input_data)
    
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # Result Display
    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        if prediction == 1:
            st.markdown(f"""
                <h2 style='color: #ef4444;'>⚠ CRITICAL FAILURE</h2>
                <p>The AI has detected patterns matching a high probability of mechanical breakdown.</p>
                <h1 style='color: #ef4444; font-size: 4rem;'>{probability:.1%}</h1>
                <p>Confidence Level</p>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <h2 style='color: #10b981;'>✅ SYSTEM NOMINAL</h2>
                <p>Machine parameters are within safe operational limits. No immediate action required.</p>
                <h1 style='color: #10b981; font-size: 4rem;'>{(1-probability):.1%}</h1>
                <p>Health Score</p>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with res_col2:
        if prediction == 1:
            st.image("https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExdzFpb25ndnZhemI3MHJqdXYwZmowenQ0amQ5aGJyNmljeDQ2Nmw3MiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/gLoMzjGQB2tQlQtB9P/giphy.gif")
        else:
            st.image("https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExM3Y4cGt2bzEwaHQ3Ym5vd3E4bGd5YWJnZHRvaG5heTJlb2VqaGR0dCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/dqP5n0cM4dNUJlWZql/giphy.gif")

    # Probability Bar
    st.write("Failure Risk Probability")
    st.progress(float(probability))
else:
    st.info("👈 Adjust the machine parameters in the sidebar and click 'Run Diagnostics' to begin.")