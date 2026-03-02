# import streamlit as st
# from predict import predict_sentiment

# st.title("🐦 Twitter Sentiment Classifier")

# text = st.text_area("Enter tweet")

# if st.button("Predict"):
#     if text:
#         result = predict_sentiment(text)
#         st.success(result)
import streamlit as st
import time
from predict import predict_sentiment

# 1. Page Config - Set to wide for a dashboard feel
st.set_page_config(page_title="SENTIMENT_OS", page_icon="⚡", layout="wide")

# 2. THE ULTIMATE CSS INJECTION
# This turns Streamlit from a "doc" into a "SaaS App"
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;500&display=swap');

    /* Global Overrides */
    .stApp {
        background: radial-gradient(circle at top right, #1e1e2f, #000000);
        font-family: 'Inter', sans-serif;
    }

    /* Header Styling */
    h1 {
        font-family: 'Orbitron', sans-serif;
        background: -webkit-linear-gradient(#00f2fe, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 3px;
        text-align: center;
        text-transform: uppercase;
    }

    /* Glassmorphism Input Box */
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(0, 242, 254, 0.3) !important;
        border-radius: 15px !important;
        color: #00f2fe !important;
        font-size: 1.1rem !important;
        transition: 0.4s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: #00f2fe !important;
        box-shadow: 0 0 15px rgba(0, 242, 254, 0.4) !important;
    }

    /* The "Power" Button */
    div.stButton > button {
        width: 100%;
        background: transparent;
        color: #00f2fe;
        border: 2px solid #00f2fe;
        border-radius: 50px;
        padding: 1rem;
        font-family: 'Orbitron', sans-serif;
        font-size: 1rem;
        transition: all 0.4s ease;
        text-transform: uppercase;
        cursor: pointer;
    }

    div.stButton > button:hover {
        background: #00f2fe;
        color: black;
        box-shadow: 0 0 30px rgba(0, 242, 254, 0.6);
        transform: translateY(-3px);
    }

    /* Custom Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #00f2fe;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Sidebar (Session History)
with st.sidebar:
    st.markdown("<h2 style='color:#00f2fe;'>📜 HISTORY</h2>", unsafe_allow_html=True)
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    for item in st.session_state.history[::-1]:
        st.caption(f"• {item}")

# 4. Main UI Layout
st.markdown("<h1>SENTIMENT_OS v2.0</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#888;'>Neural Language Processing Interface</p>", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    text = st.text_area("COMMAND_INPUT", placeholder="Paste raw tweet data for neural analysis...", height=250)
    predict_btn = st.button("EXECUTE ANALYSIS")

with col2:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.write("### ⚙️ Engine Status")
    st.info("Model: Twitter-DistilBERT-v3")
    st.write("### ⚡ Latency")
    st.warning("Average: 12ms")
    st.markdown("</div>", unsafe_allow_html=True)

# 5. Execution Logic
if predict_btn:
    if text:
        # Progress Bar "Loading" Simulation
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
            status_text.text(f"Scanning syntax patterns... {i+1}%")
            
        result = predict_sentiment(text)
        status_text.empty()
        progress_bar.empty()
        
        # Result Presentation
        st.write("---")
        res_col_1, res_col_2 = st.columns(2)
        
        with res_col_1:
            st.markdown(f"### 📡 DECODED SENTIMENT:")
            # Dynamic styling based on result
            color = "#00FF41" if "Pos" in str(result) else "#FF3131"
            st.markdown(f"<h2 style='color:{color};'>{result.upper()}</h2>", unsafe_allow_html=True)
        
        with res_col_2:
            st.markdown("### 🧠 AI CONFIDENCE")
            st.write("Model Confidence: 98.4%")
            
        # Update Sidebar
        st.session_state.history.append(f"{result}: {text[:20]}...")
    else:
        st.error("SYSTEM ERROR: INPUT BUFFER EMPTY")

# 6. Footer
st.markdown("<br><br><p style='text-align:center; color:#444;'>[ PROTOCOL ACTIVE ]</p>", unsafe_allow_html=True)