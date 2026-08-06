# =====================================
# IMPORTS
# =====================================

import io
import zipfile

from PIL import Image
import streamlit as st

try:
    from src.predict import predict_image
except Exception as e:
    st.exception(e)
    st.stop()
    # =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="HydroVision",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)
# =====================================
# SESSION STATE
# =====================================

if "result" not in st.session_state:
    st.session_state.result = None
    # =====================================
# COLORS
# =====================================

COLORS = {

    "bg": "#0B0F19",
    "sidebar": "#151B26",
    "card": "#1B2230",
    "text": "#F8FAFC",
    "secondary": "#94A3B8",
    "accent": "#4F9DFF"

}# =====================================
# CSS
# =====================================

st.markdown(
f"""
<style>

/* App */

.stApp{{
    background:{COLORS["bg"]};
    color:{COLORS["text"]};
}}

.block-container{{
    max-width:1100px;
    padding-top:1rem;
}}


/* Sidebar */

section[data-testid="stSidebar"]{{
    background:{COLORS["sidebar"]};
    border-right:1px solid #2A3345;
}}


/* Text */

h1,h2,h3,h4,h5,h6,p,label,span,div{{
    color:{COLORS["text"]};
}}


/* Upload Box */

div[data-testid="stFileUploader"]{{
    background:{COLORS["card"]};
    border:1px solid #2D3648;
    border-radius:15px;
    padding:18px;
}}


/* Upload Button */

.stFileUploader button{{
    background:#232C3B;
    color:white;
    border-radius:25px;
    border:none;
    font-weight:600;
}}

.stFileUploader button:hover{{
    background:#2E3B50;
}}


/* Buttons */

.stButton > button,
.stDownloadButton > button{{
    background:{COLORS["accent"]};
    color:white;
    border:none;
    border-radius:10px;
}}

.stButton > button:hover,
.stDownloadButton > button:hover{{
    opacity:0.9;
}}


/* Expanders */

.streamlit-expanderHeader{{
    font-weight:600;
}}

hr{{
    border:1px solid #263041;
}}

</style>
""",
unsafe_allow_html=True
)
# =====================================
# SIDEBAR
# =====================================

with st.sidebar:

    st.title("🌊 HydroVision")

    st.caption("Satellite-Based Flood Detection & Risk Assessment")

    st.divider()

    with st.expander("ⓘ About FloodGuard AI"):

        st.write("""
FloodGuard AI detects flooded regions from satellite imagery using an AI segmentation model.

The system provides:

• Flood Overlay

• Flood Mask

• Flood Coverage

• Risk Level

• AI Confidence
""")

    with st.expander("🤖 AI System"):

        st.write("""
Model : U-Net

Framework : TensorFlow / Keras

Input : 256 × 256 Satellite Image

Task : Binary Segmentation
""")

    with st.expander("🛰 How It Works"):

        st.write("""
1. Upload a satellite image

2. AI preprocesses the image

3. Flood regions are detected

4. Flood analysis is generated
""")

    with st.expander("📊 Model Information"):

        st.write("""
• Deep Learning

• U-Net Architecture

• Satellite Image Analysis

• Binary Segmentation
""")

    with st.expander("🔄 New Analysis"):

        if st.button(
            "Start New Analysis",
            use_container_width=True
        ):
            st.session_state.result = None
            st.rerun()
# =====================================
# HEADER
# =====================================

st.markdown(
"""
# 🌊 HydroVision

Satellite-Based Flood Detection & Risk Assessment
"""
)
# =====================================
# IMAGE UPLOAD
# =====================================

uploaded_file = st.file_uploader(
    "📤 Upload Satellite Image",
    type=[
        "png",
        "jpg",
        "jpeg",
        "tif",
        "tiff"
    ]
)
# =====================================
# PREDICTION
# =====================================

if uploaded_file is not None:

    with st.spinner("🌊 Analyzing Satellite Image..."):

        st.session_state.result = predict_image(uploaded_file)
# =====================================
# DETECTION RESULTS
# =====================================

if st.session_state.result is not None:

    result = st.session_state.result

    st.markdown("## 🛰 Detection Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(result["original"], caption="Original")

    with col2:
        st.image(result["overlay"], caption="Flood Overlay")

    with col3:
        st.image(result["mask"], caption="Flood Mask")
# =====================================
# FLOOD ANALYSIS
# =====================================

if st.session_state.result is not None:

    result = st.session_state.result

    st.markdown("## 🌊 Flood Analysis")

    # Create the 3 columns
    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("""
        <p style="font-size:18px;font-weight:600;">
        🌊 Flood Coverage
        </p>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <p style="font-size:30px;font-weight:bold;color:#4F9DFF;">
        {result["flood_percentage"]}
        </p>
        """, unsafe_allow_html=True)

    
    with col2:

        st.markdown("""
        <p style="font-size:18px;font-weight:600;">
        ⚠️ Risk Level
        </p>
        """, unsafe_allow_html=True)

        risk = result["risk_level"]

        if risk.lower() == "low":
           st.success(risk)

        elif risk.lower() == "moderate":
            st.warning(risk)

        else:
            st.error(risk)
    with col3:

        st.markdown("""
        <p style="font-size:18px;font-weight:600;">
        🧠 AI Confidence
        </p>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <p style="font-size:25px;font-weight:bold;color:#38BDF8;">
        {result["confidence"]}
        </p>
        """, unsafe_allow_html=True)
# =====================================
# RECOMMENDATIONS
# =====================================

if st.session_state.result is not None:

    result = st.session_state.result

    st.markdown("---")
    st.markdown("## 💡 Recommendations")

    st.write(result["recommendation"])
# =====================================
# DOWNLOAD REPORT
# =====================================

if st.session_state.result is not None:

    result = st.session_state.result

    report = f"""
FloodGuard AI Report
===============================

Flood Coverage : {result["flood_percentage"]}

Risk Level     : {result["risk_level"]}

AI Confidence  : {result["confidence"]}

----------------------------------------

Recommendations

{result["recommendation"]}

----------------------------------------

Generated by FloodGuard AI
"""

    st.download_button(

        label="📥 Download ",

        data=report,

        file_name="HydroVision_Report.txt",

        mime="text/plain",

        use_container_width=True

    )
# =====================================
# FOOTER
# =====================================

st.markdown("---")

st.markdown(
"""
<div style="text-align:center; padding:10px 0;">

<b>🌊 HydroVision</b>


<small>
Built using TensorFlow, U-Net & Streamlit
</small>

</div>
""",
unsafe_allow_html=True
)