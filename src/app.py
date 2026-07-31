import streamlit as st
import numpy as np
from PIL import Image

from predict import predict_image


# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="FloodGuard AI",
    page_icon="🌊",
    layout="wide"
)


# ==========================
# CUSTOM CSS
# ==========================

st.markdown("""
<style>

.stApp {
    background: #F4F9FF;
}


/* Main title */

.hero-title {
    text-align:center;
    font-size:50px;
    font-weight:800;
    color:#0D47A1;
}


.hero-subtitle {
    text-align:center;
    font-size:20px;
    color:#455A64;
}


/* Cards */

.card {

    background:white;
    padding:25px;
    border-radius:18px;
    box-shadow:0px 4px 15px rgba(0,0,0,0.08);
    margin-bottom:20px;

}


.card-title {

    color:#1565C0;
    font-size:25px;
    font-weight:bold;

}


/* Upload box */

[data-testid="stFileUploader"] {

    background:white;
    padding:20px;
    border-radius:15px;

}


/* Footer */

.footer {

text-align:center;
color:#607D8B;
font-size:15px;

}


</style>

""", unsafe_allow_html=True)



# ==========================
# SIDEBAR
# ==========================

with st.sidebar:


    try:
        st.image(
            "assets/logo.png",
            width=170
        )

    except:
        pass


    st.title("🌊 FloodGuard AI")


    st.markdown("""

### AI Flood Monitoring System

Detect flooded regions from satellite images using deep learning.

---

### Core Features

✅ Flood Segmentation

✅ Risk Analysis

✅ Flood Coverage

✅ Safety Guidance


---

### Technology

🧠 U-Net CNN Model

🛰 Satellite Images

🐍 Python

⚡ Streamlit


""")



# ==========================
# HERO SECTION
# ==========================


st.markdown(
"""
<div class="hero-title">
🌊 FloodGuard AI
</div>

<div class="hero-subtitle">
AI Powered Flood Detection From Satellite Images
</div>

""",
unsafe_allow_html=True
)


st.write("")



# ==========================
# ABOUT SECTION
# ==========================


col1,col2 = st.columns(2)


with col1:

    st.markdown(
    """
    <div class="card">

    <div class="card-title">
    📖 About FloodGuard AI
    </div>

    FloodGuard AI is a deep learning based flood detection system
    that identifies flooded regions from satellite imagery.

    The system uses a U-Net segmentation model to generate flood
    masks and estimate affected area.

    </div>

    """,
    unsafe_allow_html=True
    )



with col2:

    st.markdown(
    """
    <div class="card">

    <div class="card-title">
    🛰 Why Satellite Images?
    </div>


    Satellite images help in:

    <br>

    🌍 Monitoring large areas

    <br>
    ⚡ Faster disaster assessment

    <br>
    📍 Identifying affected regions

    <br>
    🚨 Supporting emergency response

    </div>

    """,
    unsafe_allow_html=True
    )



# ==========================
# WORKFLOW PREVIEW
# ==========================


st.markdown(
"""
<div class="card">

<div class="card-title">
⚙️ How It Works
</div>


1️⃣ Upload satellite image

<br>

2️⃣ Image preprocessing

<br>

3️⃣ AI model predicts flood regions

<br>

4️⃣ Flood coverage and risk analysis

</div>

""",
unsafe_allow_html=True
)



# ==========================
# UPLOAD SECTION
# ==========================


st.subheader("📤 Upload Satellite Image")


uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png","jpg","jpeg"]
)



# Prediction code will be added in Part 2


if uploaded_file is None:

    st.info(
        "👆 Upload a satellite image to start flood detection"
    )



# ==========================
# FOOTER
# ==========================


st.markdown("---")


st.markdown(
"""
<div class="footer">

🌊 FloodGuard AI<br>
Artificial Intelligence & Machine Learning Mini Project<br>
Developed by <b>Akshaya Karasani</b>

</div>

""",
unsafe_allow_html=True
)