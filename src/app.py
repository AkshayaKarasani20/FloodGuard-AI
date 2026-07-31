import streamlit as st
import numpy as np
from PIL import Image

from predict import predict_image

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="FloodGuard AI",
    page_icon="🌊",
    layout="wide"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

.stApp{
    background-color:#EEF6FF;
}

.block-container{
    padding-top:2rem;
}

.title{
    text-align:center;
    color:#1565C0;
    font-size:52px;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    color:#555;
    font-size:20px;
    margin-bottom:25px;
}

.footer{
    text-align:center;
    color:gray;
    font-size:14px;
    padding-top:30px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    try:
        st.image("assets/logo.png", width=180)
    except:
        pass

    st.title("🌊 FloodGuard AI")

    st.markdown("""
AI-powered flood detection from satellite images using a **U-Net Deep Learning model**.
""")

    st.divider()

    st.subheader("✨ Features")

    st.markdown("""
- 🛰 Flood Detection
- 🌊 Flood Segmentation
- 📊 Flood Percentage
- 🚨 Risk Analysis
- 💡 Safety Recommendations
""")

    st.divider()

    st.subheader("🧠 Model")

    st.markdown("""
**Architecture:** U-Net

**Framework:** TensorFlow

**Dataset:** Sen1Floods11

**Task:** Binary Segmentation
""")

# ==========================================================
# HEADER
# ==========================================================

st.markdown(
    "<div class='title'>🌊 FloodGuard AI</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>AI Powered Flood Detection from Satellite Images</div>",
    unsafe_allow_html=True
)

# ==========================================================
# ABOUT
# ==========================================================

with st.expander("ℹ About this Project"):

    st.write("""
FloodGuard AI detects flooded regions from satellite images using a Deep Learning U-Net model.

The application predicts flood masks, estimates flood coverage, and provides a flood risk assessment.
""")

# ==========================================================
# FILE UPLOAD
# ==========================================================

uploaded_file = st.file_uploader(
    "📤 Upload a Satellite Image",
    type=["png", "jpg", "jpeg"]
)

# ==========================================================
# PREDICTION
# ==========================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image = np.array(image)

    with st.spinner("🔍 Analyzing Satellite Image..."):

        original, mask, overlay, flood = predict_image(image)

    st.success("✅ Prediction Completed Successfully!")

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.image(
            original,
            caption="🛰 Original Image",
            use_container_width=True
        )

    with col2:

        st.image(
            mask,
            caption="🌊 Predicted Flood Mask",
            use_container_width=True,
            clamp=True
        )

    with col3:

        st.image(
            overlay,
            caption="🗺 Flood Overlay",
            use_container_width=True
        )

    st.divider()

    st.subheader("📊 Flood Analysis")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Flood Coverage", f"{flood:.2f}%")

    with c2:
        st.metric("Model", "U-Net")

    with c3:

        if flood < 10:
            st.metric("Risk", "LOW")

        elif flood < 30:
            st.metric("Risk", "MEDIUM")

        else:
            st.metric("Risk", "HIGH")

    st.write("### Flood Coverage")

    st.progress(min(int(flood),100))

    st.divider()

    st.subheader("🚨 Flood Risk")

    if flood < 10:

        st.success("🟢 LOW RISK")

    elif flood < 30:

        st.warning("🟠 MEDIUM RISK")

    else:

        st.error("🔴 HIGH RISK")

    st.subheader("💡 Recommendations")

    if flood < 10:

        st.info("""
✅ No major flood detected.

✅ Continue monitoring.

✅ Area appears safe.
""")

    elif flood < 30:

        st.warning("""
✅ Water accumulation detected.

✅ Stay alert.

✅ Avoid low-lying roads.

✅ Monitor weather updates.
""")

    else:

        st.error("""
✅ High flood possibility detected.

✅ Avoid travelling.

✅ Contact local authorities.

✅ Move to safer locations if necessary.
""")

else:

    st.info("👆 Upload a satellite image to begin prediction.")

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown(
"""
<div class='footer'>

🌊 <b>FloodGuard AI</b><br><br>

Artificial Intelligence & Machine Learning Mini Project<br><br>

Developed by <b>Akshaya Karasani</b>

</div>
""",
unsafe_allow_html=True
)