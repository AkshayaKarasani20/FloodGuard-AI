import streamlit as st
import numpy as np
from PIL import Image

from predict import predict_image

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="FloodGuard AI",
    page_icon="🌊",
    layout="wide"
)

# -----------------------------------
# Custom CSS
# -----------------------------------
st.markdown("""
<style>

.stApp{
    background: linear-gradient(to bottom,#F4FAFF,#EAF4FF);
}

.block-container{
    padding-top:1.5rem;
}

.title{
    text-align:center;
    font-size:52px;
    font-weight:700;
    color:#1565C0;
}

.subtitle{
    text-align:center;
    color:#555;
    font-size:20px;
    margin-bottom:20px;
}

.footer{
    text-align:center;
    color:#666;
    font-size:15px;
}

div[data-testid="stMetric"]{
    background:white;
    padding:20px;
    border-radius:15px;
    border-left:8px solid #1565C0;
    box-shadow:0px 4px 10px rgba(0,0,0,0.12);
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------
# Sidebar
# -----------------------------------

with st.sidebar:

    try:
        st.image("assets/logo.png", width=180)
    except:
        pass

    st.title("🌊 FloodGuard AI")

    st.markdown("""
### About

FloodGuard AI is a Deep Learning application that detects flooded regions from satellite images using a U-Net segmentation model.

---

### Features

✅ Flood Detection

✅ Flood Percentage

✅ Flood Risk Analysis

✅ Flood Overlay

✅ Safety Recommendations

---

### Technologies

🧠 TensorFlow

🛰 U-Net

📷 OpenCV

📊 Streamlit

🌍 Satellite Images

---

Made with ❤️ by

**Akshaya Karasani**
""")

# -----------------------------------
# Header Image
# -----------------------------------

try:
    st.image("assets/background.jpg", use_container_width=True)
except:
    pass

# -----------------------------------
# Title
# -----------------------------------

st.markdown(
    "<div class='title'>🌊 FloodGuard AI</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>AI Powered Flood Detection from Satellite Images</div>",
    unsafe_allow_html=True
)

st.write("")

st.info("""
👋 **Welcome to FloodGuard AI**

Upload a satellite image and the system will:

✔ Detect flooded regions

✔ Generate flood segmentation mask

✔ Display flood overlay

✔ Calculate flood percentage

✔ Predict flood risk level

✔ Provide safety recommendations
""")

st.write("")

# -----------------------------------
# Upload
# -----------------------------------

uploaded_file = st.file_uploader(
    "📤 Upload Satellite Image",
    type=["png","jpg","jpeg"]
)

# -----------------------------------
# Prediction
# -----------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image = np.array(image)

    original, mask, overlay, flood = predict_image(image)

    st.success("✅ Prediction Completed Successfully!")

    st.progress(min(int(flood),100))

    st.write("")

    col1,col2,col3 = st.columns(3)

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

    c1,c2,c3,c4 = st.columns(4)

    with c1:
        st.metric("Flood Coverage",f"{flood:.2f}%")

    with c2:
        st.metric("Image Width",f"{original.shape[1]} px")

    with c3:
        st.metric("Model","U-Net")

    with c4:
        st.metric("Prediction","Completed")

    st.write("")

    st.subheader("📈 Flood Severity")

    st.progress(min(int(flood),100))

    if flood < 10:

        st.success("🟢 LOW RISK")

    elif flood < 30:

        st.warning("🟠 MEDIUM RISK")

    else:

        st.error("🔴 HIGH RISK")

    st.write("")

    st.subheader("💡 Safety Recommendations")

    if flood < 10:

        st.info("""
✅ No major flood detected.

✅ Continue monitoring.

✅ Area appears safe.

✅ Normal activities can continue.
""")

    elif flood < 30:

        st.warning("""
⚠ Moderate flooding detected.

⚠ Stay alert.

⚠ Avoid low-lying roads.

⚠ Monitor weather updates.
""")

    else:

        st.error("""
🚨 High flood possibility detected.

🚨 Avoid travelling.

🚨 Contact local authorities.

🚨 Move to safer locations immediately.

🚨 Follow emergency instructions.
""")

else:

    st.info("👆 Upload a satellite image to begin prediction.")

# -----------------------------------
# Footer
# -----------------------------------

st.markdown("---")

st.markdown("""
<div class='footer'>

<h3 style='color:#1565C0;'>🌊 FloodGuard AI</h3>

AI Powered Flood Detection using Satellite Images

<br>

<b>Developed by Akshaya Karasani</b>

<br><br>

Artificial Intelligence & Machine Learning Mini Project

</div>
""", unsafe_allow_html=True)