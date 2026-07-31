import streamlit as st
import numpy as np
from PIL import Image

from predict import predict_image

# -----------------------
# Page Configuration
# -----------------------
st.set_page_config(
    page_title="FloodGuard AI",
    page_icon="🌊",
    layout="wide"
)

# -----------------------
# CSS
# -----------------------
st.markdown("""
<style>

.stApp{
    background-color:#EEF6FF;
}

.title{
    text-align:center;
    color:#1565C0;
    font-size:48px;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    color:#555555;
    font-size:18px;
}

.block-container{
    padding-top:2rem;
}

.footer{
    text-align:center;
    color:gray;
    font-size:15px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------
# Sidebar
# -----------------------

with st.sidebar:

    try:
        st.image("assets/logo.png", width=180)
    except:
        pass

    st.title("🌊 FloodGuard AI")

    st.markdown("""
### About

FloodGuard AI is an AI-powered application that detects flooded regions from satellite images using a Deep Learning U-Net model.

### Features

✅ Flood Detection

✅ Flood Percentage

✅ Flood Risk Analysis

✅ Safety Recommendations

---

Developed for an AIML Mini Project
""")

# -----------------------
# Header
# -----------------------

try:
    st.image("assets/background.jpg", use_container_width=True)
except:
    pass

st.markdown(
    "<div class='title'>🌊 FloodGuard AI</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>AI Powered Flood Detection from Satellite Images</div>",
    unsafe_allow_html=True
)

st.info("""
👋 **Welcome!**

Upload a satellite image and FloodGuard AI will:

- Detect flooded regions
- Generate a flood segmentation mask
- Show flood overlay
- Calculate flood coverage
- Predict flood risk
""")

# -----------------------
# Upload
# -----------------------

uploaded_file = st.file_uploader(
    "📤 Upload Satellite Image",
    type=["png", "jpg", "jpeg"]
)

# -----------------------
# Prediction
# -----------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image = np.array(image)

    original, mask, overlay, flood = predict_image(image)

    st.success("✅ Prediction Completed Successfully!")

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

    st.metric(
        label="Flood Coverage",
        value=f"{flood:.2f}%"
    )

    if flood < 10:
        st.success("🟢 Flood Risk : LOW")

    elif flood < 30:
        st.warning("🟠 Flood Risk : MEDIUM")

    else:
        st.error("🔴 Flood Risk : HIGH")

    st.subheader("💡 Recommendations")

    if flood < 10:

        st.info("""
✔ No major flood detected.

✔ Continue monitoring.

✔ Area appears safe.
""")

    elif flood < 30:

        st.warning("""
✔ Water accumulation detected.

✔ Stay alert.

✔ Avoid low-lying roads.

✔ Monitor weather updates.
""")

    else:

        st.error("""
✔ High flood possibility detected.

✔ Avoid travelling.

✔ Contact local authorities.

✔ Move to safer locations immediately if required.
""")

else:

    st.info("👆 Upload a satellite image to begin prediction.")

# -----------------------
# Footer
# -----------------------

st.markdown("---")

st.markdown(
"""
<div class="footer">

🌊 <b>FloodGuard AI</b><br>

Artificial Intelligence & Machine Learning Mini Project<br><br>

Developed by <b>Akshaya Karasani</b>

</div>
""",
unsafe_allow_html=True
)