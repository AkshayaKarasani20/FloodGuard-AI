import streamlit as st
import numpy as np
import cv2
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

.main{
    background-color:#f4f8fb;
}

.title{
    text-align:center;
    color:#1565C0;
    font-size:45px;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    color:gray;
    font-size:18px;
}

.resultbox{
    padding:15px;
    border-radius:12px;
    background:#E3F2FD;
}

</style>
""", unsafe_allow_html=True)

# -----------------------
# Header
# -----------------------

st.markdown(
    "<div class='title'>🌊 FloodGuard AI</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>AI Powered Flood Detection from Satellite Images</div>",
    unsafe_allow_html=True
)

st.write("")
st.write("")

# -----------------------
# Upload
# -----------------------

uploaded_file = st.file_uploader(
    "📤 Upload Satellite Image",
    type=["png","jpg","jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    image = np.array(image)

    original, mask, overlay, flood = predict_image(image)

    st.success("Prediction Completed Successfully!")

    col1,col2,col3 = st.columns(3)

    with col1:
        st.image(
            original,
            caption="Original Image",
            use_container_width=True
        )

    with col2:
        st.image(
            mask,
            caption="Predicted Flood Mask",
            use_container_width=True,
            clamp=True
        )

    with col3:
        st.image(
            overlay,
            caption="Flood Overlay",
            use_container_width=True
        )

    st.write("")
    st.write("---")

    st.subheader("📊 Flood Analysis")

    st.metric(
        "Flood Coverage",
        f"{flood:.2f}%"
    )

    if flood < 10:
        st.success("🟢 Flood Risk : LOW")

    elif flood < 30:
        st.warning("🟠 Flood Risk : MEDIUM")

    else:
        st.error("🔴 Flood Risk : HIGH")

    st.write("")

    st.subheader("💡 Recommendations")

    if flood < 10:

        st.info("""
✔ No major flood detected.

✔ Continue monitoring.

✔ Safe for now.
""")

    elif flood < 30:

        st.warning("""
✔ Water accumulation detected.

✔ Stay alert.

✔ Avoid low-lying roads.
""")

    else:

        st.error("""
✔ High flood possibility.

✔ Avoid travelling.

✔ Contact local authorities.

✔ Move to safer locations if necessary.
""")

else:

    st.info("👆 Upload a satellite image to begin prediction.")