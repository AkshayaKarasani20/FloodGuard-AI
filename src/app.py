import streamlit as st
import numpy as np
from PIL import Image
from predict import predict_image

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------

st.set_page_config(
    page_title="FloodGuard AI",
    page_icon="🌊",
    layout="wide"
)

# -------------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------------

st.markdown("""
<style>

.stApp{
    background:#F6F8FB;
}

.block-container{
    padding-top:1.5rem;
    padding-bottom:2rem;
}

.hero{
    background:linear-gradient(135deg,#1565C0,#42A5F5);
    padding:35px;
    border-radius:18px;
    text-align:center;
    color:white;
    margin-bottom:25px;
}

.hero h1{
    font-size:50px;
    margin-bottom:5px;
}

.hero p{
    font-size:20px;
    color:white;
}

.feature{
    background:white;
    padding:15px;
    border-radius:15px;
    text-align:center;
    box-shadow:0px 2px 10px rgba(0,0,0,0.08);
}

.footer{
    text-align:center;
    color:gray;
    padding-top:25px;
    font-size:15px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------

with st.sidebar:

    try:
        st.image("assets/logo.png", width=180)
    except:
        pass

    st.title("🌊 FloodGuard AI")

    st.write("### AI Powered Flood Detection")

    st.info("""
Upload satellite images and detect flooded regions using Deep Learning.

### Features

✅ Flood Detection

✅ Flood Mask

✅ Flood Overlay

✅ Flood Percentage

✅ Risk Analysis

✅ Safety Tips
""")

    st.divider()

    st.success("Developed using U-Net + TensorFlow")

# -------------------------------------------------------
# HERO SECTION
# -------------------------------------------------------

st.markdown("""
<div class="hero">

<h1>🌊 FloodGuard AI</h1>

<p>AI Powered Flood Detection using Satellite Images</p>

</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# FEATURES
# -------------------------------------------------------

c1,c2,c3,c4=st.columns(4)

with c1:
    st.markdown("""
<div class="feature">
<h3>🛰</h3>
<b>Satellite Images</b>
</div>
""",unsafe_allow_html=True)

with c2:
    st.markdown("""
<div class="feature">
<h3>🤖</h3>
<b>Deep Learning</b>
</div>
""",unsafe_allow_html=True)

with c3:
    st.markdown("""
<div class="feature">
<h3>🌊</h3>
<b>Flood Segmentation</b>
</div>
""",unsafe_allow_html=True)

with c4:
    st.markdown("""
<div class="feature">
<h3>📊</h3>
<b>Risk Analysis</b>
</div>
""",unsafe_allow_html=True)

st.write("")
st.write("")

# -------------------------------------------------------
# UPLOAD
# -------------------------------------------------------

uploaded_file = st.file_uploader(
    "📤 Upload Satellite Image",
    type=["png","jpg","jpeg"]
)

# -------------------------------------------------------
# PREDICTION
# -------------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image = np.array(image)

    with st.spinner("🔍 AI is analyzing the satellite image..."):

        original,mask,overlay,flood = predict_image(image)

    st.success("✅ Prediction Completed Successfully")

    st.write("")

    col1,col2,col3=st.columns(3)

    with col1:
        st.image(
            original,
            caption="🛰 Original Image",
            use_container_width=True
        )

    with col2:
        st.image(
            mask,
            caption="🌊 Flood Mask",
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

    a,b,c=st.columns(3)

    with a:
        st.metric(
            "Flood Coverage",
            f"{flood:.2f}%"
        )

    with b:

        if flood<10:
            st.metric("Risk Level","🟢 LOW")

        elif flood<30:
            st.metric("Risk Level","🟠 MEDIUM")

        else:
            st.metric("Risk Level","🔴 HIGH")

    with c:

        if flood<10:
            st.metric("Status","Safe")

        elif flood<30:
            st.metric("Status","Monitor")

        else:
            st.metric("Status","Danger")

    st.divider()

    st.subheader("💡 Safety Recommendations")

    if flood<10:

        st.success("""

✅ No significant flooding detected.

✅ Continue monitoring weather updates.

✅ Area appears safe.

""")

    elif flood<30:

        st.warning("""

⚠ Water accumulation detected.

⚠ Stay alert.

⚠ Avoid low-lying roads.

⚠ Monitor weather conditions.

""")

    else:

        st.error("""

🚨 High flood probability detected.

🚨 Avoid travelling.

🚨 Contact emergency services.

🚨 Move to safer locations immediately.

""")

else:

    st.info("👆 Upload a satellite image to start flood detection.")

# -------------------------------------------------------
# ABOUT PROJECT
# -------------------------------------------------------

st.divider()

st.subheader("ℹ About FloodGuard AI")

st.write("""
FloodGuard AI is a Deep Learning application that detects flooded regions
from satellite imagery using a trained U-Net segmentation model.

The system predicts flood masks, estimates flood coverage, and provides
risk analysis to assist in disaster monitoring.
""")

# -------------------------------------------------------
# FOOTER
# -------------------------------------------------------

st.markdown("""
<div class="footer">

🌊 <b>FloodGuard AI</b>

<br><br>

Artificial Intelligence & Machine Learning Mini Project

<br><br>

Developed by <b>Akshaya Karasani</b>

</div>
""",unsafe_allow_html=True)