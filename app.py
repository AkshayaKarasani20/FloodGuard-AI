# =====================================
# IMPORTS
# =====================================

import streamlit as st
import io
import zipfile

from PIL import Image

from src.predict import predict_image
# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="FloodGuard AI",
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

    "bg": "#0F172A",

    "card": "#1E293B",

    "text": "#F8FAFC",

    "secondary": "#CBD5E1",

    "accent": "#38BDF8"

}
# =====================================
# CSS
# =====================================


st.markdown(
f"""
<style>


.stApp {{

background:{COLORS["bg"]};

}}



.block-container {{

padding-top:1rem;

}}



h1,h2,h3 {{

color:{COLORS["text"]};

}}



</style>

""",
unsafe_allow_html=True
)
# =====================================
# SIDEBAR
# =====================================

with st.sidebar:

    # Logo + Title

    st.markdown(
        f"""
        <div style="
        text-align:center;
        padding:10px;
        ">

        <h1 style="
        color:{COLORS["text"]};
        font-size:28px;
        margin-bottom:5px;
        ">
        🌊 FloodGuard AI
        </h1>


        <p style="
        color:{COLORS["secondary"]};
        font-size:13px;
        ">
        AI Flood Monitoring System
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    st.divider()



    # ==============================
    # ABOUT
    # ==============================

    with st.expander("ⓘ About FloodGuard AI"):

        st.write(
            """
            FloodGuard AI is an AI-powered
            flood detection system that analyzes
            satellite images to identify
            possible flood affected regions.
            """
        )



    # ==============================
    # AI SYSTEM
    # ==============================

    with st.expander("🤖 AI System"):

        st.write(
            """
            🧠 Architecture:
            U-Net Deep Learning Model


            ⚡ Optimization:
            Image Segmentation Model


            🛰 Input:
            Satellite Images


            🎯 Task:
            Flood Region Segmentation
            """
        )



    # ==============================
    # HOW IT WORKS
    # ==============================

    with st.expander("🛰 How It Works"):

        st.write(
            """
            1️⃣ Upload Satellite Image


            2️⃣ AI model analyzes the image


            3️⃣ Flood regions are detected


            4️⃣ Flood coverage and risk level
            are generated
            """
        )



    # ==============================
    # MODEL INFORMATION
    # ==============================

    with st.expander("📊 Model Information"):

        st.write(
            """
            🔹 Deep Learning Based


            🔹 U-Net Architecture


            🔹 Binary Image Segmentation


            🔹 Satellite Image Analysis
            """
        )



    # ==============================
    # NEW ANALYSIS
    # ==============================

    with st.expander("🔄 New Analysis"):

        st.write(
            """
            Clear the current prediction
            and upload a new satellite image.
            """
        )


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
<div style="
text-align:center;
margin-top:20px;
">


<div style="
font-size:70px;
">
🌊
</div>


<h1>
FloodGuard AI
</h1>


<p>
AI-Powered Flood Detection from Satellite Images
</p>


</div>
""",
unsafe_allow_html=True
)
# =====================================
# IMAGE UPLOAD
# =====================================

col1, col2, col3 = st.columns([2, 4, 2])


with col2:

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

    with st.spinner(
        "🌊 Analyzing satellite image..."
    ):

        st.session_state.result = predict_image(
            uploaded_file
        )
# =====================================
# IMAGE RESULTS
# =====================================


if st.session_state.result is not None:

    result = st.session_state.result


    st.markdown(
        """
        <br>
        <h2 style="text-align:center;">
        🛰 Detection Results
        </h2>
        """,
        unsafe_allow_html=True
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.write("📷 Original Image")

        st.image(
            result["original"],
            width=300
        )


    with col2:

        st.write("🛰 Flood Overlay")

        st.image(
            result["overlay"],
            width=300
        )


    with col3:

        st.write("🌊 Flood Mask")

        st.image(
            result["mask"],
            width=300
        )
# =====================================
# FLOOD ANALYSIS RESULTS
# =====================================


if st.session_state.result is not None:

    result = st.session_state.result


    st.markdown(
        """
        <br>
        <h2 style="text-align:center;">
        🌊 Flood Analysis
        </h2>
        """,
        unsafe_allow_html=True
    )


    col1, col2, col3 = st.columns(3)



    with col1:

        st.markdown(
            f"""
            <div style="text-align:center;">

            <p style="
            font-size:16px;
            margin-bottom:5px;
            ">
            🌊 Flood Coverage
            </p>


            <p style="
            font-size:24px;
            margin:0;
            ">
            {result["flood_percentage"]}
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )



    with col2:

        st.markdown(
            f"""
            <div style="text-align:center;">

            <p style="
            font-size:16px;
            margin-bottom:5px;
            ">
            ⚠ Risk Level
            </p>


            <p style="
            font-size:24px;
            margin:0;
            ">
            {result["risk_level"]}
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )



    with col3:

        st.markdown(
            f"""
            <div style="text-align:center;">

            <p style="
            font-size:16px;
            margin-bottom:5px;
            ">
            🧠 AI Confidence
            </p>


            <p style="
            font-size:24px;
            margin:0;
            ">
            {result["confidence"]}
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )
# =====================================
# RECOMMENDATIONS
# =====================================


if st.session_state.result is not None:

    result = st.session_state.result


    st.markdown(
        """
        <br>

        <h2 style="text-align:center;">
        💡 Recommendations
        </h2>
        """,
        unsafe_allow_html=True
    )


    recommendation = result["recommendation"]


    st.markdown(
        f"""
        <div style="
        text-align:center;
        font-size:16px;
        line-height:1.8;
        ">

        {recommendation.replace(chr(10), "<br>")}

        </div>
        """,
        unsafe_allow_html=True
    )
# =====================================
# DOWNLOAD BUTTON
# =====================================

if st.session_state.result is not None:

    result = st.session_state.result


    st.download_button(

        label="📥 Download ",

        data="FloodGuard AI Analysis Completed",

        file_name="FloodGuard_Result.txt",

        mime="text/plain"

    )
# =====================================
# FOOTER
# =====================================

st.markdown(
"""
<br>
<hr>


<div style="
text-align:center;
font-size:13px;
color:#CBD5E1;
">


🌊 <b>FloodGuard AI</b><br>

AI-Powered Flood Detection from Satellite Images<br>

Built using Deep Learning and Streamlit


</div>

""",
unsafe_allow_html=True
)
