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

if "theme" not in st.session_state:
    st.session_state.theme = "Dark"
# =====================================
# THEME COLORS
# =====================================

if st.session_state.theme == "Dark":

    COLORS = {
        "bg": "#0F172A",
        "card": "#1E293B",
        "border": "#334155",
        "text": "#F8FAFC",
        "secondary": "#CBD5E1",
        "accent": "#38BDF8",
        "success": "#22C55E",
        "warning": "#F59B0B",
        "danger": "#EF4444"
    }

else:

    COLORS = {
        "bg": "#F8FAFC",
        "card": "#FFFFFF",
        "border": "#CBD5E1",
        "text": "#0F172A",
        "secondary": "#475569",
        "accent": "#0284C7",
        "success": "#16A34A",
        "warning": "#D97706",
        "danger": "#DC2626"
    }
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
            Light U-Net


            ⚡ Optimization:
            Lightweight Deep Learning Model


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


            2️⃣ AI Model analyzes the image


            3️⃣ Flood regions are detected


            4️⃣ Risk level and recommendations
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


            🔹 Image Segmentation


            🔹 Automated Flood Analysis


            🔹 Visual Flood Mapping
            """
        )
# =====================================
# COMPACT FULL SCREEN CSS
# =====================================

st.markdown(
f"""
<style>

.stApp {{

    background:{COLORS["bg"]};
    color:{COLORS["text"]};

}}


.block-container {{

    max-width:1500px;
    width:92%;
    padding-top:0.3rem;
    padding-left:1rem;
    padding-right:1rem;
    padding-bottom:1rem;

}}


.title {{

    text-align:center;
    font-size:32px;
    font-weight:700;
    margin:0;

}}


.subtitle {{

    text-align:center;
    font-size:14px;
    margin-bottom:10px;

}}


.card {{

    background:{COLORS["card"]};
    border-radius:12px;
    padding:8px;

}}


.metric-card {{

    background:{COLORS["card"]};
    border-radius:10px;
    padding:8px;

}}


/* Reduce all vertical spacing */

.stMarkdown {{

    margin-bottom:5px;

}}


</style>
""",
unsafe_allow_html=True
)
# =====================================
# HEADER
# =====================================

col1, col2 = st.columns([8, 2])


with col2:

    dark_mode = st.toggle(
        "🌙 Dark Mode",
        value=(st.session_state.theme == "Dark")
    )

    selected_theme = "Dark" if dark_mode else "Light"


    if selected_theme != st.session_state.theme:

        st.session_state.theme = selected_theme

        st.rerun()



st.markdown(
f"""
<div style="text-align:center; margin-top:30px;">

<div style="
font-size:70px;
">
🌊
</div>


<div class="title">
FloodGuard AI
</div>


<div class="subtitle">
AI-Powered Flood Detection from Satellite Images
</div>


</div>
""",
unsafe_allow_html=True
)
# =====================================
# UPLOAD SECTION
# =====================================

st.markdown("<br>", unsafe_allow_html=True)


col1, col2, col3 = st.columns([2,4,2])


with col2:

    uploaded_file = st.file_uploader(
        "📤 Upload",
        type=[
            "png",
            "jpg",
            "jpeg",
            "tif",
            "tiff"
        ]
    )


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
    f"""
    <h3 style="
    color:{COLORS["text"]};
    font-size:20px;
    margin-top:15px;
    margin-bottom:10px;
    ">
    🛰 Detection Results
    </h3>
    """,
    unsafe_allow_html=True
    )


    img1, img2, img3 = st.columns(
        3,
        gap="medium"
    )


    def display_image(column, title, image):

        with column:

            st.markdown(
            f"""
            <p style="
            color:{COLORS["text"]};
            font-size:15px;
            font-weight:600;
            text-align:center;
            margin-bottom:8px;
            ">
            {title}
            </p>
            """,
            unsafe_allow_html=True
            )


            st.image(
                image,
                width=300
            )



    display_image(
        img1,
        "📷 Original",
        result["original"]
    )


    display_image(
        img2,
        "🛰 Overlay",
        result["overlay"]
    )


    display_image(
        img3,
        "🌊 Flood Mask",
        result["mask"]
    )
# =====================================
# STEP 7
# FLOOD COVERAGE & RISK LEVEL
# =====================================

if st.session_state.result is not None:

    result = st.session_state.result


    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )


    col1, col2 = st.columns(
        2,
        gap="large"
    )

# =====================================
# FLOOD COVERAGE, RISK, CONFIDENCE
# =====================================

if st.session_state.result is not None:

    result = st.session_state.result


    col1, col2, col3 = st.columns(3)

# =====================================
# STEP 7
# FLOOD SEVERITY - RISK - CONFIDENCE
# =====================================

if st.session_state.result is not None:

    result = st.session_state.result


    col1, col2, col3 = st.columns(
        3,
        gap="medium"
    )


    # ==============================
    # FLOOD SEVERITY
    # ==============================

    with col1:

        flood_text = result["flood_percentage"]


        flood_number = float(
            flood_text.replace("%", "")
        )


        flood_number = max(
            0,
            min(
                flood_number,
                100
            )
        )


        st.markdown(
        f"""
        <p style="
        color:{COLORS['secondary']};
        font-size:14px;
        font-weight:600;
        margin:0;
        ">
        🌊 Flood Severity
        </p>


        <p style="
        color:{COLORS['text']};
        font-size:24px;
        font-weight:700;
        margin:5px 0;
        ">
        {flood_text}
        </p>


        <div style="
        background:{COLORS['border']};
        height:8px;
        width:100%;
        border-radius:10px;
        ">

            <div style="
            background:{COLORS['accent']};
            height:8px;
            width:{flood_number}%;
            border-radius:10px;
            ">
            </div>

        </div>

        """,
        unsafe_allow_html=True
        )



    # ==============================
    # RISK LEVEL
    # ==============================

    with col2:

        risk = result["risk_level"]


        if "Low" in risk:

            risk_color = COLORS["success"]

        elif "Moderate" in risk:

            risk_color = COLORS["warning"]

        else:

            risk_color = COLORS["danger"]



        st.markdown(
        f"""
        <p style="
        color:{COLORS['secondary']};
        font-size:14px;
        font-weight:600;
        margin:0;
        ">
        ⚠ Risk Level
        </p>


        <p style="
        color:{risk_color};
        font-size:24px;
        font-weight:700;
        margin:5px 0;
        ">
        {risk}
        </p>

        """,
        unsafe_allow_html=True
        )



    # ==============================
    # AI CONFIDENCE
    # ==============================

    with col3:

        confidence = result["confidence"]


        st.markdown(
        f"""
        <p style="
        color:{COLORS['secondary']};
        font-size:14px;
        font-weight:600;
        margin:0;
        ">
        🧠 AI Confidence
        </p>


        <p style="
        color:{COLORS['text']};
        font-size:24px;
        font-weight:700;
        margin:5px 0;
        ">
        {confidence}
        </p>

        """,
        unsafe_allow_html=True
        )
# =====================================
# STEP 8
# RECOMMENDATIONS
# =====================================

if st.session_state.result is not None:

    result = st.session_state.result


    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )


    st.markdown(
    f"""
    <p style="
    color:{COLORS["secondary"]};
    font-size:20px;
    font-weight:600;
    margin-bottom:5px;
    ">
    💡 Recommendations
    </p>
    """,
    unsafe_allow_html=True
    )


    recommendation = result["recommendation"]


    # Convert new lines safely

    recommendation_html = recommendation.replace(
        "\n",
        "<br>"
    )


    st.markdown(
    f"""
    <div style="
    color:{COLORS["text"]};
    font-size:16px;
    line-height:1.6;
    ">

    {recommendation_html}

    </div>
    """,
    unsafe_allow_html=True
    )
# =====================================
# DOWNLOAD BUTTON
# =====================================

if st.session_state.result is not None:

    result = st.session_state.result


    zip_buffer = io.BytesIO()


    with zipfile.ZipFile(
        zip_buffer,
        "w"
    ) as zip_file:


        # Overlay

        overlay_buffer = io.BytesIO()

        Image.fromarray(
            result["overlay"]
        ).save(
            overlay_buffer,
            format="PNG"
        )

        zip_file.writestr(
            "flood_overlay.png",
            overlay_buffer.getvalue()
        )


        # Mask

        mask_buffer = io.BytesIO()

        Image.fromarray(
            result["mask"]
        ).save(
            mask_buffer,
            format="PNG"
        )

        zip_file.writestr(
            "flood_mask.png",
            mask_buffer.getvalue()
        )


        # Report

        report = f"""
FloodGuard AI Report

Flood Coverage:
{result["flood_percentage"]}

Risk Level:
{result["risk_level"]}


Recommendations:

{result["recommendation"]}
"""

        zip_file.writestr(
            "flood_report.txt",
            report
        )


    zip_buffer.seek(0)


    st.download_button(
        label="📥 Download",
        data=zip_buffer,
        file_name="FloodGuard_Result.zip",
        mime="application/zip"
    )
# =====================================
# FOOTER
# =====================================

st.markdown(
f"""
<br><br>

<hr style="
border:1px solid {COLORS["border"]};
">


<div style="
text-align:center;
color:{COLORS["secondary"]};
font-size:13px;
line-height:1.6;
">

🌊 <b>FloodGuard AI</b><br>

AI-Powered Flood Detection from Satellite Images<br>

Built using Deep Learning and Streamlit

</div>

""",
unsafe_allow_html=True
)
# =====================================
# RESET ANALYSIS
# =====================================

if st.session_state.result is not None:

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )


    if st.button(
        "🔄 New Analysis",
        use_container_width=True
    ):

        st.session_state.result = None
        st.session_state.uploaded_file = None

        st.rerun()