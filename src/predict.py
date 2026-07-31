import os
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf


# =====================================
# MODEL PATH
# =====================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_model.keras"
)


# =====================================
# LOAD MODEL
# =====================================

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)



# =====================================
# PREPROCESS IMAGE
# =====================================

def preprocess_image(image):

    image = np.array(image)


    # Remove alpha channel

    if image.shape[-1] == 4:
        image = image[:, :, :3]


    # IMPORTANT:
    # Model was trained with 256x256

    image = cv2.resize(
        image,
        (256, 256)
    )


    # Normalize

    image = image / 255.0


    # Add batch dimension

    image = np.expand_dims(
        image,
        axis=0
    )


    return image



# =====================================
# RISK LEVEL
# =====================================

def get_risk_level(percentage):

    if percentage < 10:
        return "Low 🟢"

    elif percentage < 40:
        return "Medium 🟡"

    else:
        return "High 🔴"



# =====================================
# RECOMMENDATIONS
# =====================================

def get_recommendation(risk):

    if "High" in risk:

        return (
            "Flood indication detected.\n"
            "• Avoid flood affected areas.\n"
            "• Follow emergency updates.\n"
            "• Stay alert."
        )


    elif "Medium" in risk:

        return (
            "Moderate flood possibility detected.\n"
            "• Monitor weather updates.\n"
            "• Stay prepared."
        )


    else:

        return (
            "No major flood indication detected.\n"
            "• Continue monitoring weather updates.\n"
            "• No immediate action is required."
        )



# =====================================
# MAIN PREDICTION
# =====================================

def predict_image(uploaded_file):


    # Open image

    image = Image.open(
        uploaded_file
    ).convert(
        "RGB"
    )


    original = np.array(
        image
    )



    # Preprocess

    processed = preprocess_image(
        image
    )



    # Prediction

    prediction = model.predict(
        processed
    )


    mask = prediction[0]



    # Convert probability mask

    mask = (
        mask > 0.5
    ).astype(
        np.uint8
    )



    # Remove channel dimension

    if mask.shape[-1] == 1:

        mask = np.squeeze(
            mask,
            axis=-1
        )



    # Resize mask to original image size

    mask = cv2.resize(
        mask,
        (
            original.shape[1],
            original.shape[0]
        )
    )



    # =====================================
    # FLOOD PERCENTAGE
    # =====================================

    flood_pixels = np.sum(
        mask == 1
    )

    total_pixels = mask.size


    flood_percentage = (
        flood_pixels /
        total_pixels
    ) * 100


    flood_percentage = round(
        flood_percentage,
        2
    )



    # =====================================
    # OVERLAY
    # =====================================

    overlay = original.copy()


    overlay[mask == 1] = [
        255,
        0,
        0
    ]


    overlay = cv2.addWeighted(
        original,
        0.7,
        overlay,
        0.3,
        0
    )



    # =====================================
    # DETAILS
    # =====================================

    confidence = round(
        float(np.max(prediction)) * 100,
        2
    )


    risk = get_risk_level(
        flood_percentage
    )


    recommendation = get_recommendation(
        risk
    )



    return {

        "original":
            original,


        "mask":
            mask * 255,


        "overlay":
            overlay,


        "flood_percentage":
            f"{flood_percentage}%",


        "risk_level":
            risk,


        "confidence":
            f"{confidence}%",


        "recommendation":
            recommendation

    }