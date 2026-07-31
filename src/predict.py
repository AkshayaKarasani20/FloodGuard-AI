import os
import time

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from src.losses import combined_loss, dice_coef, iou_metric


# =====================================
# MODEL CACHE
# =====================================

_model = None


def load_my_model():

    global _model

    if _model is None:

        model_path = os.path.join(
            "models",
            "best_model.keras"
        )

        _model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "combined_loss": combined_loss,
                "dice_coef": dice_coef,
                "iou_metric": iou_metric
            },
            compile=False
        )

    return _model



# =====================================
# RECOMMENDATIONS
# =====================================

def get_recommendation(risk_level):

    if "Low" in risk_level:

        return (
            "No major flood indication detected.\n\n"
            "• Continue monitoring weather updates.\n"
            "• No immediate action is required."
        )


    elif "Moderate" in risk_level:

        return (
            "Possible flood affected regions detected.\n\n"
            "• Review local water levels.\n"
            "• Follow official weather alerts."
        )


    elif "High" in risk_level:

        return (
            "High flood possibility detected.\n\n"
            "• Avoid unnecessary travel.\n"
            "• Prepare emergency supplies.\n"
            "• Follow instructions from local authorities."
        )


    else:

        return (
            "Severe flood indication detected.\n\n"
            "• Move to a safer location if instructed.\n"
            "• Follow emergency warnings immediately."
        )



# =====================================
# PREDICTION
# =====================================

def predict_image(uploaded_file):


    start_time = time.time()


    model = load_my_model()



    # =================================
    # READ IMAGE
    # =================================

    image = Image.open(
        uploaded_file
    ).convert(
        "RGB"
    )


    image = np.array(image)


    original = image.copy()



    # =================================
    # PREPROCESSING
    # =================================

    resized = cv2.resize(
        image,
        (256,256)
    )


    resized = resized.astype(
        np.float32
    ) / 255.0



    resized = np.expand_dims(
        resized,
        axis=0
    )



    # =================================
    # MODEL PREDICTION
    # =================================

    prediction = model.predict(
        resized,
        verbose=0
    )


    # AI Confidence

    confidence = round(
        float(np.max(prediction) * 100),
        2
    )



    # =================================
    # CREATE MASK
    # =================================

    mask = prediction[0,:,:,0]


    mask = cv2.resize(
        mask,
        (
            original.shape[1],
            original.shape[0]
        )
    )


    binary_mask = (
        mask > 0.5
    ).astype(
        np.uint8
    )



    # =================================
    # FLOOD COVERAGE
    # =================================

    flood_percentage = round(
        float(binary_mask.mean() * 100),
        2
    )



    # =================================
    # RISK LEVEL
    # =================================

    if flood_percentage < 10:

        risk_level = "Low 🟢"


    elif flood_percentage < 30:

        risk_level = "Moderate 🟡"


    elif flood_percentage < 60:

        risk_level = "High 🟠"


    else:

        risk_level = "Severe 🔴"



    # =================================
    # FLOOD OVERLAY
    # =================================

    overlay = original.copy()


    red = np.zeros_like(
        overlay
    )


    red[:,:,0] = 255



    overlay = np.where(
        binary_mask[:,:,None] == 1,

        (
            0.4 * overlay +
            0.6 * red
        ).astype(
            np.uint8
        ),

        overlay
    )



    # =================================
    # MASK IMAGE
    # =================================

    mask_image = (
        binary_mask * 255
    ).astype(
        np.uint8
    )



    # =================================
    # PROCESSING TIME
    # =================================

    processing_time = round(
        time.time() - start_time,
        2
    )



    # =================================
    # RETURN RESULTS
    # =================================

    return {


        "original": original,


        "overlay": overlay,


        "mask": mask_image,


        "flood_percentage":
            f"{flood_percentage}%",



        "risk_level":
            risk_level,



        "confidence":
            f"{confidence}%",



        "recommendation":
            get_recommendation(
                risk_level
            ),



        "processing_time":
            processing_time

    }