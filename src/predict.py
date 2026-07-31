import os
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model


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
# CUSTOM METRICS
# =====================================

def dice_coef(y_true, y_pred):

    smooth = 1e-7

    y_true = tf.keras.backend.flatten(y_true)
    y_pred = tf.keras.backend.flatten(y_pred)

    intersection = tf.reduce_sum(
        y_true * y_pred
    )

    return (
        (2.0 * intersection + smooth)
        /
        (
            tf.reduce_sum(y_true)
            +
            tf.reduce_sum(y_pred)
            +
            smooth
        )
    )



def iou_metric(y_true, y_pred):

    smooth = 1e-7

    y_true = tf.keras.backend.flatten(y_true)
    y_pred = tf.keras.backend.flatten(y_pred)


    intersection = tf.reduce_sum(
        y_true * y_pred
    )


    union = (
        tf.reduce_sum(y_true)
        +
        tf.reduce_sum(y_pred)
        -
        intersection
    )


    return (
        (intersection + smooth)
        /
        (union + smooth)
    )



# =====================================
# CUSTOM LOSS
# =====================================

def combined_loss(y_true, y_pred):

    bce = tf.keras.losses.binary_crossentropy(
        y_true,
        y_pred
    )


    dice_loss = 1 - dice_coef(
        y_true,
        y_pred
    )


    return bce + dice_loss




# =====================================
# LOAD MODEL
# =====================================

model = load_model(
    MODEL_PATH,
    custom_objects={
        "combined_loss": combined_loss,
        "dice_coef": dice_coef,
        "iou_metric": iou_metric
    }
)



# =====================================
# IMAGE SIZE
# =====================================

IMG_SIZE = 256




# =====================================
# PREPROCESS IMAGE
# =====================================

def preprocess_image(image):

    image = image.convert(
        "RGB"
    )


    original = np.array(
        image
    )


    resized = cv2.resize(
        original,
        (
            IMG_SIZE,
            IMG_SIZE
        )
    )


    resized = resized / 255.0


    resized = np.expand_dims(
        resized,
        axis=0
    )


    return resized, original




# =====================================
# FLOOD ANALYSIS
# =====================================

def analyze_flood(mask):

    flood_pixels = np.sum(
        mask > 0.5
    )


    total_pixels = mask.size


    flood_percentage = (
        flood_pixels /
        total_pixels
    ) * 100


    flood_percentage = round(
        float(flood_percentage),
        2
    )


    if flood_percentage < 10:

        risk_level = "Low 🟢"

        recommendation = (
            "No major flood indication detected.\n\n"
            "• Continue monitoring weather updates.\n"
            "• No immediate action is required."
        )


    elif flood_percentage < 40:

        risk_level = "Moderate 🟡"

        recommendation = (
            "Possible flood affected regions detected.\n\n"
            "• Monitor weather conditions.\n"
            "• Stay updated with local warnings."
        )


    else:

        risk_level = "High 🔴"

        recommendation = (
            "Large flood affected regions detected.\n\n"
            "• Follow emergency guidelines.\n"
            "• Avoid affected areas."
        )


    return (
        flood_percentage,
        risk_level,
        recommendation
    )





# =====================================
# CREATE OVERLAY
# =====================================

def create_overlay(
        original,
        mask
):

    mask = cv2.resize(
        mask,
        (
            original.shape[1],
            original.shape[0]
        )
    )


    overlay = original.copy()


    overlay[
        mask > 0
    ] = [
        0,
        0,
        255
    ]


    result = cv2.addWeighted(
        original,
        0.7,
        overlay,
        0.3,
        0
    )


    return result





# =====================================
# MAIN PREDICTION FUNCTION
# =====================================

def predict_image(uploaded_file):


    image = Image.open(
        uploaded_file
    )


    processed_image, original = preprocess_image(
        image
    )



    prediction = model.predict(
        processed_image
    )



    mask = prediction[0]



    if mask.shape[-1] == 1:

        mask = np.squeeze(
            mask,
            axis=-1
        )



    binary_mask = (
        mask > 0.5
    ).astype(
        np.uint8
    )



    flood_percentage, risk_level, recommendation = analyze_flood(
        mask
    )



    confidence = round(
        float(np.max(mask) * 100),
        2
    )



    overlay = create_overlay(
        original,
        binary_mask
    )



    return {

        "original":
            original,


        "mask":
            binary_mask * 255,


        "overlay":
            overlay,


        "flood_percentage":
            f"{flood_percentage}%",


        "risk_level":
            risk_level,


        "confidence":
            f"{confidence}%",


        "recommendation":
            recommendation

    }