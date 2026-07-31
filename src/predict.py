import tensorflow as tf
import cv2
import numpy as np

# Load model only once
model = tf.keras.models.load_model(
    "models/best_model.keras",
    compile=False
)


def predict_image(image):
    """
    Predict flood mask from uploaded image.

    Returns:
    original image,
    predicted mask,
    overlay image,
    flood percentage
    """

    original = image.copy()

    # Resize to model input size
    resized = cv2.resize(image, (256, 256))

    # Normalize
    resized = resized.astype(np.float32) / 255.0
    resized = np.expand_dims(resized, axis=0)

    # Predict
    prediction = model.predict(resized, verbose=0)

    mask = prediction[0, :, :, 0]
    mask = (mask > 0.5).astype(np.uint8)

    # Flood percentage
    flood_percentage = round(mask.mean() * 100, 2)

    # Create overlay
    overlay = cv2.resize(original, (256, 256))

    red = np.zeros_like(overlay)
    red[:, :, 0] = 255

    overlay = np.where(
        mask[:, :, np.newaxis] == 1,
        (0.5 * overlay + 0.5 * red).astype(np.uint8),
        overlay
    )

    return original, mask, overlay, flood_percentage