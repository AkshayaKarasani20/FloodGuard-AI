import tensorflow as tf


# -----------------------------
# Dice Loss
# -----------------------------
def dice_loss(y_true, y_pred):
    smooth = 1e-6

    y_true = tf.keras.backend.flatten(y_true)
    y_pred = tf.keras.backend.flatten(y_pred)

    intersection = tf.reduce_sum(y_true * y_pred)

    dice = (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true) +
        tf.reduce_sum(y_pred) +
        smooth
    )

    return 1.0 - dice


# -----------------------------
# Binary Cross Entropy
# -----------------------------
def bce_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred)


# -----------------------------
# Combined Loss
# -----------------------------
def combined_loss(y_true, y_pred):
    bce = bce_loss(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)

    return 0.5 * bce + 0.5 * dice


# -----------------------------
# Dice Coefficient
# -----------------------------
def dice_coef(y_true, y_pred):
    smooth = 1e-6

    y_true = tf.keras.backend.flatten(y_true)
    y_pred = tf.keras.backend.flatten(y_pred)

    y_pred = tf.cast(y_pred > 0.5, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred)

    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true) +
        tf.reduce_sum(y_pred) +
        smooth
    )


# -----------------------------
# IoU Metric
# -----------------------------
def iou_metric(y_true, y_pred):
    smooth = 1e-6

    y_true = tf.keras.backend.flatten(y_true)
    y_pred = tf.keras.backend.flatten(y_pred)

    y_pred = tf.cast(y_pred > 0.5, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred)
    union = (
        tf.reduce_sum(y_true) +
        tf.reduce_sum(y_pred) -
        intersection
    )

    return (intersection + smooth) / (union + smooth)