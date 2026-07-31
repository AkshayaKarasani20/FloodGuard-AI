import tensorflow as tf

from data_loader import load_dataset
from model import build_model
from losses import combined_loss, dice_coef, iou_metric


# -----------------------------
# Load Dataset
# -----------------------------
print("Loading dataset...\n")

X_train, X_test, y_train, y_test = load_dataset()

print("\nTraining Images :", X_train.shape)
print("Training Masks  :", y_train.shape)
print("Testing Images  :", X_test.shape)
print("Testing Masks   :", y_test.shape)


# -----------------------------
# Build Model
# -----------------------------
model = build_model()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=combined_loss,
    metrics=[
        "accuracy",
        dice_coef,
        iou_metric
    ]
)

model.summary()


# -----------------------------
# Callbacks
# -----------------------------
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "models/best_model.keras",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    verbose=1
)


# -----------------------------
# Train
# -----------------------------
print("\nStarting Training...\n")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=8,
    callbacks=[
        checkpoint,
        early_stop,
        reduce_lr
    ]
)


# -----------------------------
# Save Final Model
# -----------------------------
model.save("models/flood_model.keras")

print("\nTraining Completed Successfully!")
print("Models saved inside models folder.")