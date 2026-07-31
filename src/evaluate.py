import tensorflow as tf
from data_loader import load_dataset
from sklearn.metrics import accuracy_score
import numpy as np


model = tf.keras.models.load_model(
    "models/best_model.keras",
    compile=False
)


X_train, X_test, y_train, y_test = load_dataset()


predictions = model.predict(X_test)


predictions = (predictions > 0.5).astype(np.float32)


accuracy = np.mean(predictions == y_test)


print("==========================")
print("MODEL EVALUATION")
print("==========================")

print("Test Accuracy:", accuracy*100,"%")
print("Test Images:", X_test.shape)
print("Test Masks:", y_test.shape)