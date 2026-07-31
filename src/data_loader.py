import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# -----------------------------
# Paths
# -----------------------------
IMAGE_DIR = "dataset/images"
MASK_DIR = "dataset/masks"

IMG_SIZE = 256


# -----------------------------
# Load Dataset
# -----------------------------
def load_dataset():

    images = []
    masks = []

    image_files = sorted(os.listdir(IMAGE_DIR))

    print(f"Found {len(image_files)} image files.\n")

    for file in image_files:

        image_path = os.path.join(IMAGE_DIR, file)
        mask_path = os.path.join(MASK_DIR, file)

        # Check if mask exists
        if not os.path.exists(mask_path):
            print(f"Mask not found: {file}")
            continue

        # Read image
        image = cv2.imread(image_path)

        if image is None:
            print(f"Cannot read image: {image_path}")
            continue

        # Read mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            print(f"Cannot read mask: {mask_path}")
            continue

        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))

        except Exception as e:
            print(f"Error processing {file}")
            print(e)
            continue

        image = image.astype(np.float32) / 255.0

        mask = mask.astype(np.float32) / 255.0
        mask = np.expand_dims(mask, axis=-1)

        images.append(image)
        masks.append(mask)

    images = np.array(images, dtype=np.float32)
    masks = np.array(masks, dtype=np.float32)

    print("\nFinished loading dataset.")
    print("Images shape:", images.shape)
    print("Masks shape :", masks.shape)

    return train_test_split(
        images,
        masks,
        test_size=0.2,
        random_state=42
    )


# -----------------------------
# Test
# -----------------------------
if __name__ == "__main__":

    X_train, X_test, y_train, y_test = load_dataset()

    print("\n========== DATASET ==========")
    print("Train Images :", X_train.shape)
    print("Train Masks  :", y_train.shape)
    print("Test Images  :", X_test.shape)
    print("Test Masks   :", y_test.shape)