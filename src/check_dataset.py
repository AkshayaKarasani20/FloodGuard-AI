import os
import cv2

IMAGE_DIR = "dataset/images"
MASK_DIR = "dataset/masks"

bad_images = []
bad_masks = []

for file in sorted(os.listdir(IMAGE_DIR)):
    img = cv2.imread(os.path.join(IMAGE_DIR, file))
    if img is None:
        bad_images.append(file)

for file in sorted(os.listdir(MASK_DIR)):
    mask = cv2.imread(os.path.join(MASK_DIR, file), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        bad_masks.append(file)

print("Bad Images:", bad_images)
print("Bad Masks:", bad_masks)
print("Total Bad Images:", len(bad_images))
print("Total Bad Masks:", len(bad_masks))