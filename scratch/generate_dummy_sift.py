import os
import cv2
import numpy as np

out_dir = "scratch/dummy_matching"
os.makedirs(out_dir, exist_ok=True)

# Generate first image with high-gradient features (shapes, text)
img1 = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.rectangle(img1, (100, 100), (300, 300), (255, 255, 255), -1)
cv2.circle(img1, (450, 200), 80, (200, 200, 200), -1)
cv2.putText(img1, "PerCV SIFT", (150, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (150, 150, 150), 3)

# Generate second image rotated and translated slightly
M = cv2.getRotationMatrix2D((320, 240), 15, 1.0)  # 15 degrees rotation
img2 = cv2.warpAffine(img1, M, (640, 480))

cv2.imwrite(os.path.join(out_dir, "img1.png"), img1)
cv2.imwrite(os.path.join(out_dir, "img2.png"), img2)

print("Generated dummy SIFT match pair under scratch/dummy_matching")
