import os
import cv2
import numpy as np

out_dir = "scratch/dummy_panorama"
os.makedirs(out_dir, exist_ok=True)

# Create a wide synthetic canvas (1200 x 480) with high-gradient details
canvas = np.zeros((480, 1200, 3), dtype=np.uint8)

# Draw shapes, textures, and lines across the canvas
cv2.rectangle(canvas, (50, 50), (250, 250), (255, 0, 0), -1)
cv2.circle(canvas, (400, 200), 120, (0, 255, 0), -1)
cv2.rectangle(canvas, (650, 150), (850, 350), (0, 0, 255), -1)
cv2.circle(canvas, (1000, 250), 90, (255, 255, 0), -1)
cv2.putText(canvas, "PerCV Panorama Stitching Pipeline", (200, 420),
            cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 4)

# Add SIFT features (grid of small circles)
for y in range(80, 400, 60):
    for x in range(80, 1100, 60):
        cv2.circle(canvas, (x, y), 5, (128, 128, 128), -1)

# Extract three overlapping frames of size 640x480
# Overlap matches: Left/Middle: 640 - 280 = 360 pixels overlap
# Overlap matches: Middle/Right: (280 + 640) - 560 = 920 - 560 = 360 pixels overlap
left = canvas[:, 0:640]
middle = canvas[:, 280:920]
right = canvas[:, 560:1200]

cv2.imwrite(os.path.join(out_dir, "left.png"), left)
cv2.imwrite(os.path.join(out_dir, "middle.png"), middle)
cv2.imwrite(os.path.join(out_dir, "right.png"), right)

print("Generated dummy panorama images under scratch/dummy_panorama")
