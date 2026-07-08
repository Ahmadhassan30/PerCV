import os
import cv2
import numpy as np

out_dir = "scratch/dummy_bdd100k"
os.makedirs(out_dir, exist_ok=True)

for i in range(20):
    # Create black canvas
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw passing lane markings (approx 45 degrees)
    # y = x - 100
    cv2.line(img, (100, 0), (480, 380), (255, 255, 255), 5)
    
    # Draw failing lines (approx 90 degrees / vertical)
    cv2.line(img, (320, 0), (320, 480), (255, 255, 255), 5)
    
    cv2.imwrite(os.path.join(out_dir, f"road_{i:02d}.jpg"), img)

print("Generated 20 dummy road images under scratch/dummy_bdd100k")
