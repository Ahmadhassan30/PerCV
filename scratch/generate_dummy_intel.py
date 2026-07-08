import os
import cv2
import numpy as np

base_dir = "scratch/dummy_intel"
classes = ["buildings", "forest", "mountain", "street"]

for split in ["train", "test"]:
    for cls in classes:
        cls_dir = os.path.join(base_dir, split, cls)
        os.makedirs(cls_dir, exist_ok=True)
        
        # Write 5 dummy images per class/split
        for i in range(5):
            img = np.zeros((128, 128, 3), dtype=np.uint8)
            # Draw distinct color squares to help dummy network train/predict
            if cls == "buildings":
                cv2.rectangle(img, (20, 20), (100, 100), (255, 0, 0), -1)
            elif cls == "forest":
                cv2.circle(img, (64, 64), 40, (0, 255, 0), -1)
            elif cls == "mountain":
                # Triangle
                pts = np.array([[64, 20], [20, 100], [108, 100]], dtype=np.int32)
                cv2.fillPoly(img, [pts], (0, 0, 255))
            else:
                # Street: vertical stripe
                cv2.rectangle(img, (48, 0), (80, 128), (255, 255, 0), -1)
                
            cv2.imwrite(os.path.join(cls_dir, f"img_{i:02d}.jpg"), img)

print("Generated dummy Intel dataset under scratch/dummy_intel")
