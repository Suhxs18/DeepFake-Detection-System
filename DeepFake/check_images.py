import os
from PIL import Image
import cv2
import numpy as np

def check_images():
    real_dir = "data/raw/real"
    print("Checking real images...")
    print("=" * 50)
    
    for filename in os.listdir(real_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')):
            filepath = os.path.join(real_dir, filename)
            try:
                # Check with PIL
                with Image.open(filepath) as img:
                    print(f"File: {filename}")
                    print(f"  Size: {img.size}")
                    print(f"  Mode: {img.mode}")
                    print(f"  Format: {img.format}")
                    
                # Check with OpenCV
                img_cv = cv2.imread(filepath)
                if img_cv is not None:
                    print(f"  OpenCV shape: {img_cv.shape}")
                    print(f"  File size: {os.path.getsize(filepath) / 1024:.1f} KB")
                else:
                    print(f"  ❌ OpenCV could not read this image")
                    
                print("-" * 30)
                
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")
                print("-" * 30)

if __name__ == "__main__":
    check_images() 