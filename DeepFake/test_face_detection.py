import os
import cv2
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image
import torch

def test_face_detection():
    # Initialize MTCNN
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(
        image_size=224,
        margin=20,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        post_process=False,
        device=device
    )
    
    real_dir = "data/raw/real"
    print("Testing face detection on real images...")
    print("=" * 60)
    
    for filename in os.listdir(real_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')):
            filepath = os.path.join(real_dir, filename)
            print(f"\nTesting: {filename}")
            print("-" * 40)
            
            try:
                # Load image
                img = Image.open(filepath).convert('RGB')
                print(f"  Image size: {img.size}")
                
                # Try to detect faces
                boxes, probs = mtcnn.detect(img)
                
                if boxes is not None and len(boxes) > 0:
                    print(f"  ✅ Found {len(boxes)} face(s)")
                    for i, (box, prob) in enumerate(zip(boxes, probs)):
                        print(f"    Face {i+1}: Confidence = {prob:.3f}")
                        print(f"    Box: {box}")
                else:
                    print(f"  ❌ No faces detected")
                    
                    # Try with different thresholds
                    print(f"  Trying with lower thresholds...")
                    mtcnn_lower = MTCNN(
                        image_size=224,
                        margin=20,
                        min_face_size=10,  # Lower minimum face size
                        thresholds=[0.3, 0.4, 0.5],  # Lower thresholds
                        factor=0.709,
                        post_process=False,
                        device=device
                    )
                    
                    boxes2, probs2 = mtcnn_lower.detect(img)
                    if boxes2 is not None and len(boxes2) > 0:
                        print(f"    ✅ Found {len(boxes2)} face(s) with lower thresholds")
                        for i, (box, prob) in enumerate(zip(boxes2, probs2)):
                            print(f"      Face {i+1}: Confidence = {prob:.3f}")
                    else:
                        print(f"    ❌ Still no faces detected")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
            
            print("-" * 40)

if __name__ == "__main__":
    test_face_detection() 