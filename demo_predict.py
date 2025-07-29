#!/usr/bin/env python3
"""
Demo Prediction Script for DeepFake Detection System
This script demonstrates how to use the trained model for predictions.
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model import DeepfakeDetector
from src.utils import get_device

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_path):
    """Load the trained model"""
    device = get_device()
    model = DeepfakeDetector(num_classes=1, dropout_rate=0.5)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model, device

def preprocess_image(image_path):
    """Preprocess a single image for prediction"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    
    # Convert to tensor
    img_tensor = torch.FloatTensor(np.array(img)) / 255.0
    img_tensor = img_tensor.permute(2, 0, 1)  # HWC to CHW
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    
    return img_tensor

def predict_image(model, device, image_path):
    """Predict if an image is real or fake"""
    # Preprocess image
    img_tensor = preprocess_image(image_path)
    img_tensor = img_tensor.to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(img_tensor)
        probability = torch.sigmoid(output).item()
        prediction = "FAKE" if probability > 0.5 else "REAL"
    
    return prediction, probability

def demo_predict():
    """Demo prediction function"""
    logger.info("Starting Demo Prediction...")
    
    # Check if model exists
    model_path = "models/demo_model.pth"
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}. Please run training first.")
        return
    
    # Load model
    model, device = load_model(model_path)
    logger.info(f"Model loaded from {model_path}")
    logger.info(f"Using device: {device}")
    
    # Test on training data
    fake_dir = "data/processed_faces/train/fake"
    real_dir = "data/processed_faces/train/real"
    
    logger.info("\n" + "="*50)
    logger.info("PREDICTION RESULTS")
    logger.info("="*50)
    
    # Test fake images
    if os.path.exists(fake_dir):
        logger.info("\nTesting FAKE images:")
        for img_name in os.listdir(fake_dir):
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(fake_dir, img_name)
                prediction, probability = predict_image(model, device, img_path)
                logger.info(f"{img_name}: {prediction} (confidence: {probability:.3f})")
    
    # Test real images (if any)
    if os.path.exists(real_dir) and len(os.listdir(real_dir)) > 0:
        logger.info("\nTesting REAL images:")
        for img_name in os.listdir(real_dir):
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(real_dir, img_name)
                prediction, probability = predict_image(model, device, img_path)
                logger.info(f"{img_name}: {prediction} (confidence: {probability:.3f})")
    
    logger.info("\n" + "="*50)
    logger.info("Demo prediction completed!")

if __name__ == "__main__":
    demo_predict() 