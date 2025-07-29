#!/usr/bin/env python3
"""
Test predictions on the trained DeepFake detection model.
"""

import os
import sys
import torch
import logging
from PIL import Image
import numpy as np

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils import get_device
from src.dataset import get_transforms
from src.model import DeepfakeDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model(model_path='models/best_model.pth'):
    """Load the trained model."""
    device = get_device()
    
    # Create model
    model = DeepfakeDetector(num_classes=1, dropout_rate=0.5)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded from {model_path}")
    return model, device


def predict_image(model, device, image_path, transform):
    """Predict on a single image."""
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Apply transform
        if hasattr(transform, '__call__'):
            # Albumentations transform
            image_np = np.array(image)
            transformed = transform(image=image_np)
            image_tensor = transformed['image']
        else:
            # PyTorch transform
            image_tensor = transform(image)
        
        image_tensor = image_tensor.unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            output = model(image_tensor)
            probability = torch.sigmoid(output).item()
            prediction = "FAKE" if probability > 0.5 else "REAL"
            
        return prediction, probability
        
    except Exception as e:
        logger.error(f"Error predicting {image_path}: {e}")
        return None, None


def test_predictions():
    """Test predictions on processed data."""
    logger.info("Testing predictions on trained model...")
    
    # Load model
    model, device = load_model()
    
    # Get transforms
    transform = get_transforms(224, is_training=False)
    
    # Test on real data
    logger.info("\n" + "="*50)
    logger.info("TESTING REAL DATA:")
    logger.info("="*50)
    
    real_dir = "data/processed_faces/test/real"
    if os.path.exists(real_dir):
        for filename in os.listdir(real_dir):
            if filename.endswith('.jpg'):
                image_path = os.path.join(real_dir, filename)
                prediction, confidence = predict_image(model, device, image_path, transform)
                if prediction:
                    logger.info(f"{filename}: {prediction} (confidence: {confidence:.3f})")
    
    # Test on fake data
    logger.info("\n" + "="*50)
    logger.info("TESTING FAKE DATA:")
    logger.info("="*50)
    
    fake_dir = "data/processed_faces/test/fake"
    if os.path.exists(fake_dir):
        for filename in os.listdir(fake_dir):
            if filename.endswith('.jpg'):
                image_path = os.path.join(fake_dir, filename)
                prediction, confidence = predict_image(model, device, image_path, transform)
                if prediction:
                    logger.info(f"{filename}: {prediction} (confidence: {confidence:.3f})")
    
    logger.info("\n" + "="*50)
    logger.info("Prediction testing completed!")
    logger.info("="*50)


if __name__ == '__main__':
    test_predictions() 