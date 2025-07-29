#!/usr/bin/env python3
"""
Example usage script for DeepFake Detection System.

This script demonstrates how to use the various components of the system
for data preprocessing, model training, and inference.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils import get_device, extract_face
from src.dataset import DeepfakeDataset, get_transforms
from src.model import DeepfakeDetector, create_model
from facenet_pytorch import MTCNN


def example_1_device_detection():
    """Example 1: Device detection and basic setup."""
    print("=" * 60)
    print("EXAMPLE 1: Device Detection")
    print("=" * 60)
    
    # Get the best available device
    device = get_device()
    print(f"Using device: {device}")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("CUDA not available, using CPU")


def example_2_face_extraction():
    """Example 2: Face extraction from images."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Face Extraction")
    print("=" * 60)
    
    # Initialize MTCNN for face detection
    device = get_device()
    mtcnn = MTCNN(image_size=224, margin=20, keep_all=False, device=device)
    
    # Example image path (you would replace this with your actual image)
    example_image_path = "data/raw/real/example.jpg"
    
    if os.path.exists(example_image_path):
        print(f"Extracting face from: {example_image_path}")
        
        # Extract face
        face = extract_face(
            image_path=example_image_path,
            mtcnn_model=mtcnn,
            save_path="extracted_face.jpg",
            size=(224, 224)
        )
        
        if face is not None:
            print(f"Face extracted successfully! Shape: {face.shape}")
            print("Face saved as: extracted_face.jpg")
        else:
            print("No face detected in the image")
    else:
        print(f"Example image not found: {example_image_path}")
        print("Please add an image to test face extraction")


def example_3_dataset_usage():
    """Example 3: Dataset loading and usage."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Dataset Usage")
    print("=" * 60)
    
    # Get transforms
    train_transforms = get_transforms(image_size=224, is_training=True)
    val_transforms = get_transforms(image_size=224, is_training=False)
    
    # Example dataset paths
    train_dir = "data/processed_faces/train"
    val_dir = "data/processed_faces/val"
    
    if os.path.exists(train_dir):
        try:
            # Create dataset
            train_dataset = DeepfakeDataset(train_dir, transform=train_transforms)
            print(f"Training dataset size: {len(train_dataset)}")
            
            # Get class distribution
            distribution = train_dataset.get_class_distribution()
            print(f"Class distribution: {distribution}")
            
            # Get a sample
            if len(train_dataset) > 0:
                image, label = train_dataset[0]
                print(f"Sample image shape: {image.shape}")
                print(f"Sample label: {label} ({train_dataset.class_names[label]})")
                
                # Get class weights for imbalanced datasets
                class_weights = train_dataset.get_class_weights()
                print(f"Class weights: {class_weights}")
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Make sure you have processed data in the correct directory structure")
    else:
        print(f"Training directory not found: {train_dir}")
        print("Please run preprocessing first: python scripts/preprocess_data.py")


def example_4_model_creation():
    """Example 4: Model creation and forward pass."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Model Creation")
    print("=" * 60)
    
    device = get_device()
    
    # Create model with different configurations
    print("Creating basic model...")
    model = DeepfakeDetector(
        num_classes=1,
        dropout_rate=0.5,
        backbone='resnet18',
        pretrained=False  # Use False for testing to avoid downloading weights
    )
    model = model.to(device)
    
    # Get model information
    model_info = model.get_model_info()
    print(f"Model parameters: {model_info['total_parameters']:,}")
    print(f"Trainable parameters: {model_info['trainable_parameters']:,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 224, 224).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        print(f"Input shape: {input_tensor.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    # Test predictions
    print("\nTesting predictions...")
    probs = model.predict_proba(input_tensor)
    predictions = model.predict(input_tensor)
    print(f"Probabilities shape: {probs.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[:3].cpu().numpy()}")


def example_5_model_factory():
    """Example 5: Using the model factory function."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Model Factory")
    print("=" * 60)
    
    device = get_device()
    
    # Different model configurations
    configs = [
        {
            'model_type': 'deepfake_detector',
            'num_classes': 1,
            'backbone': 'resnet18',
            'dropout_rate': 0.5
        },
        {
            'model_type': 'deepfake_detector',
            'num_classes': 1,
            'backbone': 'resnet50',
            'dropout_rate': 0.3
        }
    ]
    
    for i, config in enumerate(configs):
        print(f"\nCreating model {i+1} with config: {config}")
        
        model = create_model(config)
        model = model.to(device)
        
        model_info = model.get_model_info()
        print(f"  Parameters: {model_info['total_parameters']:,}")
        print(f"  Backbone: {model_info['backbone']}")
        
        # Test forward pass
        input_tensor = torch.randn(2, 3, 224, 224).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            print(f"  Output shape: {output.shape}")


def example_6_training_workflow():
    """Example 6: Complete training workflow demonstration."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Training Workflow")
    print("=" * 60)
    
    print("This example demonstrates the complete training workflow:")
    print("1. Data preprocessing")
    print("2. Model creation")
    print("3. Training loop")
    print("4. Evaluation")
    
    print("\nTo run the complete training workflow:")
    print("1. Add your data to data/raw/real/ and data/raw/fake/")
    print("2. Run preprocessing: python scripts/preprocess_data.py")
    print("3. Run training: python scripts/train.py")
    print("4. Make predictions: python scripts/predict.py --model models/best_model.pth --image your_image.jpg")


def example_7_prediction_workflow():
    """Example 7: Prediction workflow demonstration."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Prediction Workflow")
    print("=" * 60)
    
    print("This example demonstrates the prediction workflow:")
    print("1. Load trained model")
    print("2. Preprocess input data")
    print("3. Make predictions")
    print("4. Interpret results")
    
    print("\nTo run predictions:")
    print("1. Single image: python scripts/predict.py --model models/best_model.pth --image image.jpg")
    print("2. Video: python scripts/predict.py --model models/best_model.pth --video video.mp4")
    print("3. Directory: python scripts/predict.py --model models/best_model.pth --input_dir images/")
    
    print("\nExample prediction output:")
    print("  Prediction: fake")
    print("  Probability: 0.847")
    print("  Confidence: 0.847")


def main():
    """Run all examples."""
    print("DeepFake Detection System - Example Usage")
    print("=" * 60)
    print("This script demonstrates various components of the system.")
    print("Some examples may require data to be present.")
    print("=" * 60)
    
    # Run examples
    example_1_device_detection()
    example_2_face_extraction()
    example_3_dataset_usage()
    example_4_model_creation()
    example_5_model_factory()
    example_6_training_workflow()
    example_7_prediction_workflow()
    
    print("\n" + "=" * 60)
    print("EXAMPLES COMPLETED")
    print("=" * 60)
    print("For more detailed usage, see the README.md file and individual script files.")
    print("Each script includes comprehensive documentation and examples.")


if __name__ == "__main__":
    main() 