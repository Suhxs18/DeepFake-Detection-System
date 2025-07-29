#!/usr/bin/env python3
"""
Demo Training Script for DeepFake Detection System
This script demonstrates the training pipeline with the current data.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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

class SimpleDataset(torch.utils.data.Dataset):
    """Simple dataset for demo purposes"""
    def __init__(self, fake_dir, real_dir=None):
        self.fake_dir = fake_dir
        self.real_dir = real_dir
        self.data = []
        self.labels = []
        
        # Load fake images
        if os.path.exists(fake_dir):
            for img_name in os.listdir(fake_dir):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    self.data.append(os.path.join(fake_dir, img_name))
                    self.labels.append(1)  # Fake = 1
        
        # Load real images (if available)
        if real_dir and os.path.exists(real_dir):
            for img_name in os.listdir(real_dir):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    self.data.append(os.path.join(real_dir, img_name))
                    self.labels.append(0)  # Real = 0
        
        logger.info(f"Loaded {len(self.data)} images")
        logger.info(f"Fake images: {sum(self.labels)}")
        logger.info(f"Real images: {len(self.labels) - sum(self.labels)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        
        # Load and preprocess image
        img = Image.open(img_path).convert('RGB')
        img = img.resize((224, 224))
        
        # Convert to tensor
        img_tensor = torch.FloatTensor(np.array(img)) / 255.0
        img_tensor = img_tensor.permute(2, 0, 1)  # HWC to CHW
        
        return img_tensor, label

def demo_train():
    """Demo training function"""
    logger.info("Starting Demo Training...")
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Create dataset
    fake_dir = "data/processed_faces/train/fake"
    real_dir = "data/processed_faces/train/real"
    
    dataset = SimpleDataset(fake_dir, real_dir)
    
    if len(dataset) == 0:
        logger.error("No data found! Please run preprocessing first.")
        return
    
    # Create data loader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Create model
    model = DeepfakeDetector(num_classes=1, dropout_rate=0.5)
    model = model.to(device)
    
    # Setup training components
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Training samples: {len(dataset)}")
    
    # Training loop
    model.train()
    for epoch in range(3):
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()
            
            logger.info(f'Epoch {epoch+1}: Batch {batch_idx+1}/{len(dataloader)}, '
                       f'Loss: {loss.item():.4f}, '
                       f'Acc: {100. * correct / total:.2f}%')
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        logger.info(f'Epoch {epoch+1} Summary: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    # Save model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/demo_model.pth')
    logger.info("Model saved to models/demo_model.pth")
    
    logger.info("Demo training completed successfully!")

if __name__ == "__main__":
    demo_train() 