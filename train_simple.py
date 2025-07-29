#!/usr/bin/env python3
"""
Simplified training script for DeepFake Detection System.
This version works with the current dataset and avoids complex class weight calculations.
"""

import os
import sys
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils import get_device
from src.dataset import DeepfakeDataset, get_transforms
from src.model import DeepfakeDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleTrainer:
    """Simplified trainer for DeepFake detection."""
    
    def __init__(self):
        self.device = get_device()
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
    def setup_data(self):
        """Setup data loaders."""
        logger.info("Setting up data loaders...")
        
        # Get transforms
        train_transforms = get_transforms(224, is_training=True)
        val_transforms = get_transforms(224, is_training=False)
        
        # Create datasets
        train_dataset = DeepfakeDataset('data/processed_faces/train', train_transforms)
        val_dataset = DeepfakeDataset('data/processed_faces/val', val_transforms)
        test_dataset = DeepfakeDataset('data/processed_faces/test', val_transforms)
        
        # Create data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
        
        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Val samples: {len(val_dataset)}")
        logger.info(f"Test samples: {len(test_dataset)}")
        
    def setup_model(self):
        """Setup model, optimizer, and loss function."""
        logger.info("Setting up model...")
        
        # Create model
        self.model = DeepfakeDetector(num_classes=1, dropout_rate=0.5)
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Loss function (simple BCE without class weights)
        self.criterion = nn.BCEWithLogitsLoss()
        
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.float().unsqueeze(1).to(self.device)  # Shape: [batch_size, 1]
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)  # Shape: [batch_size, 1]
            
            # Calculate loss
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Calculate accuracy
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()
            
            if batch_idx % 2 == 0:
                logger.info(f"Epoch {epoch}: Batch {batch_idx}/{len(self.train_loader)}, "
                          f"Loss: {loss.item():.4f}, Acc: {100 * correct / total:.2f}%")
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100 * correct / total
        logger.info(f"Epoch {epoch} Summary: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        return avg_loss, accuracy
    
    def validate_epoch(self, epoch):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.float().unsqueeze(1).to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100 * correct / total
        logger.info(f"Validation Epoch {epoch}: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        return avg_loss, accuracy
    
    def test_model(self):
        """Test the model on test set."""
        logger.info("Testing model on test set...")
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                predictions = (torch.sigmoid(outputs) > 0.5).float().squeeze()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='binary')
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        
        logger.info("=" * 50)
        logger.info("TEST RESULTS:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1-Score: {f1:.4f}")
        logger.info(f"Confusion Matrix:\n{conf_matrix}")
        logger.info("=" * 50)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': conf_matrix
        }
    
    def train(self, epochs=10):
        """Main training loop."""
        logger.info("Starting training...")
        
        best_val_acc = 0.0
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        
        for epoch in range(1, epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
            # Validate
            val_loss, val_acc = self.validate_epoch(epoch)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'models/best_model.pth')
                logger.info(f"New best model saved with validation accuracy: {best_val_acc:.2f}%")
        
        # Test final model
        self.test_model()
        
        # Plot training history
        self.plot_training_history(train_losses, train_accs, val_losses, val_accs)
        
        logger.info("Training completed!")
    
    def plot_training_history(self, train_losses, train_accs, val_losses, val_accs):
        """Plot training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot losses
        ax1.plot(train_losses, label='Train Loss')
        ax1.plot(val_losses, label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Plot accuracies
        ax2.plot(train_accs, label='Train Acc')
        ax2.plot(val_accs, label='Val Acc')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        logger.info("Training history plot saved as 'training_history.png'")


def main():
    """Main function."""
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Initialize trainer
    trainer = SimpleTrainer()
    
    # Setup data and model
    trainer.setup_data()
    trainer.setup_model()
    
    # Train
    trainer.train(epochs=10)


if __name__ == '__main__':
    main() 