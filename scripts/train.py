#!/usr/bin/env python3
"""
Training script for DeepFake Detection System.

This script implements a complete training pipeline including:
- Model training with validation
- Performance evaluation and metrics
- Model checkpointing and saving
- Training visualization and logging
"""

import os
import sys
import argparse
import logging
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import random
import numpy as np

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from utils import get_device
from dataset import DeepfakeDataset, get_transforms, create_data_loaders
from model import DeepfakeDetector, create_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DeepfakeTrainer:
    """
    Training class for deepfake detection models.
    
    This class handles the complete training pipeline including:
    - Data loading and preprocessing
    - Model training and validation
    - Performance evaluation
    - Model checkpointing
    - Visualization and logging
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the trainer.
        
        Args:
            config (Dict[str, Any]): Training configuration dictionary
        """
        self.config = config
        self.device = get_device()
        
        # Set random seeds for reproducibility
        self._set_random_seeds()
        
        # Initialize components
        self._initialize_data_loaders()
        self._initialize_model()
        self._initialize_training_components()
        self._initialize_logging()
        
        logger.info("DeepfakeTrainer initialized successfully")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Model: {config['model_config']['backbone']}")
        logger.info(f"Batch size: {config['batch_size']}")
        logger.info(f"Learning rate: {config['learning_rate']}")
    
    def _set_random_seeds(self):
        """Set random seeds for reproducibility."""
        random.seed(self.config['random_seed'])
        np.random.seed(self.config['random_seed'])
        torch.manual_seed(self.config['random_seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config['random_seed'])
            torch.cuda.manual_seed_all(self.config['random_seed'])
    
    def _initialize_data_loaders(self):
        """Initialize data loaders for training, validation, and testing."""
        logger.info("Initializing data loaders...")
        
        # Get transforms
        train_transforms = get_transforms(
            image_size=self.config['image_size'],
            is_training=True,
            use_albumentations=self.config['use_albumentations']
        )
        val_transforms = get_transforms(
            image_size=self.config['image_size'],
            is_training=False,
            use_albumentations=self.config['use_albumentations']
        )
        
        # Create datasets
        self.train_dataset = DeepfakeDataset(
            self.config['train_dir'],
            transform=train_transforms,
            use_albumentations=self.config['use_albumentations']
        )
        
        self.val_dataset = DeepfakeDataset(
            self.config['val_dir'],
            transform=val_transforms,
            use_albumentations=self.config['use_albumentations']
        )
        
        if self.config['test_dir']:
            self.test_dataset = DeepfakeDataset(
                self.config['test_dir'],
                transform=val_transforms,
                use_albumentations=self.config['use_albumentations']
            )
        else:
            self.test_dataset = None
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )
        
        if self.test_dataset:
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=self.config['num_workers'],
                pin_memory=True
            )
        else:
            self.test_loader = None
        
        logger.info(f"Train samples: {len(self.train_dataset)}")
        logger.info(f"Validation samples: {len(self.val_dataset)}")
        if self.test_dataset:
            logger.info(f"Test samples: {len(self.test_dataset)}")
    
    def _initialize_model(self):
        """Initialize the model."""
        logger.info("Initializing model...")
        
        # Create model
        self.model = create_model(self.config['model_config'])
        self.model = self.model.to(self.device)
        
        # Log model information
        model_info = self.model.get_model_info()
        logger.info(f"Model parameters: {model_info['total_parameters']:,}")
        logger.info(f"Trainable parameters: {model_info['trainable_parameters']:,}")
    
    def _initialize_training_components(self):
        """Initialize optimizer, loss function, and scheduler."""
        # Optimizer
        if self.config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        elif self.config['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                momentum=0.9,
                weight_decay=self.config['weight_decay']
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config['optimizer']}")
        
        # Loss function
        if self.config['model_config']['num_classes'] == 1:
            # Binary classification
            if self.config['use_class_weights']:
                try:
                    class_weights = self.train_dataset.get_class_weights().to(self.device)
                    self.criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
                except:
                    logger.warning("Could not calculate class weights, using default BCEWithLogitsLoss")
                    self.criterion = nn.BCEWithLogitsLoss()
            else:
                self.criterion = nn.BCEWithLogitsLoss()
        else:
            # Multi-class classification
            if self.config['use_class_weights']:
                try:
                    class_weights = self.train_dataset.get_class_weights().to(self.device)
                    self.criterion = nn.CrossEntropyLoss(weight=class_weights)
                except:
                    logger.warning("Could not calculate class weights, using default CrossEntropyLoss")
                    self.criterion = nn.CrossEntropyLoss()
            else:
                self.criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler
        if self.config['scheduler'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config['scheduler_step_size'],
                gamma=self.config['scheduler_gamma']
            )
        elif self.config['scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['epochs']
            )
        else:
            self.scheduler = None
        
        logger.info(f"Optimizer: {self.config['optimizer']}")
        logger.info(f"Loss function: {type(self.criterion).__name__}")
        if self.scheduler:
            logger.info(f"Scheduler: {self.config['scheduler']}")
    
    def _initialize_logging(self):
        """Initialize TensorBoard logging."""
        self.writer = SummaryWriter(log_dir=self.config['log_dir'])
        
        # Log model graph
        dummy_input = torch.randn(1, 3, self.config['image_size'], self.config['image_size']).to(self.device)
        self.writer.add_graph(self.model, dummy_input)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch (int): Current epoch number
            
        Returns:
            Dict[str, float]: Training metrics for the epoch
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Progress tracking
        num_batches = len(self.train_loader)
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            # Move data to device
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Calculate loss
            if self.config['model_config']['num_classes'] == 1:
                # Binary classification
                labels = labels.float().unsqueeze(1)
                loss = self.criterion(outputs, labels)
                predictions = (torch.sigmoid(outputs) > 0.5).long().squeeze()
                labels = labels.squeeze().long()
            else:
                # Multi-class classification
                loss = self.criterion(outputs, labels)
                predictions = torch.argmax(outputs, dim=1)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            # Log progress
            if batch_idx % self.config['log_interval'] == 0:
                logger.info(f'Epoch {epoch}: [{batch_idx}/{num_batches}] '
                          f'Loss: {loss.item():.4f}, '
                          f'Acc: {100. * correct / total:.2f}%')
        
        # Calculate epoch metrics
        avg_loss = total_loss / num_batches
        accuracy = 100. * correct / total
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Args:
            epoch (int): Current epoch number
            
        Returns:
            Dict[str, float]: Validation metrics for the epoch
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                # Move data to device
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Calculate loss
                if self.config['model_config']['num_classes'] == 1:
                    # Binary classification
                    labels_float = labels.float().unsqueeze(1)
                    loss = self.criterion(outputs, labels_float)
                    probabilities = torch.sigmoid(outputs).squeeze()
                    predictions = (probabilities > 0.5).long()
                else:
                    # Multi-class classification
                    loss = self.criterion(outputs, labels)
                    probabilities = torch.softmax(outputs, dim=1)
                    predictions = torch.argmax(outputs, dim=1)
                
                # Update metrics
                total_loss += loss.item()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                
                # Store predictions and labels for detailed metrics
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                if self.config['model_config']['num_classes'] == 1:
                    all_probabilities.extend(probabilities.cpu().numpy())
                else:
                    all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # Probability of fake class
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        # Calculate additional metrics
        precision = precision_score(all_labels, all_predictions, average='binary')
        recall = recall_score(all_labels, all_predictions, average='binary')
        f1 = f1_score(all_labels, all_predictions, average='binary')
        auc = roc_auc_score(all_labels, all_probabilities)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        
        best_val_accuracy = 0.0
        best_val_f1 = 0.0
        training_history = {
            'train_loss': [], 'train_accuracy': [],
            'val_loss': [], 'val_accuracy': [], 'val_f1': [], 'val_auc': []
        }
        
        for epoch in range(1, self.config['epochs'] + 1):
            start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate_epoch(epoch)
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Log metrics
            epoch_time = time.time() - start_time
            logger.info(f'Epoch {epoch}/{self.config["epochs"]} ({epoch_time:.2f}s): '
                       f'Train Loss: {train_metrics["loss"]:.4f}, '
                       f'Train Acc: {train_metrics["accuracy"]:.2f}%, '
                       f'Val Loss: {val_metrics["loss"]:.4f}, '
                       f'Val Acc: {val_metrics["accuracy"]:.2f}%, '
                       f'Val F1: {val_metrics["f1"]:.4f}, '
                       f'Val AUC: {val_metrics["auc"]:.4f}')
            
            # Log to TensorBoard
            self.writer.add_scalar('Loss/Train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/Validation', val_metrics['loss'], epoch)
            self.writer.add_scalar('Accuracy/Train', train_metrics['accuracy'], epoch)
            self.writer.add_scalar('Accuracy/Validation', val_metrics['accuracy'], epoch)
            self.writer.add_scalar('F1/Validation', val_metrics['f1'], epoch)
            self.writer.add_scalar('AUC/Validation', val_metrics['auc'], epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Store history
            training_history['train_loss'].append(train_metrics['loss'])
            training_history['train_accuracy'].append(train_metrics['accuracy'])
            training_history['val_loss'].append(val_metrics['loss'])
            training_history['val_accuracy'].append(val_metrics['accuracy'])
            training_history['val_f1'].append(val_metrics['f1'])
            training_history['val_auc'].append(val_metrics['auc'])
            
            # Save best model
            if val_metrics['accuracy'] > best_val_accuracy:
                best_val_accuracy = val_metrics['accuracy']
                self._save_checkpoint(epoch, val_metrics, is_best=True)
                logger.info(f"New best model saved with validation accuracy: {best_val_accuracy:.2f}%")
            
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                self._save_checkpoint(epoch, val_metrics, is_best_f1=True)
                logger.info(f"New best F1 model saved with validation F1: {best_val_f1:.4f}")
            
            # Save regular checkpoint
            if epoch % self.config['save_interval'] == 0:
                self._save_checkpoint(epoch, val_metrics, is_best=False)
        
        # Training completed
        logger.info("Training completed!")
        self.writer.close()
        
        # Plot training history
        self._plot_training_history(training_history)
        
        # Evaluate on test set if available
        if self.test_loader:
            self._evaluate_test_set()
    
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False, is_best_f1: bool = False):
        """
        Save model checkpoint.
        
        Args:
            epoch (int): Current epoch
            metrics (Dict[str, float]): Validation metrics
            is_best (bool): Whether this is the best accuracy model
            is_best_f1 (bool): Whether this is the best F1 model
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.config['checkpoint_dir'], f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config['checkpoint_dir'], 'best_model.pth')
            torch.save(checkpoint, best_path)
        
        if is_best_f1:
            best_f1_path = os.path.join(self.config['checkpoint_dir'], 'best_f1_model.pth')
            torch.save(checkpoint, best_f1_path)
    
    def _plot_training_history(self, history: Dict[str, List[float]]):
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(history['train_loss'], label='Train Loss')
        axes[0, 0].plot(history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(history['train_accuracy'], label='Train Accuracy')
        axes[0, 1].plot(history['val_accuracy'], label='Validation Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1 score plot
        axes[1, 0].plot(history['val_f1'], label='Validation F1', color='green')
        axes[1, 0].set_title('Validation F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # AUC plot
        axes[1, 1].plot(history['val_auc'], label='Validation AUC', color='red')
        axes[1, 1].set_title('Validation AUC')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('AUC')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(self.config['log_dir'], 'training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training history plot saved to: {plot_path}")
    
    def _evaluate_test_set(self):
        """Evaluate the model on the test set."""
        logger.info("Evaluating on test set...")
        
        # Load best model
        best_model_path = os.path.join(self.config['checkpoint_dir'], 'best_model.pth')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded best model from epoch {checkpoint['epoch']}")
        
        # Evaluate
        test_metrics = self.validate_epoch(0)  # Use 0 as epoch for test
        
        # Calculate confusion matrix
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                
                if self.config['model_config']['num_classes'] == 1:
                    predictions = (torch.sigmoid(outputs) > 0.5).long().squeeze()
                else:
                    predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Real', 'Fake'],
                   yticklabels=['Real', 'Fake'])
        plt.title('Confusion Matrix (Test Set)')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        cm_path = os.path.join(self.config['log_dir'], 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print test results
        logger.info("=" * 50)
        logger.info("TEST SET EVALUATION")
        logger.info("=" * 50)
        logger.info(f"Accuracy: {test_metrics['accuracy']:.2f}%")
        logger.info(f"Precision: {test_metrics['precision']:.4f}")
        logger.info(f"Recall: {test_metrics['recall']:.4f}")
        logger.info(f"F1 Score: {test_metrics['f1']:.4f}")
        logger.info(f"AUC: {test_metrics['auc']:.4f}")
        logger.info(f"Confusion Matrix:\n{cm}")
        logger.info(f"Confusion matrix plot saved to: {cm_path}")
        
        # Save test results
        test_results = {
            'accuracy': test_metrics['accuracy'],
            'precision': test_metrics['precision'],
            'recall': test_metrics['recall'],
            'f1_score': test_metrics['f1'],
            'auc': test_metrics['auc'],
            'confusion_matrix': cm.tolist()
        }
        
        results_path = os.path.join(self.config['log_dir'], 'test_results.json')
        with open(results_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        logger.info(f"Test results saved to: {results_path}")


def get_default_config() -> Dict[str, Any]:
    """
    Get default training configuration.
    
    Returns:
        Dict[str, Any]: Default configuration dictionary
    """
    return {
        # Data configuration
        'train_dir': 'data/processed_faces/train',
        'val_dir': 'data/processed_faces/val',
        'test_dir': 'data/processed_faces/test',
        'image_size': 224,
        'batch_size': 32,
        'num_workers': 4,
        'use_albumentations': True,
        
        # Model configuration
        'model_config': {
            'model_type': 'deepfake_detector',
            'num_classes': 1,
            'dropout_rate': 0.5,
            'backbone': 'resnet18',
            'pretrained': True,
            'freeze_backbone': False
        },
        
        # Training configuration
        'epochs': 50,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'optimizer': 'adam',
        'scheduler': 'step',
        'scheduler_step_size': 20,
        'scheduler_gamma': 0.1,
        'use_class_weights': True,
        
        # Logging and saving
        'log_dir': 'logs',
        'checkpoint_dir': 'models',
        'save_interval': 10,
        'log_interval': 10,
        
        # Reproducibility
        'random_seed': 42
    }


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train deepfake detection model')
    
    # Data arguments
    parser.add_argument('--train_dir', type=str, default='data/processed_faces/train',
                       help='Training data directory')
    parser.add_argument('--val_dir', type=str, default='data/processed_faces/val',
                       help='Validation data directory')
    parser.add_argument('--test_dir', type=str, default='data/processed_faces/test',
                       help='Test data directory')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Input image size')
    
    # Model arguments
    parser.add_argument('--backbone', type=str, default='resnet18',
                       choices=['resnet18', 'resnet34', 'resnet50'],
                       help='Backbone architecture')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                       help='Dropout rate')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pre-trained weights')
    parser.add_argument('--freeze_backbone', action='store_true',
                       help='Freeze backbone layers')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'sgd'],
                       help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='step',
                       choices=['step', 'cosine', 'none'],
                       help='Learning rate scheduler')
    
    # Logging arguments
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Logging directory')
    parser.add_argument('--checkpoint_dir', type=str, default='models',
                       help='Checkpoint directory')
    
    return parser.parse_args()


def main():
    """Main function to run the training pipeline."""
    # Parse arguments
    args = parse_arguments()
    
    # Create configuration
    config = get_default_config()
    
    # Update config with command line arguments
    config.update({
        'train_dir': args.train_dir,
        'val_dir': args.val_dir,
        'test_dir': args.test_dir,
        'batch_size': args.batch_size,
        'image_size': args.image_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'optimizer': args.optimizer,
        'scheduler': args.scheduler if args.scheduler != 'none' else None,
        'log_dir': args.log_dir,
        'checkpoint_dir': args.checkpoint_dir
    })
    
    config['model_config'].update({
        'backbone': args.backbone,
        'dropout_rate': args.dropout_rate,
        'pretrained': args.pretrained,
        'freeze_backbone': args.freeze_backbone
    })
    
    # Create directories
    os.makedirs(config['log_dir'], exist_ok=True)
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # Validate data directories
    if not os.path.exists(config['train_dir']):
        logger.error(f"Training directory not found: {config['train_dir']}")
        sys.exit(1)
    
    if not os.path.exists(config['val_dir']):
        logger.error(f"Validation directory not found: {config['val_dir']}")
        sys.exit(1)
    
    try:
        # Initialize trainer
        trainer = DeepfakeTrainer(config)
        
        # Start training
        trainer.train()
        
        logger.info("Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main() 