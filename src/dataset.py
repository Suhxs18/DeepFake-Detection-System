"""
Dataset module for DeepFake Detection System.

This module provides the DeepfakeDataset class for loading face images
and their corresponding labels (real/fake) for training and evaluation.
"""

import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import List, Tuple, Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeepfakeDataset(Dataset):
    """
    PyTorch Dataset class for loading face images and their labels.
    
    This dataset loads face images from organized directories and provides
    them with appropriate labels for binary classification (real vs fake).
    
    Attributes:
        image_paths (List[str]): List of paths to face images
        labels (List[int]): List of labels (0 for real, 1 for fake)
        transform (Optional[transforms.Compose]): Image transformations
        class_names (List[str]): Names of the classes ['real', 'fake']
    """
    
    def __init__(self, 
                 root_dir: str, 
                 transform: Optional[transforms.Compose] = None,
                 use_albumentations: bool = True):
        """
        Initialize the DeepfakeDataset.
        
        Args:
            root_dir (str): Root directory containing 'real' and 'fake' subdirectories
            transform (Optional[transforms.Compose]): PyTorch transforms to apply
            use_albumentations (bool): Whether to use Albumentations for augmentation
            
        Raises:
            FileNotFoundError: If root_dir doesn't exist
            ValueError: If real/fake directories are missing
            
        Example:
            dataset = DeepfakeDataset('data/processed_faces/train/')
        """
        self.root_dir = root_dir
        self.transform = transform
        self.use_albumentations = use_albumentations
        self.class_names = ['real', 'fake']
        
        # Validate root directory
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Root directory not found: {root_dir}")
        
        # Check for real and fake subdirectories
        real_dir = os.path.join(root_dir, 'real')
        fake_dir = os.path.join(root_dir, 'fake')
        
        if not os.path.exists(real_dir):
            raise ValueError(f"Real directory not found: {real_dir}")
        if not os.path.exists(fake_dir):
            raise ValueError(f"Fake directory not found: {fake_dir}")
        
        # Load image paths and labels
        self.image_paths, self.labels = self._load_data(real_dir, fake_dir)
        
        logger.info(f"Loaded {len(self.image_paths)} images from {root_dir}")
        logger.info(f"Real images: {sum(1 for label in self.labels if label == 0)}")
        logger.info(f"Fake images: {sum(1 for label in self.labels if label == 1)}")
    
    def _load_data(self, real_dir: str, fake_dir: str) -> Tuple[List[str], List[int]]:
        """
        Load image paths and labels from real and fake directories.
        
        Args:
            real_dir (str): Directory containing real face images
            fake_dir (str): Directory containing fake face images
            
        Returns:
            Tuple[List[str], List[int]]: Image paths and corresponding labels
        """
        image_paths = []
        labels = []
        
        # Load real images (label 0)
        real_images = self._get_image_files(real_dir)
        image_paths.extend(real_images)
        labels.extend([0] * len(real_images))
        
        # Load fake images (label 1)
        fake_images = self._get_image_files(fake_dir)
        image_paths.extend(fake_images)
        labels.extend([1] * len(fake_images))
        
        return image_paths, labels
    
    def _get_image_files(self, directory: str) -> List[str]:
        """
        Get all image files from a directory.
        
        Args:
            directory (str): Directory to search for images
            
        Returns:
            List[str]: List of full paths to image files
        """
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        for filename in os.listdir(directory):
            if os.path.splitext(filename)[1].lower() in image_extensions:
                image_files.append(os.path.join(directory, filename))
        
        return sorted(image_files)
    
    def __len__(self) -> int:
        """
        Return the number of images in the dataset.
        
        Returns:
            int: Number of images
        """
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single image and its label.
        
        Args:
            idx (int): Index of the image to retrieve
            
        Returns:
            Tuple[torch.Tensor, int]: Image tensor and label
            
        Raises:
            IndexError: If idx is out of range
            FileNotFoundError: If image file doesn't exist
        """
        if idx >= len(self.image_paths):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.image_paths)}")
        
        # Load image
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            # Load image using PIL
            image = Image.open(image_path).convert('RGB')
            
            # Apply transformations
            if self.transform is not None:
                if self.use_albumentations and isinstance(self.transform, A.Compose):
                    # Convert PIL to numpy for Albumentations
                    image_np = np.array(image)
                    transformed = self.transform(image=image_np)
                    image = transformed['image']  # Already a tensor
                else:
                    # Use PyTorch transforms
                    image = self.transform(image)
            
            return image, label
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            # Return a placeholder image and label
            placeholder = torch.zeros(3, 224, 224)
            return placeholder, label
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for handling imbalanced datasets.
        
        Returns:
            torch.Tensor: Class weights for loss function
            
        Example:
            class_weights = dataset.get_class_weights()
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        """
        class_counts = np.bincount(self.labels)
        total_samples = len(self.labels)
        
        # Calculate weights inversely proportional to class frequencies
        class_weights = total_samples / (len(class_counts) * class_counts)
        
        return torch.FloatTensor(class_weights)
    
    def get_class_distribution(self) -> Dict[str, int]:
        """
        Get the distribution of classes in the dataset.
        
        Returns:
            Dict[str, int]: Dictionary with class names and counts
            
        Example:
            distribution = dataset.get_class_distribution()
            print(f"Real: {distribution['real']}, Fake: {distribution['fake']}")
        """
        class_counts = np.bincount(self.labels)
        return {
            'real': int(class_counts[0]) if len(class_counts) > 0 else 0,
            'fake': int(class_counts[1]) if len(class_counts) > 1 else 0
        }


def get_transforms(image_size: int = 224, 
                  is_training: bool = True,
                  use_albumentations: bool = True) -> Any:
    """
    Get image transformations for training or validation.
    
    Args:
        image_size (int): Target size for images (assumes square)
        is_training (bool): Whether transforms are for training (includes augmentation)
        use_albumentations (bool): Whether to use Albumentations library
        
    Returns:
        Any: Transform pipeline (either Albumentations Compose or PyTorch transforms)
        
    Example:
        train_transforms = get_transforms(224, is_training=True)
        val_transforms = get_transforms(224, is_training=False)
    """
    if use_albumentations:
        return get_albumentations_transforms(image_size, is_training)
    else:
        return get_pytorch_transforms(image_size, is_training)


def get_albumentations_transforms(image_size: int, is_training: bool) -> A.Compose:
    """
    Get Albumentations transforms for image processing.
    
    Args:
        image_size (int): Target size for images
        is_training (bool): Whether transforms are for training
        
    Returns:
        A.Compose: Albumentations transform pipeline
    """
    if is_training:
        # Training transforms with augmentation
        transform = A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.RandomGamma(p=0.2),
            A.HueSaturationValue(p=0.2),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    else:
        # Validation transforms (no augmentation)
        transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    return transform


def get_pytorch_transforms(image_size: int, is_training: bool) -> transforms.Compose:
    """
    Get PyTorch transforms for image processing.
    
    Args:
        image_size (int): Target size for images
        is_training (bool): Whether transforms are for training
        
    Returns:
        transforms.Compose: PyTorch transform pipeline
    """
    if is_training:
        # Training transforms with augmentation
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        # Validation transforms (no augmentation)
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    return transform


def create_data_loaders(train_dir: str,
                       val_dir: str,
                       test_dir: Optional[str] = None,
                       batch_size: int = 32,
                       num_workers: int = 4,
                       image_size: int = 224,
                       use_albumentations: bool = True) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for training, validation, and testing.
    
    Args:
        train_dir (str): Directory containing training data
        val_dir (str): Directory containing validation data
        test_dir (Optional[str]): Directory containing test data
        batch_size (int): Batch size for DataLoaders
        num_workers (int): Number of worker processes for data loading
        image_size (int): Target size for images
        use_albumentations (bool): Whether to use Albumentations
        
    Returns:
        Dict[str, DataLoader]: Dictionary containing train, val, and test DataLoaders
        
    Example:
        loaders = create_data_loaders('data/train/', 'data/val/', 'data/test/')
        train_loader = loaders['train']
    """
    # Get transforms
    train_transforms = get_transforms(image_size, is_training=True, use_albumentations=use_albumentations)
    val_transforms = get_transforms(image_size, is_training=False, use_albumentations=use_albumentations)
    
    # Create datasets
    train_dataset = DeepfakeDataset(train_dir, transform=train_transforms, use_albumentations=use_albumentations)
    val_dataset = DeepfakeDataset(val_dir, transform=val_transforms, use_albumentations=use_albumentations)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    loaders = {
        'train': train_loader,
        'val': val_loader
    }
    
    # Add test loader if test directory provided
    if test_dir is not None:
        test_dataset = DeepfakeDataset(test_dir, transform=val_transforms, use_albumentations=use_albumentations)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        loaders['test'] = test_loader
    
    return loaders


if __name__ == '__main__':
    """
    Test script for dataset functionality.
    
    This block demonstrates how to use the dataset classes
    and can be used for testing during development.
    """
    print("Testing DeepFake Dataset")
    print("=" * 30)
    
    # Test dataset creation (this will fail if no data exists, but shows the structure)
    try:
        # Example paths (these would need to exist with actual data)
        test_train_dir = "data/processed_faces/train"
        test_val_dir = "data/processed_faces/val"
        
        if os.path.exists(test_train_dir):
            # Test dataset creation
            train_transforms = get_transforms(224, is_training=True)
            dataset = DeepfakeDataset(test_train_dir, transform=train_transforms)
            
            print(f"Dataset size: {len(dataset)}")
            print(f"Class distribution: {dataset.get_class_distribution()}")
            
            # Test getting a sample
            if len(dataset) > 0:
                image, label = dataset[0]
                print(f"Sample image shape: {image.shape}")
                print(f"Sample label: {label} ({dataset.class_names[label]})")
            
            # Test class weights
            class_weights = dataset.get_class_weights()
            print(f"Class weights: {class_weights}")
            
        else:
            print(f"Test directory {test_train_dir} does not exist.")
            print("Create the directory structure and add some face images to test.")
        
        # Test transform creation
        print("\nTesting transforms:")
        train_transforms = get_transforms(224, is_training=True)
        val_transforms = get_transforms(224, is_training=False)
        print(f"Training transforms: {type(train_transforms)}")
        print(f"Validation transforms: {type(val_transforms)}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        print("This is expected if no data exists yet.")
    
    print("\nDataset test completed!") 