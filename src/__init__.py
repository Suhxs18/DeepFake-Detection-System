"""
DeepFake Detection System - Source Package

This package contains the core modules for the deepfake detection system:
- utils: Utility functions for device detection, face extraction, and video processing
- dataset: Dataset classes and data loading utilities
- model: Neural network models for deepfake detection
"""

from .utils import (
    get_device,
    extract_face,
    video_to_frames_and_faces,
    validate_image_file,
    validate_video_file,
    is_image_file,
    is_video_file
)

from .dataset import (
    DeepfakeDataset,
    get_transforms,
    create_data_loaders
)

from .model import (
    DeepfakeDetector,
    DeepfakeDetectorWithAttention,
    AttentionModule,
    create_model
)

__version__ = "1.0.0"
__author__ = "DeepFake Detection System"
__description__ = "A comprehensive deep learning system for detecting manipulated/synthetic media"

__all__ = [
    # Utils
    'get_device',
    'extract_face',
    'video_to_frames_and_faces',
    'validate_image_file',
    'validate_video_file',
    'is_image_file',
    'is_video_file',
    
    # Dataset
    'DeepfakeDataset',
    'get_transforms',
    'create_data_loaders',
    
    # Model
    'DeepfakeDetector',
    'DeepfakeDetectorWithAttention',
    'AttentionModule',
    'create_model'
] 