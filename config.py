"""
Configuration file for DeepFake Detection System.

This file contains all the configuration settings for the system,
making it easy to customize parameters without modifying the source code.
"""

import os
from typing import Dict, Any, Tuple

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

DATA_CONFIG = {
    # Directory paths
    'raw_data_dir': 'data/raw',
    'processed_faces_dir': 'data/processed_faces',
    
    # Face extraction settings
    'face_size': (224, 224),
    'face_margin': 20,
    'frame_interval': 30,  # Extract every 30th frame from videos
    
    # Data splits
    'train_split': 0.7,    # 70% for training
    'val_split': 0.15,     # 15% for validation
    'test_split': 0.15,    # 15% for testing
    
    # Image processing
    'image_size': 224,
    'use_albumentations': True,
    
    # Data loading
    'batch_size': 32,
    'num_workers': 4,
    'pin_memory': True
}

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

MODEL_CONFIG = {
    # Model architecture
    'model_type': 'deepfake_detector',  # 'deepfake_detector' or 'deepfake_detector_attention'
    'num_classes': 1,  # 1 for binary classification (BCE), 2 for CrossEntropy
    'backbone': 'resnet18',  # 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
    'dropout_rate': 0.5,
    'pretrained': True,
    'freeze_backbone': False,
    
    # Attention mechanism (for attention model)
    'use_attention': True,
    'attention_reduction': 16
}

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

TRAINING_CONFIG = {
    # Training parameters
    'epochs': 50,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'optimizer': 'adam',  # 'adam' or 'sgd'
    'scheduler': 'step',  # 'step', 'cosine', or None
    
    # Learning rate scheduler
    'scheduler_step_size': 20,
    'scheduler_gamma': 0.1,
    
    # Loss function
    'use_class_weights': True,
    
    # Logging and saving
    'log_dir': 'logs',
    'checkpoint_dir': 'models',
    'save_interval': 10,
    'log_interval': 10,
    
    # Reproducibility
    'random_seed': 42
}

# =============================================================================
# INFERENCE CONFIGURATION
# =============================================================================

INFERENCE_CONFIG = {
    # Model loading
    'default_model_path': 'models/best_model.pth',
    
    # Prediction settings
    'confidence_threshold': 0.5,
    'frame_interval': 30,  # For video processing
    
    # Output settings
    'save_predictions': True,
    'output_format': 'json'  # 'json' or 'csv'
}

# =============================================================================
# SYSTEM CONFIGURATION
# =============================================================================

SYSTEM_CONFIG = {
    # Device settings
    'force_cpu': False,  # Force CPU usage even if GPU is available
    
    # Memory settings
    'max_memory_usage': 0.8,  # Maximum GPU memory usage (0.0 to 1.0)
    
    # Logging
    'log_level': 'INFO',  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    'log_to_file': True,
    'log_to_console': True,
    
    # Performance
    'num_threads': 4,  # Number of CPU threads for data loading
    'prefetch_factor': 2  # Prefetch factor for data loading
}

# =============================================================================
# FILE EXTENSIONS
# =============================================================================

SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif'}
SUPPORTED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}

# =============================================================================
# CONFIGURATION UTILITIES
# =============================================================================

def get_full_config() -> Dict[str, Any]:
    """
    Get the complete configuration dictionary.
    
    Returns:
        Dict[str, Any]: Complete configuration
    """
    return {
        'data': DATA_CONFIG,
        'model': MODEL_CONFIG,
        'training': TRAINING_CONFIG,
        'inference': INFERENCE_CONFIG,
        'system': SYSTEM_CONFIG
    }

def update_config(config_name: str, updates: Dict[str, Any]) -> None:
    """
    Update a specific configuration section.
    
    Args:
        config_name (str): Name of the configuration section
        updates (Dict[str, Any]): Updates to apply
    """
    if config_name == 'data':
        DATA_CONFIG.update(updates)
    elif config_name == 'model':
        MODEL_CONFIG.update(updates)
    elif config_name == 'training':
        TRAINING_CONFIG.update(updates)
    elif config_name == 'inference':
        INFERENCE_CONFIG.update(updates)
    elif config_name == 'system':
        SYSTEM_CONFIG.update(updates)
    else:
        raise ValueError(f"Unknown configuration section: {config_name}")

def create_directories() -> None:
    """Create necessary directories for the project."""
    directories = [
        DATA_CONFIG['raw_data_dir'],
        DATA_CONFIG['processed_faces_dir'],
        TRAINING_CONFIG['log_dir'],
        TRAINING_CONFIG['checkpoint_dir'],
        'data/raw/real',
        'data/raw/fake',
        'data/processed_faces/train/real',
        'data/processed_faces/train/fake',
        'data/processed_faces/val/real',
        'data/processed_faces/val/fake',
        'data/processed_faces/test/real',
        'data/processed_faces/test/fake'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def validate_config() -> bool:
    """
    Validate the configuration settings.
    
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    # Validate data splits
    total_split = DATA_CONFIG['train_split'] + DATA_CONFIG['val_split'] + DATA_CONFIG['test_split']
    if abs(total_split - 1.0) > 1e-6:
        print(f"Error: Data splits must sum to 1.0, got {total_split}")
        return False
    
    # Validate model configuration
    if MODEL_CONFIG['num_classes'] not in [1, 2]:
        print(f"Error: num_classes must be 1 or 2, got {MODEL_CONFIG['num_classes']}")
        return False
    
    # Validate training configuration
    if TRAINING_CONFIG['epochs'] <= 0:
        print(f"Error: epochs must be positive, got {TRAINING_CONFIG['epochs']}")
        return False
    
    if TRAINING_CONFIG['learning_rate'] <= 0:
        print(f"Error: learning_rate must be positive, got {TRAINING_CONFIG['learning_rate']}")
        return False
    
    return True

# =============================================================================
# EXAMPLE CONFIGURATIONS
# =============================================================================

# Example configuration for quick experimentation
QUICK_EXPERIMENT_CONFIG = {
    'data': {
        'batch_size': 16,
        'epochs': 10,
        'learning_rate': 0.01
    },
    'training': {
        'epochs': 10,
        'save_interval': 5
    }
}

# Example configuration for production use
PRODUCTION_CONFIG = {
    'data': {
        'batch_size': 64,
        'num_workers': 8
    },
    'model': {
        'backbone': 'resnet50',
        'pretrained': True
    },
    'training': {
        'epochs': 100,
        'learning_rate': 0.0001,
        'weight_decay': 1e-5
    },
    'system': {
        'log_level': 'WARNING'
    }
}

# =============================================================================
# CONFIGURATION PRESETS
# =============================================================================

def apply_preset(preset_name: str) -> None:
    """
    Apply a configuration preset.
    
    Args:
        preset_name (str): Name of the preset ('quick', 'production')
    """
    if preset_name == 'quick':
        for section, updates in QUICK_EXPERIMENT_CONFIG.items():
            update_config(section, updates)
    elif preset_name == 'production':
        for section, updates in PRODUCTION_CONFIG.items():
            update_config(section, updates)
    else:
        raise ValueError(f"Unknown preset: {preset_name}")

if __name__ == '__main__':
    """Test configuration validation."""
    print("Validating configuration...")
    if validate_config():
        print("Configuration is valid!")
    else:
        print("Configuration has errors!")
    
    print("\nCreating directories...")
    create_directories()
    print("Directories created!")
    
    print("\nFull configuration:")
    import json
    print(json.dumps(get_full_config(), indent=2)) 