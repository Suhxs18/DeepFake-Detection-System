"""
Model module for DeepFake Detection System.

This module provides the DeepfakeDetector neural network model for
binary classification of face images (real vs fake).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, Tuple, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeepfakeDetector(nn.Module):
    """
    Deep learning model for deepfake detection.
    
    This model uses a pre-trained ResNet18 backbone with a custom
    classification head for binary classification of face images.
    
    Attributes:
        backbone (nn.Module): Pre-trained ResNet18 feature extractor
        classifier (nn.Sequential): Custom classification head
        dropout_rate (float): Dropout rate for regularization
        num_classes (int): Number of output classes (2 for binary)
    """
    
    def __init__(self, 
                 num_classes: int = 1, 
                 dropout_rate: float = 0.5,
                 backbone: str = 'resnet18',
                 pretrained: bool = True,
                 freeze_backbone: bool = False):
        """
        Initialize the DeepfakeDetector model.
        
        Args:
            num_classes (int): Number of output classes (1 for binary with BCE, 2 for CrossEntropy)
            dropout_rate (float): Dropout rate for regularization
            backbone (str): Backbone architecture ('resnet18', 'resnet34', 'resnet50')
            pretrained (bool): Whether to use pre-trained weights
            freeze_backbone (bool): Whether to freeze backbone layers during training
            
        Example:
            model = DeepfakeDetector(num_classes=1, dropout_rate=0.5)
        """
        super(DeepfakeDetector, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.backbone_name = backbone
        self.pretrained = pretrained
        
        # Initialize backbone
        self.backbone = self._create_backbone(backbone, pretrained)
        
        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()
        
        # Get the number of features from the backbone
        if backbone in ['resnet18', 'resnet34']:
            num_features = 512
        elif backbone in ['resnet50', 'resnet101', 'resnet152']:
            num_features = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Create classification head
        self.classifier = self._create_classifier(num_features, num_classes, dropout_rate)
        
        logger.info(f"Initialized DeepfakeDetector with {backbone} backbone")
        logger.info(f"Number of features: {num_features}, Classes: {num_classes}")
        logger.info(f"Dropout rate: {dropout_rate}, Pretrained: {pretrained}")
    
    def _create_backbone(self, backbone: str, pretrained: bool) -> nn.Module:
        """
        Create the backbone network.
        
        Args:
            backbone (str): Backbone architecture name
            pretrained (bool): Whether to use pre-trained weights
            
        Returns:
            nn.Module: Backbone network
            
        Raises:
            ValueError: If backbone architecture is not supported
        """
        if backbone == 'resnet18':
            model = models.resnet18(pretrained=pretrained)
        elif backbone == 'resnet34':
            model = models.resnet34(pretrained=pretrained)
        elif backbone == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
        elif backbone == 'resnet101':
            model = models.resnet101(pretrained=pretrained)
        elif backbone == 'resnet152':
            model = models.resnet152(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove the final classification layer
        backbone = nn.Sequential(*list(model.children())[:-1])
        
        return backbone
    
    def _create_classifier(self, num_features: int, num_classes: int, dropout_rate: float) -> nn.Sequential:
        """
        Create the classification head.
        
        Args:
            num_features (int): Number of input features from backbone
            num_classes (int): Number of output classes
            dropout_rate (float): Dropout rate for regularization
            
        Returns:
            nn.Sequential: Classification head
        """
        classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
            nn.Flatten(),  # Flatten the features
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        return classifier
    
    def _freeze_backbone(self):
        """
        Freeze the backbone layers to prevent gradient updates.
        
        This is useful for transfer learning when you want to
        only train the classification head initially.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone layers frozen")
    
    def unfreeze_backbone(self):
        """
        Unfreeze the backbone layers to allow fine-tuning.
        
        This can be called after initial training of the classifier
        to fine-tune the entire network.
        """
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Backbone layers unfrozen for fine-tuning")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
            
        Example:
            model = DeepfakeDetector()
            output = model(input_tensor)
        """
        # Extract features using backbone
        features = self.backbone(x)
        
        # Apply classification head
        logits = self.classifier(features)
        
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the backbone (without classification).
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Feature tensor from backbone
        """
        features = self.backbone(x)
        return features
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get probability predictions.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Probability predictions
        """
        with torch.no_grad():
            logits = self.forward(x)
            if self.num_classes == 1:
                # Binary classification with sigmoid
                probs = torch.sigmoid(logits)
            else:
                # Multi-class classification with softmax
                probs = F.softmax(logits, dim=1)
        
        return probs
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class predictions.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Class predictions
        """
        with torch.no_grad():
            logits = self.forward(x)
            if self.num_classes == 1:
                # Binary classification
                predictions = (torch.sigmoid(logits) > 0.5).long()
            else:
                # Multi-class classification
                predictions = torch.argmax(logits, dim=1)
        
        return predictions
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model architecture.
        
        Returns:
            Dict[str, Any]: Model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = {
            'backbone': self.backbone_name,
            'num_classes': self.num_classes,
            'dropout_rate': self.dropout_rate,
            'pretrained': self.pretrained,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': total_params - trainable_params
        }
        
        return info


class AttentionModule(nn.Module):
    """
    Attention module for focusing on important regions in face images.
    
    This module can be added to the model to improve performance
    by learning to focus on regions that are most indicative of deepfakes.
    """
    
    def __init__(self, in_channels: int, reduction: int = 16):
        """
        Initialize the attention module.
        
        Args:
            in_channels (int): Number of input channels
            reduction (int): Reduction factor for the attention mechanism
        """
        super(AttentionModule, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply attention mechanism.
        
        Args:
            x (torch.Tensor): Input feature tensor
            
        Returns:
            torch.Tensor: Attention-weighted features
        """
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        attention = self.sigmoid(avg_out + max_out).unsqueeze(2).unsqueeze(3)
        
        return x * attention.expand_as(x)


class DeepfakeDetectorWithAttention(DeepfakeDetector):
    """
    Enhanced DeepfakeDetector with attention mechanism.
    
    This model adds attention modules to focus on important regions
    in face images that are most indicative of deepfakes.
    """
    
    def __init__(self, 
                 num_classes: int = 1, 
                 dropout_rate: float = 0.5,
                 backbone: str = 'resnet18',
                 pretrained: bool = True,
                 freeze_backbone: bool = False,
                 use_attention: bool = True):
        """
        Initialize the enhanced model with attention.
        
        Args:
            num_classes (int): Number of output classes
            dropout_rate (float): Dropout rate for regularization
            backbone (str): Backbone architecture
            pretrained (bool): Whether to use pre-trained weights
            freeze_backbone (bool): Whether to freeze backbone layers
            use_attention (bool): Whether to use attention mechanism
        """
        super().__init__(num_classes, dropout_rate, backbone, pretrained, freeze_backbone)
        
        self.use_attention = use_attention
        
        if use_attention:
            # Add attention modules to different layers
            if backbone in ['resnet18', 'resnet34']:
                self.attention1 = AttentionModule(64)   # After first conv layer
                self.attention2 = AttentionModule(128)  # After second conv layer
                self.attention3 = AttentionModule(256)  # After third conv layer
                self.attention4 = AttentionModule(512)  # After fourth conv layer
            else:
                self.attention1 = AttentionModule(256)   # After first conv layer
                self.attention2 = AttentionModule(512)   # After second conv layer
                self.attention3 = AttentionModule(1024)  # After third conv layer
                self.attention4 = AttentionModule(2048)  # After fourth conv layer
            
            logger.info("Attention mechanism added to the model")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention mechanism.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output logits
        """
        if not self.use_attention:
            return super().forward(x)
        
        # Apply attention at different stages
        # Note: This is a simplified implementation. In practice, you might
        # want to modify the backbone to apply attention at specific layers
        
        # Extract features using backbone
        features = self.backbone(x)
        
        # Apply final attention
        if hasattr(self, 'attention4'):
            features = self.attention4(features)
        
        # Apply classification head
        logits = self.classifier(features)
        
        return logits


def create_model(model_config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to create a model based on configuration.
    
    Args:
        model_config (Dict[str, Any]): Model configuration dictionary
        
    Returns:
        nn.Module: Initialized model
        
    Example:
        config = {
            'model_type': 'deepfake_detector',
            'num_classes': 1,
            'dropout_rate': 0.5,
            'backbone': 'resnet18',
            'pretrained': True
        }
        model = create_model(config)
    """
    model_type = model_config.get('model_type', 'deepfake_detector')
    
    if model_type == 'deepfake_detector':
        return DeepfakeDetector(
            num_classes=model_config.get('num_classes', 1),
            dropout_rate=model_config.get('dropout_rate', 0.5),
            backbone=model_config.get('backbone', 'resnet18'),
            pretrained=model_config.get('pretrained', True),
            freeze_backbone=model_config.get('freeze_backbone', False)
        )
    elif model_type == 'deepfake_detector_attention':
        return DeepfakeDetectorWithAttention(
            num_classes=model_config.get('num_classes', 1),
            dropout_rate=model_config.get('dropout_rate', 0.5),
            backbone=model_config.get('backbone', 'resnet18'),
            pretrained=model_config.get('pretrained', True),
            freeze_backbone=model_config.get('freeze_backbone', False),
            use_attention=model_config.get('use_attention', True)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == '__main__':
    """
    Test script for model functionality.
    
    This block demonstrates how to use the model classes
    and can be used for testing during development.
    """
    print("Testing DeepFake Detection Model")
    print("=" * 35)
    
    # Test device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test basic model
    print("\n1. Testing basic DeepfakeDetector:")
    model = DeepfakeDetector(num_classes=1, dropout_rate=0.5)
    model = model.to(device)
    
    # Test forward pass
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 224, 224).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        print(f"Input shape: {input_tensor.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    # Test model info
    info = model.get_model_info()
    print(f"Model info: {info}")
    
    # Test predictions
    probs = model.predict_proba(input_tensor)
    predictions = model.predict(input_tensor)
    print(f"Probabilities shape: {probs.shape}")
    print(f"Predictions shape: {predictions.shape}")
    
    # Test attention model
    print("\n2. Testing DeepfakeDetectorWithAttention:")
    attention_model = DeepfakeDetectorWithAttention(
        num_classes=1, 
        dropout_rate=0.5,
        use_attention=True
    )
    attention_model = attention_model.to(device)
    
    with torch.no_grad():
        attention_output = attention_model(input_tensor)
        print(f"Attention model output shape: {attention_output.shape}")
    
    # Test factory function
    print("\n3. Testing model factory:")
    config = {
        'model_type': 'deepfake_detector',
        'num_classes': 1,
        'dropout_rate': 0.5,
        'backbone': 'resnet18',
        'pretrained': False  # Use False for testing to avoid downloading weights
    }
    
    factory_model = create_model(config)
    factory_model = factory_model.to(device)
    
    with torch.no_grad():
        factory_output = factory_model(input_tensor)
        print(f"Factory model output shape: {factory_output.shape}")
    
    print("\nModel test completed!") 