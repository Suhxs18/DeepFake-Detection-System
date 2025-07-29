#!/usr/bin/env python3
"""
Prediction script for DeepFake Detection System.

This script provides inference capabilities for:
- Single image prediction
- Video frame prediction
- Batch processing of multiple files
- Real-time prediction with confidence scores
"""

import os
import sys
import argparse
import logging
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import cv2
import numpy as np
from PIL import Image

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn.functional as F
from torchvision import transforms

from utils import (
    get_device, 
    extract_face, 
    is_image_file, 
    is_video_file,
    validate_image_file,
    validate_video_file
)
from dataset import get_transforms
from model import DeepfakeDetector
from facenet_pytorch import MTCNN

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DeepfakePredictor:
    """
    Prediction class for deepfake detection.
    
    This class handles model loading, preprocessing, and inference
    for both images and videos.
    """
    
    def __init__(self, model_path: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the predictor.
        
        Args:
            model_path (str): Path to the trained model checkpoint
            config (Optional[Dict[str, Any]]): Model configuration
        """
        self.device = get_device()
        self.model_path = model_path
        
        # Load model and configuration
        self._load_model(model_path, config)
        
        # Initialize face detection
        self._initialize_face_detection()
        
        # Initialize transforms
        self._initialize_transforms()
        
        logger.info("DeepfakePredictor initialized successfully")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Model loaded from: {model_path}")
    
    def _load_model(self, model_path: str, config: Optional[Dict[str, Any]] = None):
        """
        Load the trained model from checkpoint.
        
        Args:
            model_path (str): Path to model checkpoint
            config (Optional[Dict[str, Any]]): Model configuration
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get configuration
        if config is None:
            if 'config' in checkpoint:
                self.config = checkpoint['config']
            else:
                # Default configuration
                self.config = {
                    'model_config': {
                        'num_classes': 1,
                        'backbone': 'resnet18',
                        'dropout_rate': 0.5
                    },
                    'image_size': 224
                }
        else:
            self.config = config
        
        # Create model
        from model import create_model
        self.model = create_model(self.config['model_config'])
        
        # Load model weights
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded successfully")
        logger.info(f"Model configuration: {self.config['model_config']}")
    
    def _initialize_face_detection(self):
        """Initialize MTCNN for face detection."""
        self.mtcnn = MTCNN(
            image_size=self.config.get('image_size', 224),
            margin=20,
            keep_all=False,
            device=self.device
        )
    
    def _initialize_transforms(self):
        """Initialize image transforms for preprocessing."""
        self.transform = get_transforms(
            image_size=self.config.get('image_size', 224),
            is_training=False,
            use_albumentations=True
        )
    
    def preprocess_image(self, image_path: str) -> Optional[torch.Tensor]:
        """
        Preprocess a single image for prediction.
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            Optional[torch.Tensor]: Preprocessed image tensor, or None if no face detected
        """
        try:
            # Extract face
            face = extract_face(
                image_path=image_path,
                mtcnn_model=self.mtcnn,
                save_path=None,
                size=(self.config.get('image_size', 224), self.config.get('image_size', 224))
            )
            
            if face is None:
                logger.warning(f"No face detected in {image_path}")
                return None
            
            # Apply transforms
            if hasattr(self.transform, 'image'):
                # Albumentations transform
                transformed = self.transform(image=face)
                image_tensor = transformed['image']
            else:
                # PyTorch transform
                image_pil = Image.fromarray(face)
                image_tensor = self.transform(image_pil)
            
            # Add batch dimension
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            
            return image_tensor
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {str(e)}")
            return None
    
    def preprocess_video_frame(self, frame: np.ndarray) -> Optional[torch.Tensor]:
        """
        Preprocess a video frame for prediction.
        
        Args:
            frame (np.ndarray): Video frame as numpy array
            
        Returns:
            Optional[torch.Tensor]: Preprocessed frame tensor, or None if no face detected
        """
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            
            # Detect faces
            boxes, probs = self.mtcnn.detect(frame_pil)
            
            if boxes is None or len(boxes) == 0:
                return None
            
            # Get the face with highest confidence
            best_face_idx = np.argmax(probs)
            box = boxes[best_face_idx]
            
            # Extract face region
            x1, y1, x2, y2 = [int(b) for b in box]
            face = frame_pil.crop((x1, y1, x2, y2))
            
            # Resize face
            face_size = self.config.get('image_size', 224)
            face = face.resize((face_size, face_size), Image.LANCZOS)
            
            # Apply transforms
            if hasattr(self.transform, 'image'):
                # Albumentations transform
                face_np = np.array(face)
                transformed = self.transform(image=face_np)
                image_tensor = transformed['image']
            else:
                # PyTorch transform
                image_tensor = self.transform(face)
            
            # Add batch dimension
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            
            return image_tensor
            
        except Exception as e:
            logger.error(f"Error preprocessing video frame: {str(e)}")
            return None
    
    def predict_image(self, image_path: str) -> Dict[str, Any]:
        """
        Predict on a single image.
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            Dict[str, Any]: Prediction results including class, probability, and confidence
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image_path)
        
        if image_tensor is None:
            return {
                'file_path': image_path,
                'prediction': 'no_face',
                'probability': 0.0,
                'confidence': 0.0,
                'error': 'No face detected'
            }
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            
            if self.config['model_config']['num_classes'] == 1:
                # Binary classification
                probability = torch.sigmoid(outputs).item()
                prediction = 'fake' if probability > 0.5 else 'real'
                confidence = max(probability, 1 - probability)
            else:
                # Multi-class classification
                probabilities = F.softmax(outputs, dim=1)
                probability = probabilities[0, 1].item()  # Probability of fake class
                prediction = 'fake' if probability > 0.5 else 'real'
                confidence = max(probability, 1 - probability)
        
        return {
            'file_path': image_path,
            'prediction': prediction,
            'probability': probability,
            'confidence': confidence,
            'error': None
        }
    
    def predict_video(self, video_path: str, frame_interval: int = 30) -> Dict[str, Any]:
        """
        Predict on a video by sampling frames.
        
        Args:
            video_path (str): Path to the input video
            frame_interval (int): Extract every Nth frame
            
        Returns:
            Dict[str, Any]: Prediction results for the video
        """
        if not os.path.exists(video_path):
            return {
                'file_path': video_path,
                'error': 'Video file not found',
                'predictions': []
            }
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {
                'file_path': video_path,
                'error': 'Cannot open video file',
                'predictions': []
            }
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"Processing video: {video_path}")
        logger.info(f"Total frames: {total_frames}, FPS: {fps:.2f}, Duration: {duration:.2f}s")
        
        predictions = []
        frame_count = 0
        processed_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every Nth frame
            if frame_count % frame_interval == 0:
                # Preprocess frame
                frame_tensor = self.preprocess_video_frame(frame)
                
                if frame_tensor is not None:
                    # Make prediction
                    with torch.no_grad():
                        outputs = self.model(frame_tensor)
                        
                        if self.config['model_config']['num_classes'] == 1:
                            # Binary classification
                            probability = torch.sigmoid(outputs).item()
                            prediction = 'fake' if probability > 0.5 else 'real'
                            confidence = max(probability, 1 - probability)
                        else:
                            # Multi-class classification
                            probabilities = F.softmax(outputs, dim=1)
                            probability = probabilities[0, 1].item()
                            prediction = 'fake' if probability > 0.5 else 'real'
                            confidence = max(probability, 1 - probability)
                    
                    predictions.append({
                        'frame': frame_count,
                        'timestamp': frame_count / fps if fps > 0 else 0,
                        'prediction': prediction,
                        'probability': probability,
                        'confidence': confidence
                    })
                    
                    processed_frames += 1
                    
                    logger.debug(f"Frame {frame_count}: {prediction} (prob: {probability:.3f})")
            
            frame_count += 1
        
        cap.release()
        
        # Calculate video-level prediction
        if predictions:
            fake_probabilities = [p['probability'] for p in predictions]
            avg_fake_probability = np.mean(fake_probabilities)
            video_prediction = 'fake' if avg_fake_probability > 0.5 else 'real'
            video_confidence = max(avg_fake_probability, 1 - avg_fake_probability)
            
            # Count predictions
            fake_count = sum(1 for p in predictions if p['prediction'] == 'fake')
            real_count = len(predictions) - fake_count
            
            result = {
                'file_path': video_path,
                'video_prediction': video_prediction,
                'video_probability': avg_fake_probability,
                'video_confidence': video_confidence,
                'processed_frames': processed_frames,
                'fake_frames': fake_count,
                'real_frames': real_count,
                'frame_predictions': predictions,
                'error': None
            }
        else:
            result = {
                'file_path': video_path,
                'error': 'No faces detected in video',
                'frame_predictions': []
            }
        
        return result
    
    def predict_batch(self, input_paths: List[str], output_file: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Predict on multiple files (images or videos).
        
        Args:
            input_paths (List[str]): List of input file paths
            output_file (Optional[str]): Path to save results JSON file
            
        Returns:
            List[Dict[str, Any]]: List of prediction results
        """
        results = []
        
        for i, file_path in enumerate(input_paths):
            logger.info(f"Processing {i+1}/{len(input_paths)}: {file_path}")
            
            if is_image_file(file_path):
                result = self.predict_image(file_path)
            elif is_video_file(file_path):
                result = self.predict_video(file_path)
            else:
                result = {
                    'file_path': file_path,
                    'error': 'Unsupported file type'
                }
            
            results.append(result)
            
            # Print result
            if 'error' in result and result['error']:
                logger.warning(f"Error processing {file_path}: {result['error']}")
            else:
                if 'prediction' in result:
                    logger.info(f"Result: {result['prediction']} (prob: {result['probability']:.3f})")
                elif 'video_prediction' in result:
                    logger.info(f"Video result: {result['video_prediction']} (prob: {result['video_probability']:.3f})")
        
        # Save results if output file specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to: {output_file}")
        
        return results
    
    def predict_directory(self, input_dir: str, output_file: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Predict on all supported files in a directory.
        
        Args:
            input_dir (str): Input directory path
            output_file (Optional[str]): Path to save results JSON file
            
        Returns:
            List[Dict[str, Any]]: List of prediction results
        """
        if not os.path.exists(input_dir):
            logger.error(f"Input directory not found: {input_dir}")
            return []
        
        # Find all supported files
        supported_files = []
        for filename in os.listdir(input_dir):
            file_path = os.path.join(input_dir, filename)
            if os.path.isfile(file_path):
                if is_image_file(file_path) or is_video_file(file_path):
                    supported_files.append(file_path)
        
        logger.info(f"Found {len(supported_files)} supported files in {input_dir}")
        
        return self.predict_batch(supported_files, output_file)


def load_model(model_path: str, config: Optional[Dict[str, Any]] = None) -> DeepfakePredictor:
    """
    Load a trained model for prediction.
    
    Args:
        model_path (str): Path to the model checkpoint
        config (Optional[Dict[str, Any]]): Model configuration
        
    Returns:
        DeepfakePredictor: Initialized predictor
    """
    return DeepfakePredictor(model_path, config)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Predict deepfake detection on images/videos')
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--image', type=str,
                       help='Path to input image')
    parser.add_argument('--video', type=str,
                       help='Path to input video')
    parser.add_argument('--input_dir', type=str,
                       help='Directory containing input files')
    parser.add_argument('--output', type=str,
                       help='Output JSON file for batch results')
    parser.add_argument('--frame_interval', type=int, default=30,
                       help='Frame interval for video processing (default: 30)')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                       help='Confidence threshold for predictions (default: 0.5)')
    
    return parser.parse_args()


def main():
    """Main function to run predictions."""
    # Parse arguments
    args = parse_arguments()
    
    # Validate arguments
    if not args.image and not args.video and not args.input_dir:
        logger.error("Must specify either --image, --video, or --input_dir")
        sys.exit(1)
    
    if not os.path.exists(args.model):
        logger.error(f"Model file not found: {args.model}")
        sys.exit(1)
    
    try:
        # Load model
        logger.info(f"Loading model from: {args.model}")
        predictor = load_model(args.model)
        
        # Run predictions
        if args.image:
            logger.info(f"Predicting on image: {args.image}")
            result = predictor.predict_image(args.image)
            
            if result['error']:
                logger.error(f"Error: {result['error']}")
            else:
                logger.info(f"Prediction: {result['prediction']}")
                logger.info(f"Probability: {result['probability']:.3f}")
                logger.info(f"Confidence: {result['confidence']:.3f}")
        
        elif args.video:
            logger.info(f"Predicting on video: {args.video}")
            result = predictor.predict_video(args.video, args.frame_interval)
            
            if result['error']:
                logger.error(f"Error: {result['error']}")
            else:
                logger.info(f"Video prediction: {result['video_prediction']}")
                logger.info(f"Video probability: {result['video_probability']:.3f}")
                logger.info(f"Processed frames: {result['processed_frames']}")
                logger.info(f"Fake frames: {result['fake_frames']}, Real frames: {result['real_frames']}")
        
        elif args.input_dir:
            logger.info(f"Predicting on directory: {args.input_dir}")
            results = predictor.predict_directory(args.input_dir, args.output)
            
            # Print summary
            total_files = len(results)
            successful_predictions = sum(1 for r in results if 'error' not in r or not r['error'])
            logger.info(f"Processed {successful_predictions}/{total_files} files successfully")
        
        logger.info("Prediction completed!")
        
    except KeyboardInterrupt:
        logger.info("Prediction interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main() 