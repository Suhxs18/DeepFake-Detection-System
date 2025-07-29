"""
Utility functions for DeepFake Detection System.

This module provides essential utilities for:
- Device detection (CUDA/CPU)
- Face extraction from images and videos
- Video frame processing
- File handling and validation
"""

import os
import cv2
import torch
import numpy as np
from PIL import Image
from typing import Optional, Tuple, List
from facenet_pytorch import MTCNN
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """
    Determine the best available device for PyTorch operations.
    
    Returns:
        torch.device: CUDA device if available, otherwise CPU
        
    Example:
        device = get_device()
        model = model.to(device)
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU device")
    
    return device


def extract_face(image_path: str, 
                mtcnn_model: MTCNN, 
                save_path: Optional[str] = None, 
                size: Tuple[int, int] = (224, 224)) -> Optional[np.ndarray]:
    """
    Extract the largest face from an image and optionally save it.
    
    Args:
        image_path (str): Path to the input image
        mtcnn_model (MTCNN): Pre-loaded MTCNN face detection model
        save_path (Optional[str]): Path to save the extracted face image
        size (Tuple[int, int]): Target size for the extracted face (width, height)
        
    Returns:
        Optional[np.ndarray]: Extracted face as numpy array, or None if no face found
        
    Raises:
        FileNotFoundError: If the input image file doesn't exist
        ValueError: If the image cannot be loaded
        
    Example:
        mtcnn = MTCNN(image_size=224)
        face = extract_face('image.jpg', mtcnn, save_path='face.jpg')
    """
    try:
        # Load image using PIL
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        image = Image.open(image_path).convert('RGB')
        
        # Detect faces using MTCNN
        boxes, probs = mtcnn_model.detect(image)
        
        if boxes is None or len(boxes) == 0:
            logger.warning(f"No faces detected in {image_path}")
            return None
        
        # Get the face with highest confidence
        best_face_idx = np.argmax(probs)
        box = boxes[best_face_idx]
        
        # Extract face region
        x1, y1, x2, y2 = [int(b) for b in box]
        face = image.crop((x1, y1, x2, y2))
        
        # Resize face to target size
        face = face.resize(size, Image.LANCZOS)
        
        # Convert to numpy array
        face_array = np.array(face)
        
        # Save face if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            face.save(save_path)
            logger.info(f"Face saved to {save_path}")
        
        return face_array
        
    except Exception as e:
        logger.error(f"Error extracting face from {image_path}: {str(e)}")
        return None


def video_to_frames_and_faces(video_path: str, 
                             output_dir: str, 
                             mtcnn_model: MTCNN, 
                             frame_interval: int = 30, 
                             size: Tuple[int, int] = (224, 224)) -> List[str]:
    """
    Extract frames from a video and detect faces in each frame.
    
    Args:
        video_path (str): Path to the input video file
        output_dir (str): Directory to save extracted faces
        mtcnn_model (MTCNN): Pre-loaded MTCNN face detection model
        frame_interval (int): Extract every Nth frame (default: 30)
        size (Tuple[int, int]): Target size for extracted faces (width, height)
        
    Returns:
        List[str]: List of paths to saved face images
        
    Raises:
        FileNotFoundError: If the video file doesn't exist
        ValueError: If the video cannot be opened
        
    Example:
        mtcnn = MTCNN(image_size=224)
        face_paths = video_to_frames_and_faces('video.mp4', 'output/', mtcnn)
    """
    try:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"Processing video: {video_path}")
        logger.info(f"Total frames: {total_frames}, FPS: {fps:.2f}, Duration: {duration:.2f}s")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        saved_faces = []
        frame_count = 0
        processed_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every Nth frame
            if frame_count % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                
                # Detect faces in the frame
                boxes, probs = mtcnn_model.detect(frame_pil)
                
                if boxes is not None and len(boxes) > 0:
                    # Get the face with highest confidence
                    best_face_idx = np.argmax(probs)
                    box = boxes[best_face_idx]
                    
                    # Extract face region
                    x1, y1, x2, y2 = [int(b) for b in box]
                    face = frame_pil.crop((x1, y1, x2, y2))
                    
                    # Resize face
                    face = face.resize(size, Image.LANCZOS)
                    
                    # Save face
                    face_filename = f"frame_{frame_count:06d}_face.jpg"
                    face_path = os.path.join(output_dir, face_filename)
                    face.save(face_path)
                    saved_faces.append(face_path)
                    processed_frames += 1
                    
                    logger.debug(f"Saved face from frame {frame_count}: {face_path}")
            
            frame_count += 1
            
            # Progress update every 1000 frames
            if frame_count % 1000 == 0:
                logger.info(f"Processed {frame_count}/{total_frames} frames, extracted {processed_frames} faces")
        
        cap.release()
        logger.info(f"Video processing complete. Extracted {len(saved_faces)} faces from {processed_frames} frames")
        
        return saved_faces
        
    except Exception as e:
        logger.error(f"Error processing video {video_path}: {str(e)}")
        return []


def validate_image_file(file_path: str) -> bool:
    """
    Validate if a file is a valid image file.
    
    Args:
        file_path (str): Path to the file to validate
        
    Returns:
        bool: True if file is a valid image, False otherwise
        
    Example:
        if validate_image_file('image.jpg'):
            process_image('image.jpg')
    """
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def validate_video_file(file_path: str) -> bool:
    """
    Validate if a file is a valid video file.
    
    Args:
        file_path (str): Path to the file to validate
        
    Returns:
        bool: True if file is a valid video, False otherwise
        
    Example:
        if validate_video_file('video.mp4'):
            process_video('video.mp4')
    """
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return False
        cap.release()
        return True
    except Exception:
        return False


def get_file_extension(file_path: str) -> str:
    """
    Get the file extension from a file path.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        str: File extension (lowercase, without dot)
        
    Example:
        ext = get_file_extension('image.jpg')  # Returns 'jpg'
    """
    return os.path.splitext(file_path)[1].lower()[1:]


def is_image_file(file_path: str) -> bool:
    """
    Check if a file is an image based on its extension.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        bool: True if file has image extension, False otherwise
        
    Example:
        if is_image_file('photo.jpg'):
            process_image('photo.jpg')
    """
    image_extensions = {'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif', 'gif'}
    return get_file_extension(file_path) in image_extensions


def is_video_file(file_path: str) -> bool:
    """
    Check if a file is a video based on its extension.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        bool: True if file has video extension, False otherwise
        
    Example:
        if is_video_file('movie.mp4'):
            process_video('movie.mp4')
    """
    video_extensions = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'}
    return get_file_extension(file_path) in video_extensions


if __name__ == '__main__':
    """
    Test script for utility functions.
    
    This block demonstrates how to use the utility functions
    and can be used for testing during development.
    """
    print("Testing DeepFake Detection Utilities")
    print("=" * 40)
    
    # Test device detection
    device = get_device()
    print(f"Detected device: {device}")
    
    # Test MTCNN initialization
    try:
        mtcnn = MTCNN(image_size=224, margin=20, keep_all=False)
        print("MTCNN model initialized successfully")
    except Exception as e:
        print(f"Error initializing MTCNN: {e}")
    
    # Test file validation functions
    test_files = [
        'nonexistent.jpg',
        'test_image.jpg',
        'test_video.mp4'
    ]
    
    for file_path in test_files:
        print(f"\nTesting file: {file_path}")
        print(f"  Is image: {is_image_file(file_path)}")
        print(f"  Is video: {is_video_file(file_path)}")
        print(f"  Valid image: {validate_image_file(file_path)}")
        print(f"  Valid video: {validate_video_file(file_path)}")
    
    print("\nUtility functions test completed!") 