#!/usr/bin/env python3
"""
Improved Data preprocessing script for DeepFake Detection System.

This script processes raw data (videos and images) to extract faces
and organize them into the required directory structure for training.
Uses more lenient face detection parameters for better real image processing.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
import random
import shutil

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import (
    extract_face, 
    video_to_frames_and_faces, 
    is_image_file, 
    is_video_file,
    validate_image_file,
    validate_video_file,
    get_device
)
from facenet_pytorch import MTCNN

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocess_improved.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ImprovedDataPreprocessor:
    """
    Improved data preprocessing class with better face detection.
    
    This class handles the complete preprocessing pipeline including:
    - Face extraction from images and videos with lenient parameters
    - Data organization into train/val/test splits
    - Quality control and validation
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the improved data preprocessor.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
        """
        self.config = config
        self.device = get_device()
        
        # Initialize MTCNN model with more lenient parameters
        self.mtcnn = MTCNN(
            image_size=config['face_size'],
            margin=config['face_margin'],
            min_face_size=10,  # Lower minimum face size
            thresholds=[0.3, 0.4, 0.5],  # More lenient thresholds
            factor=0.709,
            keep_all=False,
            device=self.device
        )
        
        # Create output directories
        self._create_output_directories()
        
        logger.info("Improved DataPreprocessor initialized successfully")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Face size: {config['face_size']}")
        logger.info(f"Face margin: {config['face_margin']}")
        logger.info("Using lenient face detection parameters")
    
    def _create_output_directories(self):
        """Create the required output directory structure."""
        base_dir = self.config['output_dir']
        
        # Create main directories
        for split in ['train', 'val', 'test']:
            for class_name in ['real', 'fake']:
                dir_path = os.path.join(base_dir, split, class_name)
                os.makedirs(dir_path, exist_ok=True)
                logger.debug(f"Created directory: {dir_path}")
    
    def process_raw_data(self):
        """
        Main method to process all raw data.
        
        This method orchestrates the complete preprocessing pipeline:
        1. Scan raw data directories
        2. Extract faces from images and videos
        3. Organize into train/val/test splits
        4. Validate and clean the processed data
        """
        logger.info("Starting improved data preprocessing...")
        
        # Process real data
        logger.info("Processing real data...")
        real_data = self._scan_raw_data('real')
        self._process_data_split(real_data, 'real')
        
        # Process fake data
        logger.info("Processing fake data...")
        fake_data = self._scan_raw_data('fake')
        self._process_data_split(fake_data, 'fake')
        
        # Generate summary
        self._generate_summary()
        
        logger.info("Improved data preprocessing completed successfully!")
    
    def _scan_raw_data(self, class_name: str) -> Dict[str, List[str]]:
        """
        Scan raw data directory for images and videos.
        
        Args:
            class_name (str): Class name ('real' or 'fake')
            
        Returns:
            Dict[str, List[str]]: Dictionary with 'images' and 'videos' lists
        """
        raw_dir = os.path.join(self.config['raw_data_dir'], class_name)
        
        if not os.path.exists(raw_dir):
            logger.warning(f"Raw data directory not found: {raw_dir}")
            return {'images': [], 'videos': []}
        
        images = []
        videos = []
        
        for filename in os.listdir(raw_dir):
            file_path = os.path.join(raw_dir, filename)
            
            if is_image_file(file_path) and validate_image_file(file_path):
                images.append(file_path)
            elif is_video_file(file_path) and validate_video_file(file_path):
                videos.append(file_path)
        
        logger.info(f"Found {len(images)} images and {len(videos)} videos for {class_name}")
        return {'images': images, 'videos': videos}
    
    def _process_data_split(self, data: Dict[str, List[str]], class_name: str):
        """
        Process data and split into train/val/test sets.
        
        Args:
            data (Dict[str, List[str]]): Dictionary with 'images' and 'videos' lists
            class_name (str): Class name ('real' or 'fake')
        """
        all_files = data['images'] + data['videos']
        
        if not all_files:
            logger.warning(f"No files found for {class_name}")
            return
        
        # Shuffle files for random split
        random.shuffle(all_files)
        
        # Calculate split sizes
        total_files = len(all_files)
        train_size = int(total_files * self.config['train_ratio'])
        val_size = int(total_files * self.config['val_ratio'])
        
        # Split files
        train_files = all_files[:train_size]
        val_files = all_files[train_size:train_size + val_size]
        test_files = all_files[train_size + val_size:]
        
        logger.info(f"Split {class_name} data: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
        
        # Process each split
        self._process_split_files(train_files, class_name, 'train')
        self._process_split_files(val_files, class_name, 'val')
        self._process_split_files(test_files, class_name, 'test')
    
    def _process_split_files(self, files: List[str], class_name: str, split_name: str):
        """
        Process files for a specific split.
        
        Args:
            files (List[str]): List of file paths
            class_name (str): Class name ('real' or 'fake')
            split_name (str): Split name ('train', 'val', or 'test')
        """
        output_dir = os.path.join(self.config['output_dir'], split_name, class_name)
        
        successful = 0
        failed = 0
        
        for i, file_path in enumerate(files):
            if is_image_file(file_path):
                if self._process_image(file_path, output_dir, i):
                    successful += 1
                else:
                    failed += 1
            elif is_video_file(file_path):
                if self._process_video(file_path, output_dir, i):
                    successful += 1
                else:
                    failed += 1
        
        logger.info(f"Processed {split_name}/{class_name}: {successful} successful, {failed} failed")
    
    def _process_image(self, image_path: str, output_dir: str, index: int) -> bool:
        """
        Process a single image and extract faces.
        
        Args:
            image_path (str): Path to the input image
            output_dir (str): Output directory for extracted faces
            index (int): Index for naming output files
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Generate output filename
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}_{index:06d}.jpg")
            
            # Extract face using improved parameters
            face_array = extract_face(image_path, self.mtcnn, output_path, self.config['face_size'])
            
            if face_array is not None:
                return True
            else:
                logger.warning(f"No face detected in {image_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            return False
    
    def _process_video(self, video_path: str, output_dir: str, start_index: int) -> bool:
        """
        Process a single video and extract faces from frames.
        
        Args:
            video_path (str): Path to the input video
            output_dir (str): Output directory for extracted faces
            start_index (int): Starting index for naming output files
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Extract faces from video frames
            face_paths = video_to_frames_and_faces(
                video_path, 
                output_dir, 
                self.mtcnn, 
                self.config['frame_interval'],
                self.config['face_size']
            )
            
            if face_paths:
                return True
            else:
                logger.warning(f"No faces extracted from video {video_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {str(e)}")
            return False
    
    def _generate_summary(self):
        """Generate a summary of the processed data."""
        logger.info("Generating data summary...")
        
        summary_lines = [
            "DeepFake Detection Data Processing Summary",
            "=" * 50,
            f"Total faces extracted: {self._count_total_faces()}",
            "",
            "TRAIN:",
            f"  Real: {self._count_faces('train', 'real')}",
            f"  Fake: {self._count_faces('train', 'fake')}",
            f"  Total: {self._count_faces('train', 'real') + self._count_faces('train', 'fake')}",
            "",
            "VAL:",
            f"  Real: {self._count_faces('val', 'real')}",
            f"  Fake: {self._count_faces('val', 'fake')}",
            f"  Total: {self._count_faces('val', 'real') + self._count_faces('val', 'fake')}",
            "",
            "TEST:",
            f"  Real: {self._count_faces('test', 'real')}",
            f"  Fake: {self._count_faces('test', 'fake')}",
            f"  Total: {self._count_faces('test', 'real') + self._count_faces('test', 'fake')}",
            ""
        ]
        
        # Print summary
        for line in summary_lines:
            logger.info(line)
        
        # Save summary to file
        summary_path = os.path.join(self.config['output_dir'], 'processing_summary_improved.txt')
        with open(summary_path, 'w') as f:
            f.write('\n'.join(summary_lines))
        
        logger.info(f"Summary saved to: {summary_path}")
    
    def _count_faces(self, split: str, class_name: str) -> int:
        """Count faces in a specific split and class."""
        dir_path = os.path.join(self.config['output_dir'], split, class_name)
        if os.path.exists(dir_path):
            return len([f for f in os.listdir(dir_path) if f.endswith('.jpg')])
        return 0
    
    def _count_total_faces(self) -> int:
        """Count total faces across all splits and classes."""
        total = 0
        for split in ['train', 'val', 'test']:
            for class_name in ['real', 'fake']:
                total += self._count_faces(split, class_name)
        return total


def get_default_config() -> Dict[str, Any]:
    """Get default configuration for the preprocessor."""
    return {
        'raw_data_dir': 'data/raw',
        'output_dir': 'data/processed_faces',
        'face_size': (224, 224),
        'face_margin': 20,
        'frame_interval': 30,
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15
    }


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Improved Data Preprocessing for DeepFake Detection')
    
    parser.add_argument('--raw-data-dir', type=str, default='data/raw',
                       help='Directory containing raw data')
    parser.add_argument('--output-dir', type=str, default='data/processed_faces',
                       help='Output directory for processed faces')
    parser.add_argument('--face-size', type=int, nargs=2, default=[224, 224],
                       help='Size of extracted faces (width height)')
    parser.add_argument('--face-margin', type=int, default=20,
                       help='Margin around detected faces')
    parser.add_argument('--frame-interval', type=int, default=30,
                       help='Frame interval for video processing')
    
    return parser.parse_args()


def main():
    """Main function to run the improved preprocessing."""
    args = parse_arguments()
    
    # Create configuration
    config = get_default_config()
    config.update(vars(args))
    
    # Initialize and run preprocessor
    preprocessor = ImprovedDataPreprocessor(config)
    preprocessor.process_raw_data()


if __name__ == '__main__':
    main() 