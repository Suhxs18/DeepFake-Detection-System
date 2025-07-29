#!/usr/bin/env python3
"""
Data preprocessing script for DeepFake Detection System.

This script processes raw data (videos and images) to extract faces
and organize them into the required directory structure for training.
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
        logging.FileHandler('preprocess.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Data preprocessing class for organizing and processing raw data.
    
    This class handles the complete preprocessing pipeline including:
    - Face extraction from images and videos
    - Data organization into train/val/test splits
    - Quality control and validation
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data preprocessor.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
        """
        self.config = config
        self.device = get_device()
        
        # Initialize MTCNN model
        self.mtcnn = MTCNN(
            image_size=config['face_size'],
            margin=config['face_margin'],
            keep_all=False,
            device=self.device
        )
        
        # Create output directories
        self._create_output_directories()
        
        logger.info("DataPreprocessor initialized successfully")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Face size: {config['face_size']}")
        logger.info(f"Face margin: {config['face_margin']}")
    
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
        logger.info("Starting data preprocessing pipeline")
        
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
        
        logger.info("Data preprocessing completed successfully!")
    
    def _scan_raw_data(self, class_name: str) -> Dict[str, List[str]]:
        """
        Scan raw data directory for a specific class.
        
        Args:
            class_name (str): Class name ('real' or 'fake')
            
        Returns:
            Dict[str, List[str]]: Dictionary with 'images' and 'videos' lists
        """
        raw_dir = os.path.join(self.config['raw_data_dir'], class_name)
        
        if not os.path.exists(raw_dir):
            logger.warning(f"Raw data directory not found: {raw_dir}")
            return {'images': [], 'videos': []}
        
        data = {'images': [], 'videos': []}
        
        for filename in os.listdir(raw_dir):
            file_path = os.path.join(raw_dir, filename)
            
            if os.path.isfile(file_path):
                if is_image_file(file_path) and validate_image_file(file_path):
                    data['images'].append(file_path)
                elif is_video_file(file_path) and validate_video_file(file_path):
                    data['videos'].append(file_path)
        
        logger.info(f"Found {len(data['images'])} images and {len(data['videos'])} videos for {class_name}")
        return data
    
    def _process_data_split(self, data: Dict[str, List[str]], class_name: str):
        """
        Process data for a specific class and split into train/val/test.
        
        Args:
            data (Dict[str, List[str]]): Dictionary with 'images' and 'videos' lists
            class_name (str): Class name ('real' or 'fake')
        """
        # Combine all files
        all_files = data['images'] + data['videos']
        
        if not all_files:
            logger.warning(f"No valid files found for class: {class_name}")
            return
        
        # Shuffle files for random splitting
        random.shuffle(all_files)
        
        # Calculate split indices
        total_files = len(all_files)
        train_end = int(total_files * self.config['train_split'])
        val_end = train_end + int(total_files * self.config['val_split'])
        
        # Split files
        train_files = all_files[:train_end]
        val_files = all_files[train_end:val_end]
        test_files = all_files[val_end:]
        
        logger.info(f"Split {class_name} data: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
        
        # Process each split
        self._process_split_files(train_files, class_name, 'train')
        self._process_split_files(val_files, class_name, 'val')
        self._process_split_files(test_files, class_name, 'test')
    
    def _process_split_files(self, files: List[str], class_name: str, split_name: str):
        """
        Process files for a specific split.
        
        Args:
            files (List[str]): List of file paths to process
            class_name (str): Class name ('real' or 'fake')
            split_name (str): Split name ('train', 'val', or 'test')
        """
        output_dir = os.path.join(self.config['output_dir'], split_name, class_name)
        
        processed_count = 0
        failed_count = 0
        
        for file_path in files:
            try:
                if is_image_file(file_path):
                    success = self._process_image(file_path, output_dir, processed_count)
                elif is_video_file(file_path):
                    success = self._process_video(file_path, output_dir, processed_count)
                else:
                    logger.warning(f"Unsupported file type: {file_path}")
                    failed_count += 1
                    continue
                
                if success:
                    processed_count += 1
                else:
                    failed_count += 1
                    
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                failed_count += 1
        
        logger.info(f"Processed {split_name}/{class_name}: {processed_count} successful, {failed_count} failed")
    
    def _process_image(self, image_path: str, output_dir: str, index: int) -> bool:
        """
        Process a single image file.
        
        Args:
            image_path (str): Path to the input image
            output_dir (str): Output directory for extracted faces
            index (int): Index for naming the output file
            
        Returns:
            bool: True if processing was successful, False otherwise
        """
        try:
            # Generate output filename
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_filename = f"{base_name}_{index:06d}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            
            # Extract face
            face = extract_face(
                image_path=image_path,
                mtcnn_model=self.mtcnn,
                save_path=output_path,
                size=self.config['face_size']
            )
            
            if face is not None:
                logger.debug(f"Successfully extracted face from {image_path}")
                return True
            else:
                logger.warning(f"No face detected in {image_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            return False
    
    def _process_video(self, video_path: str, output_dir: str, start_index: int) -> bool:
        """
        Process a single video file.
        
        Args:
            video_path (str): Path to the input video
            output_dir (str): Output directory for extracted faces
            start_index (int): Starting index for naming output files
            
        Returns:
            bool: True if processing was successful, False otherwise
        """
        try:
            # Create temporary directory for video processing
            temp_dir = os.path.join(output_dir, f"temp_{start_index}")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Extract faces from video
            face_paths = video_to_frames_and_faces(
                video_path=video_path,
                output_dir=temp_dir,
                mtcnn_model=self.mtcnn,
                frame_interval=self.config['frame_interval'],
                size=self.config['face_size']
            )
            
            if not face_paths:
                logger.warning(f"No faces extracted from video {video_path}")
                shutil.rmtree(temp_dir, ignore_errors=True)
                return False
            
            # Move faces to output directory with proper naming
            for i, face_path in enumerate(face_paths):
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                new_filename = f"{base_name}_{start_index + i:06d}.jpg"
                new_path = os.path.join(output_dir, new_filename)
                
                shutil.move(face_path, new_path)
            
            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            logger.debug(f"Successfully extracted {len(face_paths)} faces from {video_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {str(e)}")
            # Clean up temporary directory if it exists
            temp_dir = os.path.join(output_dir, f"temp_{start_index}")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return False
    
    def _generate_summary(self):
        """Generate a summary of the processed data."""
        logger.info("Generating data summary...")
        
        summary = {}
        total_faces = 0
        
        for split in ['train', 'val', 'test']:
            summary[split] = {}
            for class_name in ['real', 'fake']:
                dir_path = os.path.join(self.config['output_dir'], split, class_name)
                if os.path.exists(dir_path):
                    file_count = len([f for f in os.listdir(dir_path) if f.endswith('.jpg')])
                    summary[split][class_name] = file_count
                    total_faces += file_count
                else:
                    summary[split][class_name] = 0
        
        # Log summary
        logger.info("=" * 50)
        logger.info("DATA PROCESSING SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total faces extracted: {total_faces}")
        
        for split in ['train', 'val', 'test']:
            logger.info(f"\n{split.upper()}:")
            real_count = summary[split]['real']
            fake_count = summary[split]['fake']
            logger.info(f"  Real: {real_count}")
            logger.info(f"  Fake: {fake_count}")
            logger.info(f"  Total: {real_count + fake_count}")
        
        # Save summary to file
        summary_file = os.path.join(self.config['output_dir'], 'processing_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("DeepFake Detection Data Processing Summary\n")
            f.write("=" * 50 + "\n")
            f.write(f"Total faces extracted: {total_faces}\n\n")
            
            for split in ['train', 'val', 'test']:
                f.write(f"{split.upper()}:\n")
                real_count = summary[split]['real']
                fake_count = summary[split]['fake']
                f.write(f"  Real: {real_count}\n")
                f.write(f"  Fake: {fake_count}\n")
                f.write(f"  Total: {real_count + fake_count}\n\n")
        
        logger.info(f"Summary saved to: {summary_file}")


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration for data preprocessing.
    
    Returns:
        Dict[str, Any]: Default configuration dictionary
    """
    return {
        'raw_data_dir': 'data/raw',
        'output_dir': 'data/processed_faces',
        'face_size': (224, 224),
        'face_margin': 20,
        'frame_interval': 30,  # Extract every 30th frame from videos
        'train_split': 0.7,    # 70% for training
        'val_split': 0.15,     # 15% for validation
        'test_split': 0.15,    # 15% for testing
        'random_seed': 42
    }


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Preprocess raw data for deepfake detection')
    
    parser.add_argument('--raw_data_dir', type=str, default='data/raw',
                       help='Directory containing raw data (default: data/raw)')
    parser.add_argument('--output_dir', type=str, default='data/processed_faces',
                       help='Output directory for processed faces (default: data/processed_faces)')
    parser.add_argument('--face_size', type=int, nargs=2, default=[224, 224],
                       help='Size of extracted faces (default: 224 224)')
    parser.add_argument('--face_margin', type=int, default=20,
                       help='Margin around detected faces (default: 20)')
    parser.add_argument('--frame_interval', type=int, default=30,
                       help='Extract every Nth frame from videos (default: 30)')
    parser.add_argument('--train_split', type=float, default=0.7,
                       help='Training split ratio (default: 0.7)')
    parser.add_argument('--val_split', type=float, default=0.15,
                       help='Validation split ratio (default: 0.15)')
    parser.add_argument('--test_split', type=float, default=0.15,
                       help='Test split ratio (default: 0.15)')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    return parser.parse_args()


def main():
    """Main function to run the data preprocessing pipeline."""
    # Parse arguments
    args = parse_arguments()
    
    # Set random seed
    random.seed(args.random_seed)
    
    # Create configuration
    config = get_default_config()
    
    # Update config with command line arguments
    config.update({
        'raw_data_dir': args.raw_data_dir,
        'output_dir': args.output_dir,
        'face_size': tuple(args.face_size),
        'face_margin': args.face_margin,
        'frame_interval': args.frame_interval,
        'train_split': args.train_split,
        'val_split': args.val_split,
        'test_split': args.test_split,
        'random_seed': args.random_seed
    })
    
    # Validate splits
    total_split = config['train_split'] + config['val_split'] + config['test_split']
    if abs(total_split - 1.0) > 1e-6:
        logger.error(f"Split ratios must sum to 1.0, got {total_split}")
        sys.exit(1)
    
    # Validate input directory
    if not os.path.exists(config['raw_data_dir']):
        logger.error(f"Raw data directory not found: {config['raw_data_dir']}")
        logger.info("Please create the directory structure:")
        logger.info(f"  {config['raw_data_dir']}/")
        logger.info(f"    ├── real/")
        logger.info(f"    └── fake/")
        sys.exit(1)
    
    # Check for real and fake subdirectories
    real_dir = os.path.join(config['raw_data_dir'], 'real')
    fake_dir = os.path.join(config['raw_data_dir'], 'fake')
    
    if not os.path.exists(real_dir):
        logger.warning(f"Real data directory not found: {real_dir}")
    if not os.path.exists(fake_dir):
        logger.warning(f"Fake data directory not found: {fake_dir}")
    
    if not os.path.exists(real_dir) and not os.path.exists(fake_dir):
        logger.error("Neither real nor fake data directories found!")
        sys.exit(1)
    
    try:
        # Initialize preprocessor
        preprocessor = DataPreprocessor(config)
        
        # Process data
        preprocessor.process_raw_data()
        
        logger.info("Data preprocessing completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Data preprocessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during data preprocessing: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main() 