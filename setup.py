#!/usr/bin/env python3
"""
Setup script for DeepFake Detection System.

This script helps users set up the project environment and install dependencies.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(command: str, description: str) -> bool:
    """
    Run a shell command and handle errors.
    
    Args:
        command (str): Command to run
        description (str): Description of what the command does
        
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"  Error: {e.stderr}")
        return False

def check_python_version() -> bool:
    """
    Check if Python version is compatible.
    
    Returns:
        bool: True if compatible, False otherwise
    """
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"✗ Python 3.8+ is required. Current version: {version.major}.{version.minor}")
        return False
    print(f"✓ Python version {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def create_virtual_environment(env_name: str = "venv") -> bool:
    """
    Create a virtual environment.
    
    Args:
        env_name (str): Name of the virtual environment
        
    Returns:
        bool: True if successful, False otherwise
    """
    if os.path.exists(env_name):
        print(f"✓ Virtual environment '{env_name}' already exists")
        return True
    
    return run_command(f"python -m venv {env_name}", f"Creating virtual environment '{env_name}'")

def install_dependencies(env_name: str = "venv") -> bool:
    """
    Install project dependencies.
    
    Args:
        env_name (str): Name of the virtual environment
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Determine the pip command based on the OS
    if os.name == 'nt':  # Windows
        pip_cmd = f"{env_name}\\Scripts\\pip"
    else:  # Unix/Linux/macOS
        pip_cmd = f"{env_name}/bin/pip"
    
    # Upgrade pip first
    if not run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install dependencies
    return run_command(f"{pip_cmd} install -r requirements.txt", "Installing dependencies")

def create_project_structure() -> bool:
    """
    Create the project directory structure.
    
    Returns:
        bool: True if successful, False otherwise
    """
    print("\nCreating project structure...")
    
    directories = [
        "data/raw/real",
        "data/raw/fake",
        "data/processed_faces/train/real",
        "data/processed_faces/train/fake",
        "data/processed_faces/val/real",
        "data/processed_faces/val/fake",
        "data/processed_faces/test/real",
        "data/processed_faces/test/fake",
        "models",
        "logs"
    ]
    
    try:
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"  ✓ Created: {directory}")
        return True
    except Exception as e:
        print(f"✗ Failed to create directories: {e}")
        return False

def test_installation(env_name: str = "venv") -> bool:
    """
    Test the installation by running basic imports.
    
    Args:
        env_name (str): Name of the virtual environment
        
    Returns:
        bool: True if successful, False otherwise
    """
    print("\nTesting installation...")
    
    # Determine the python command based on the OS
    if os.name == 'nt':  # Windows
        python_cmd = f"{env_name}\\Scripts\\python"
    else:  # Unix/Linux/macOS
        python_cmd = f"{env_name}/bin/python"
    
    test_script = """
import sys
print("Testing imports...")

try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
except ImportError as e:
    print(f"✗ PyTorch import failed: {e}")
    sys.exit(1)

try:
    import torchvision
    print(f"✓ TorchVision {torchvision.__version__}")
except ImportError as e:
    print(f"✗ TorchVision import failed: {e}")
    sys.exit(1)

try:
    import cv2
    print(f"✓ OpenCV {cv2.__version__}")
except ImportError as e:
    print(f"✗ OpenCV import failed: {e}")
    sys.exit(1)

try:
    import facenet_pytorch
    print("✓ FaceNet-PyTorch")
except ImportError as e:
    print(f"✗ FaceNet-PyTorch import failed: {e}")
    sys.exit(1)

try:
    from src.utils import get_device
    device = get_device()
    print(f"✓ DeepFake Detection System modules")
    print(f"✓ Using device: {device}")
except ImportError as e:
    print(f"✗ DeepFake Detection System import failed: {e}")
    sys.exit(1)

print("✓ All imports successful!")
"""
    
    # Write test script to temporary file
    test_file = "test_installation.py"
    with open(test_file, "w") as f:
        f.write(test_script)
    
    try:
        result = subprocess.run(f"{python_cmd} {test_file}", shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        os.remove(test_file)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Installation test failed:")
        print(e.stdout)
        print(e.stderr)
        if os.path.exists(test_file):
            os.remove(test_file)
        return False

def print_activation_instructions(env_name: str = "venv"):
    """Print instructions for activating the virtual environment."""
    print("\n" + "="*60)
    print("SETUP COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    print("\nTo activate the virtual environment:")
    if os.name == 'nt':  # Windows
        print(f"  {env_name}\\Scripts\\activate")
    else:  # Unix/Linux/macOS
        print(f"  source {env_name}/bin/activate")
    
    print("\nNext steps:")
    print("1. Activate the virtual environment")
    print("2. Add your data to data/raw/real/ and data/raw/fake/")
    print("3. Run preprocessing: python scripts/preprocess_data.py")
    print("4. Train the model: python scripts/train.py")
    print("5. Make predictions: python scripts/predict.py --model models/best_model.pth --image your_image.jpg")
    
    print("\nFor more information, see the README.md file.")

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup DeepFake Detection System")
    parser.add_argument("--env-name", default="venv", help="Virtual environment name (default: venv)")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency installation")
    parser.add_argument("--skip-test", action="store_true", help="Skip installation test")
    
    args = parser.parse_args()
    
    print("DeepFake Detection System - Setup")
    print("="*40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create virtual environment
    if not create_virtual_environment(args.env_name):
        sys.exit(1)
    
    # Install dependencies
    if not args.skip_deps:
        if not install_dependencies(args.env_name):
            sys.exit(1)
    
    # Create project structure
    if not create_project_structure():
        sys.exit(1)
    
    # Test installation
    if not args.skip_test and not args.skip_deps:
        if not test_installation(args.env_name):
            print("\nInstallation test failed. Please check the error messages above.")
            sys.exit(1)
    
    # Print instructions
    print_activation_instructions(args.env_name)

if __name__ == "__main__":
    main() 