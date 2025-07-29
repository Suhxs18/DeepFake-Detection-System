#!/usr/bin/env python3
"""
VS Code Setup Verification Script
This script helps verify that VS Code is using the correct Python interpreter.
"""

import sys
import os

def check_imports():
    """Check if all required packages can be imported"""
    print("=" * 60)
    print("VS CODE SETUP VERIFICATION")
    print("=" * 60)
    
    # Check Python interpreter
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version: {sys.version}")
    
    # Check if we're in the virtual environment
    if "venv" in sys.executable:
        print("✅ Using virtual environment")
    else:
        print("❌ NOT using virtual environment")
        print("   Please select the interpreter from venv\\Scripts\\python.exe")
    
    print("\n" + "=" * 60)
    print("IMPORT TESTS")
    print("=" * 60)
    
    # Test imports
    imports_to_test = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
        ("cv2", "OpenCV"),
        ("facenet_pytorch", "FaceNet PyTorch"),
        ("sklearn", "Scikit-learn"),
        ("matplotlib", "Matplotlib")
    ]
    
    all_successful = True
    
    for module_name, display_name in imports_to_test:
        try:
            __import__(module_name)
            print(f"✅ {display_name} imported successfully")
        except ImportError as e:
            print(f"❌ {display_name} import failed: {e}")
            all_successful = False
    
    print("\n" + "=" * 60)
    print("PROJECT STRUCTURE")
    print("=" * 60)
    
    # Check project structure
    required_dirs = ["src", "scripts", "data", "models"]
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✅ {dir_name}/ directory exists")
        else:
            print(f"❌ {dir_name}/ directory missing")
    
    # Check key files
    key_files = [
        "src/utils.py",
        "src/model.py", 
        "src/dataset.py",
        "scripts/train.py",
        "scripts/predict.py",
        "requirements.txt"
    ]
    
    for file_path in key_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if all_successful:
        print("🎉 All imports successful! VS Code should work correctly.")
        print("\nTo fix import errors in VS Code:")
        print("1. Press Ctrl+Shift+P")
        print("2. Type 'Python: Select Interpreter'")
        print("3. Choose: venv\\Scripts\\python.exe")
        print("4. Reload VS Code window (Ctrl+Shift+P -> 'Developer: Reload Window')")
    else:
        print("❌ Some imports failed. Please check your virtual environment.")
        print("\nTo fix:")
        print("1. Activate virtual environment: venv\\Scripts\\activate")
        print("2. Install packages: pip install -r requirements.txt")
        print("3. Select correct interpreter in VS Code")

if __name__ == "__main__":
    check_imports() 