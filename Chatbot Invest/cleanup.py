#!/usr/bin/env python
"""
Script to clean up unnecessary files in the Finance Advisor project
"""
import os
import shutil
import sys

def cleanup_project():
    """Remove unnecessary files to save space"""
    
    # Get the base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Files and directories to delete
    to_delete = [
        # Original Flask application
        os.path.join(base_dir, "r"),
        
        # Temporary utility scripts
        os.path.join(base_dir, "fix_old_responses.py"),
        
        # __pycache__ directories (will find and delete all of them)
    ]
    
    # Find all __pycache__ directories
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            if dir_name == "__pycache__":
                to_delete.append(os.path.join(root, dir_name))
    
    # Confirm with the user
    print("The following files and directories will be deleted:")
    for item in to_delete:
        if os.path.exists(item):
            print(f" - {item}")
    
    if len(sys.argv) > 1 and sys.argv[1] == "--force":
        confirm = "yes"
    else:
        confirm = input("\nAre you sure you want to delete these files? (yes/no): ")
    
    if confirm.lower() == "yes":
        # Delete the files
        for item in to_delete:
            if os.path.exists(item):
                try:
                    if os.path.isdir(item):
                        shutil.rmtree(item)
                        print(f"Deleted directory: {item}")
                    else:
                        os.remove(item)
                        print(f"Deleted file: {item}")
                except Exception as e:
                    print(f"Error deleting {item}: {e}")
        
        print("\nCleanup completed successfully!")
    else:
        print("Operation cancelled.")

if __name__ == "__main__":
    cleanup_project()
