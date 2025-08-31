import os
import sys
import django

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'finance_advisor.settings')
django.setup()

# Import User model after Django setup
from django.contrib.auth.models import User

def create_superuser():
    """Create a superuser with username 'admin'"""
    username = 'admin'
    email = 'admin@example.com'
    password = 'admin123'
    
    # Check if user already exists
    if User.objects.filter(username=username).exists():
        print(f"Superuser '{username}' already exists.")
        return
    
    # Create superuser
    User.objects.create_superuser(username=username, email=email, password=password)
    print(f"Superuser '{username}' created successfully!")
    print(f"Email: {email}")
    print(f"Password: {password}")
    print("\nYou can now log in to the admin panel at: http://127.0.0.1:8000/admin/")

if __name__ == "__main__":
    create_superuser()
