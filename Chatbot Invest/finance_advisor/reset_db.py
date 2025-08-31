import os
import sys
import django

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'finance_advisor.settings')
django.setup()

# Import models after Django setup
from django.contrib.auth.models import User
from advisor.models import FinancialProfile, InvestmentAdvice, ChatMessage
from django.db import connection
from django.contrib.sessions.models import Session
from django.contrib.admin.models import LogEntry

def reset_database():
    """Delete all data from the database including users and superusers"""
    print("Deleting all data from the database...")
    
    # Delete all chat messages
    chat_count = ChatMessage.objects.count()
    ChatMessage.objects.all().delete()
    print(f"Deleted {chat_count} chat messages")
    
    # Delete all investment advice
    advice_count = InvestmentAdvice.objects.count()
    InvestmentAdvice.objects.all().delete()
    print(f"Deleted {advice_count} investment advice entries")
    
    # Delete all financial profiles
    profile_count = FinancialProfile.objects.count()
    FinancialProfile.objects.all().delete()
    print(f"Deleted {profile_count} financial profiles")
    
    # Delete all sessions
    session_count = Session.objects.count()
    Session.objects.all().delete()
    print(f"Deleted {session_count} sessions")
    
    # Delete all admin log entries
    log_count = LogEntry.objects.count()
    LogEntry.objects.all().delete()
    print(f"Deleted {log_count} admin log entries")
    
    # Delete all users including superusers
    user_count = User.objects.count()
    User.objects.all().delete()
    print(f"Deleted {user_count} users (including superusers)")
    
    print("Database reset complete!")

if __name__ == "__main__":
    # Ask for confirmation
    confirm = input("This will delete ALL data including ALL users and superusers. Are you sure? (yes/no): ")
    if confirm.lower() == 'yes':
        reset_database()
    else:
        print("Database reset cancelled.")
