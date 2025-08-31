#!/usr/bin/env python
"""
Script to fix formatting in existing investment advice and chat messages
This will remove asterisks and properly format all existing content
"""
import os
import django
import re

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'finance_advisor.settings')
django.setup()

from advisor.models import InvestmentAdvice, ChatMessage
from advisor.utils import format_investment_advice

def fix_investment_advice():
    """Fix formatting in all existing investment advice"""
    advice_count = 0
    
    # Get all investment advice
    all_advice = InvestmentAdvice.objects.all()
    print(f"Found {len(all_advice)} investment advice records to process")
    
    for advice in all_advice:
        original_content = advice.content
        
        # Apply our improved formatting function
        # We need to extract the raw content from the HTML first
        raw_content = original_content
        
        # Process the content with our improved formatter
        formatted_content = format_investment_advice(raw_content)
        
        # Save the formatted content back to the database
        advice.content = formatted_content
        advice.save()
        advice_count += 1
        
    print(f"Successfully updated {advice_count} investment advice records")

def fix_chat_messages():
    """Fix formatting in all existing chat messages"""
    message_count = 0
    
    # Get all advisor chat messages
    advisor_messages = ChatMessage.objects.filter(message_type='advisor')
    print(f"Found {len(advisor_messages)} advisor messages to process")
    
    for message in advisor_messages:
        original_content = message.content
        
        # Apply our improved formatting function
        # We need to extract the raw content from the HTML first
        raw_content = original_content
        
        # Process the content with our improved formatter
        formatted_content = format_investment_advice(raw_content)
        
        # Save the formatted content back to the database
        message.content = formatted_content
        message.save()
        message_count += 1
        
    print(f"Successfully updated {message_count} chat messages")

if __name__ == "__main__":
    print("Starting to fix old responses...")
    fix_investment_advice()
    fix_chat_messages()
    print("All responses have been fixed!")
