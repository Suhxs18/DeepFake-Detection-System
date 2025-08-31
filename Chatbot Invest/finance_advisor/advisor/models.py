from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
from django.core.validators import MinValueValidator, MaxValueValidator
from decimal import Decimal

class FinancialProfile(models.Model):
    """Financial profile of a user with investment preferences and goals"""
    
    # Risk tolerance choices
    RISK_CHOICES = [
        ('1', 'Very Conservative'),
        ('2', 'Conservative'),
        ('3', 'Neutral'),
        ('4', 'Aggressive'),
        ('5', 'Extremely Aggressive'),
    ]
    
    # Investment goal choices
    INVESTMENT_GOAL_CHOICES = [
        ('retirement', 'Retirement Planning'),
        ('wealth', 'Wealth Building'),
        ('education', 'Education Funding'),
        ('home', 'Home Purchase'),
        ('other', 'Other Goals'),
    ]
    
    # Knowledge level choices
    KNOWLEDGE_CHOICES = [
        ('1', 'Beginner'),
        ('2', 'Intermediate'),
        ('3', 'Advanced'),
        ('4', 'Expert'),
        ('5', 'Extremely Expert'),
    ]
    
    # User relationship
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='financial_profile')
    
    # Personal information
    name = models.CharField(max_length=100)
    age = models.PositiveIntegerField(
        validators=[
            MinValueValidator(18, message="You must be at least 18 years old"),
            MaxValueValidator(100, message="Please enter a valid age")
        ]
    )
    occupation = models.CharField(max_length=100)
    family_size = models.PositiveIntegerField(
        default=1,
        validators=[MinValueValidator(1, message="Family size must be at least 1")]
    )
    
    # Financial information
    monthly_income = models.DecimalField(
        max_digits=12,
        decimal_places=2,
        default=0,
        validators=[MinValueValidator(0, message="Monthly income cannot be negative")]
    )
    annual_income = models.DecimalField(
        max_digits=12,
        decimal_places=2,
        default=0,
        validators=[MinValueValidator(0, message="Annual income cannot be negative")],
        blank=True
    )
    monthly_expenses = models.DecimalField(
        max_digits=12,
        decimal_places=2,
        default=0,
        validators=[MinValueValidator(0, message="Monthly expenses cannot be negative")]
    )
    monthly_savings = models.DecimalField(
        max_digits=12,
        decimal_places=2,
        default=0,
        validators=[MinValueValidator(0, message="Monthly savings cannot be negative")]
    )
    current_debts = models.DecimalField(
        max_digits=12,
        decimal_places=2,
        default=0,
        validators=[MinValueValidator(0, message="Current debts cannot be negative")]
    )
    savings = models.DecimalField(
        max_digits=12,
        decimal_places=2,
        default=0,
        validators=[MinValueValidator(0, message="Savings cannot be negative")],
        blank=True
    )
    debt_interest_rate = models.DecimalField(
        max_digits=5,
        decimal_places=2,
        default=0,
        validators=[
            MinValueValidator(0, message="Debt interest rate cannot be negative"),
            MaxValueValidator(100, message="Debt interest rate cannot exceed 100%")
        ]
    )
    
    # Investment preferences
    risk_tolerance = models.CharField(max_length=1, choices=RISK_CHOICES)
    investment_goal = models.CharField(max_length=20, choices=INVESTMENT_GOAL_CHOICES, default='other')
    investment_knowledge = models.CharField(max_length=1, choices=KNOWLEDGE_CHOICES, default='3')
    has_investment_experience = models.BooleanField(default=False)
    previous_investments = models.TextField(blank=True)
    
    # Short-term goals (1-3 years)
    short_term_goals = models.TextField(blank=True, default='')
    short_term_goal_amount = models.DecimalField(
        max_digits=12,
        decimal_places=2,
        default=0,
        validators=[MinValueValidator(0, message="Goal amount cannot be negative")]
    )
    
    # Medium-term goals (5-10 years)
    medium_term_goals = models.TextField(blank=True, default='')
    medium_term_goal_amount = models.DecimalField(
        max_digits=12,
        decimal_places=2,
        default=0,
        validators=[MinValueValidator(0, message="Goal amount cannot be negative")]
    )
    
    # Long-term goals (10+ years)
    long_term_goals = models.TextField(blank=True, default='')
    long_term_goal_amount = models.DecimalField(
        max_digits=12,
        decimal_places=2,
        default=0,
        validators=[MinValueValidator(0, message="Goal amount cannot be negative")]
    )
    
    # Additional assets
    other_assets = models.TextField(blank=True)
    retirement_plans = models.TextField(blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.user.username}'s Financial Profile"
    
    def clean(self):
        """Additional validation at the model level"""
        from django.core.exceptions import ValidationError
        
        # Validate monthly income and expenses
        if self.monthly_expenses > self.monthly_income:
            raise ValidationError('Monthly expenses cannot be greater than monthly income')
        
        # Validate monthly savings
        expected_savings = self.monthly_income - self.monthly_expenses
        if self.monthly_savings > expected_savings:
            raise ValidationError('Monthly savings cannot be greater than (income - expenses)')
    
    def save(self, *args, **kwargs):
        """Override save to ensure validation"""
        # Calculate annual income if not set
        if not self.annual_income and self.monthly_income:
            self.annual_income = self.monthly_income * 12
            
        # Calculate monthly income if not set
        if not self.monthly_income and self.annual_income:
            self.monthly_income = self.annual_income / 12
            
        # Set savings based on monthly savings if not explicitly set
        if not self.savings and hasattr(self, 'monthly_savings'):
            self.savings = self.monthly_savings * 12
            
        self.full_clean()
        super().save(*args, **kwargs)


class InvestmentAdvice(models.Model):
    """Investment advice generated for a user's financial profile"""
    
    # Relationships
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='investment_advice')
    profile = models.ForeignKey(FinancialProfile, on_delete=models.CASCADE, related_name='advice')
    
    # Advice content
    title = models.CharField(max_length=200)
    content = models.TextField()
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Investment Advice for {self.user.username}"
    
    class Meta:
        ordering = ['-created_at']


class ChatMessage(models.Model):
    """Chat messages between user and AI advisor"""
    
    # Message types
    MESSAGE_TYPES = [
        ('user', 'User Message'),
        ('advisor', 'Advisor Message'),
    ]
    
    # Relationships
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='chat_messages')
    
    # Message content
    message_type = models.CharField(max_length=10, choices=MESSAGE_TYPES)
    content = models.TextField()
    
    # Timestamps
    timestamp = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['timestamp']
    
    def __str__(self):
        return f"{self.message_type.capitalize()} message"
