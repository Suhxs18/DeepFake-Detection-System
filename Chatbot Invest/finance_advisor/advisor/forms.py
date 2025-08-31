from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from .models import FinancialProfile
from decimal import Decimal


class UserRegistrationForm(UserCreationForm):
    """Form for user registration"""
    email = forms.EmailField(required=True)
    
    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2']
    
    def clean_email(self):
        email = self.cleaned_data.get('email')
        if User.objects.filter(email=email).exists():
            raise forms.ValidationError('Email already in use')
        return email


class UserLoginForm(forms.Form):
    """Form for user login"""
    username = forms.CharField(max_length=150)
    password = forms.CharField(widget=forms.PasswordInput)


class FinancialProfileForm(forms.ModelForm):
    """Form for collecting detailed financial information"""
    
    class Meta:
        model = FinancialProfile
        exclude = ['user', 'created_at', 'updated_at', 'annual_income', 'savings']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control', 'required': True}),
            'age': forms.NumberInput(attrs={'class': 'form-control', 'min': '18', 'max': '100', 'required': True}),
            'occupation': forms.TextInput(attrs={'class': 'form-control', 'required': True}),
            'family_size': forms.NumberInput(attrs={'class': 'form-control', 'min': '1', 'required': True}),
            'monthly_income': forms.NumberInput(attrs={'class': 'form-control', 'min': '0', 'step': '0.01', 'required': True}),
            'monthly_expenses': forms.NumberInput(attrs={'class': 'form-control', 'min': '0', 'step': '0.01', 'required': True}),
            'monthly_savings': forms.NumberInput(attrs={'class': 'form-control', 'min': '0', 'step': '0.01', 'required': True}),
            'current_debts': forms.NumberInput(attrs={'class': 'form-control', 'min': '0', 'step': '0.01', 'required': True}),
            'debt_interest_rate': forms.NumberInput(attrs={'class': 'form-control', 'min': '0', 'max': '100', 'step': '0.01', 'required': True}),
            'risk_tolerance': forms.Select(attrs={'class': 'form-control', 'required': True}),
            'investment_goal': forms.Select(attrs={'class': 'form-control', 'required': True}),
            'investment_knowledge': forms.Select(attrs={'class': 'form-control', 'required': True}),
            'has_investment_experience': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'previous_investments': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
            'short_term_goals': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
            'short_term_goal_amount': forms.NumberInput(attrs={'class': 'form-control', 'min': '0', 'step': '0.01', 'required': True}),
            'medium_term_goals': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
            'medium_term_goal_amount': forms.NumberInput(attrs={'class': 'form-control', 'min': '0', 'step': '0.01', 'required': True}),
            'long_term_goals': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
            'long_term_goal_amount': forms.NumberInput(attrs={'class': 'form-control', 'min': '0', 'step': '0.01', 'required': True}),
            'other_assets': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
            'retirement_plans': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add help text to fields
        self.fields['risk_tolerance'].help_text = '1 = Very Conservative, 5 = Extremely Aggressive'
        self.fields['investment_knowledge'].help_text = '1 = Beginner, 5 = Extremely Expert'
        self.fields['previous_investments'].help_text = 'Describe any previous investment experience'
        self.fields['other_assets'].help_text = 'List any other assets (property, vehicles, etc.)'
        self.fields['retirement_plans'].help_text = 'Describe any existing retirement plans or accounts'
        
        # Add required field indicators
        for field_name, field in self.fields.items():
            if field.required:
                field.label = f"{field.label} *"
    
    def clean(self):
        cleaned_data = super().clean()
        
        # Validate monthly income and expenses
        monthly_income = cleaned_data.get('monthly_income')
        monthly_expenses = cleaned_data.get('monthly_expenses')
        monthly_savings = cleaned_data.get('monthly_savings')
        
        if monthly_income and monthly_expenses:
            if monthly_expenses > monthly_income:
                raise forms.ValidationError('Monthly expenses cannot be greater than monthly income')
            
            # Calculate expected savings
            expected_savings = monthly_income - monthly_expenses
            if monthly_savings and monthly_savings > expected_savings:
                raise forms.ValidationError('Monthly savings cannot be greater than (income - expenses)')
        
        # Validate goal amounts
        short_term_amount = cleaned_data.get('short_term_goal_amount')
        medium_term_amount = cleaned_data.get('medium_term_goal_amount')
        long_term_amount = cleaned_data.get('long_term_goal_amount')
        
        if short_term_amount and short_term_amount < 0:
            raise forms.ValidationError('Short-term goal amount cannot be negative')
        if medium_term_amount and medium_term_amount < 0:
            raise forms.ValidationError('Medium-term goal amount cannot be negative')
        if long_term_amount and long_term_amount < 0:
            raise forms.ValidationError('Long-term goal amount cannot be negative')
        
        # Validate debt interest rate
        debt_interest_rate = cleaned_data.get('debt_interest_rate')
        if debt_interest_rate and (debt_interest_rate < 0 or debt_interest_rate > 100):
            raise forms.ValidationError('Debt interest rate must be between 0 and 100')
        
        return cleaned_data
    
    def clean_age(self):
        age = self.cleaned_data.get('age')
        if age < 18:
            raise forms.ValidationError('You must be at least 18 years old')
        if age > 100:
            raise forms.ValidationError('Please enter a valid age')
        return age
    
    def clean_family_size(self):
        family_size = self.cleaned_data.get('family_size')
        if family_size < 1:
            raise forms.ValidationError('Family size must be at least 1')
        return family_size
