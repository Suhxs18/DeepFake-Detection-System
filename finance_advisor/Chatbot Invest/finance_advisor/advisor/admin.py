from django.contrib import admin
from .models import FinancialProfile, InvestmentAdvice, ChatMessage

@admin.register(FinancialProfile)
class FinancialProfileAdmin(admin.ModelAdmin):
    list_display = ('name', 'age', 'occupation', 'monthly_income', 'risk_tolerance', 'created_at')
    list_filter = ('risk_tolerance', 'investment_knowledge')
    search_fields = ('name', 'occupation')
    readonly_fields = ('created_at', 'updated_at')
    fieldsets = (
        ('Personal Information', {
            'fields': ('name', 'age', 'occupation', 'family_size')
        }),
        ('Financial Information', {
            'fields': ('monthly_income', 'monthly_expenses', 'monthly_savings', 'current_debts', 'debt_interest_rate')
        }),
        ('Investment Profile', {
            'fields': ('risk_tolerance', 'investment_knowledge', 'has_investment_experience', 'previous_investments')
        }),
        ('Financial Goals', {
            'fields': (
                'short_term_goals', 'short_term_goal_amount',
                'medium_term_goals', 'medium_term_goal_amount',
                'long_term_goals', 'long_term_goal_amount'
            )
        }),
        ('Additional Information', {
            'fields': ('other_assets', 'retirement_plans')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        })
    )

@admin.register(InvestmentAdvice)
class InvestmentAdviceAdmin(admin.ModelAdmin):
    list_display = ('title', 'user', 'created_at')
    list_filter = ('created_at',)
    search_fields = ('title', 'content')
    readonly_fields = ('created_at',)

@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ('user', 'message_type', 'timestamp')
    list_filter = ('message_type', 'timestamp')
    search_fields = ('content',)
    readonly_fields = ('timestamp',)
