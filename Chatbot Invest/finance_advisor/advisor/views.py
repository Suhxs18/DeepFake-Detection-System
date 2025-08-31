from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login, authenticate, logout
from django.contrib import messages
from django.http import JsonResponse
from django.utils.safestring import mark_safe
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

from .models import FinancialProfile, InvestmentAdvice, ChatMessage
from .forms import UserRegistrationForm, UserLoginForm, FinancialProfileForm
from .utils import format_investment_advice, query_ollama

import json
import markdown
import bleach
import re
from datetime import datetime


def index(request):
    """Home page view - redirects to login if not authenticated"""
    if request.user.is_authenticated:
        return redirect('dashboard')
    return redirect('login')


@csrf_exempt
def register_view(request):
    """User registration view"""
    if request.user.is_authenticated:
        return redirect('dashboard')
        
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, 'Registration successful! Please complete your financial profile.')
            return redirect('create_profile')
    else:
        form = UserRegistrationForm()
    
    return render(request, 'advisor/register.html', {'form': form})


@csrf_exempt
def login_view(request):
    """User login view"""
    if request.user.is_authenticated:
        return redirect('dashboard')
        
    if request.method == 'POST':
        form = UserLoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(username=username, password=password)
            if user:
                login(request, user)
                return redirect('dashboard')
            else:
                messages.error(request, 'Invalid username or password')
    else:
        form = UserLoginForm()
    
    return render(request, 'advisor/login.html', {'form': form})


def logout_view(request):
    """User logout view"""
    logout(request)
    return redirect('login')


@login_required
def dashboard(request):
    """User dashboard view"""
    try:
        profile = FinancialProfile.objects.get(user=request.user)
        advice_list = InvestmentAdvice.objects.filter(user=request.user).order_by('-created_at')
        return render(request, 'advisor/dashboard.html', {
            'profile': profile,
            'advice_list': advice_list,
        })
    except FinancialProfile.DoesNotExist:
        messages.info(request, 'Please complete your financial profile to get investment advice.')
        return redirect('create_profile')


@login_required
@csrf_exempt
def create_profile(request):
    """Create financial profile view"""
    try:
        # Check if profile already exists
        profile = FinancialProfile.objects.get(user=request.user)
        messages.info(request, 'You already have a financial profile. You can update it instead.')
        return redirect('update_profile')
    except FinancialProfile.DoesNotExist:
        if request.method == 'POST':
            form = FinancialProfileForm(request.POST)
            if form.is_valid():
                profile = form.save(commit=False)
                profile.user = request.user
                
                # Calculate annual_income and savings from monthly values
                profile.annual_income = profile.monthly_income * 12
                profile.savings = profile.monthly_savings * 12
                
                profile.save()
                
                # Generate initial investment advice
                generate_investment_advice(request.user, profile)
                
                messages.success(request, 'Financial profile created successfully!')
                return redirect('dashboard')
        else:
            form = FinancialProfileForm()
        
        return render(request, 'advisor/profile_form.html', {
            'form': form,
            'action': 'Create',
        })


@login_required
@csrf_exempt
def update_profile(request):
    """Update financial profile view"""
    try:
        profile = FinancialProfile.objects.get(user=request.user)
        if request.method == 'POST':
            form = FinancialProfileForm(request.POST, instance=profile)
            if form.is_valid():
                form.save()
                messages.success(request, 'Financial profile updated successfully!')
                return redirect('dashboard')
        else:
            form = FinancialProfileForm(instance=profile)
        
        return render(request, 'advisor/profile_form.html', {
            'form': form,
            'action': 'Update',
        })
    except FinancialProfile.DoesNotExist:
        messages.error(request, 'Financial profile not found. Please create one first.')
        return redirect('create_profile')


@login_required
def view_profile(request):
    """View financial profile details"""
    try:
        profile = FinancialProfile.objects.get(user=request.user)
        return render(request, 'advisor/profile_detail.html', {'profile': profile})
    except FinancialProfile.DoesNotExist:
        messages.error(request, 'Financial profile not found. Please create one first.')
        return redirect('create_profile')


@login_required
def investment_advice(request):
    """Generate new investment advice"""
    try:
        profile = FinancialProfile.objects.get(user=request.user)
        advice = generate_investment_advice(request.user, profile)
        
        if advice:
            messages.success(request, 'New investment advice generated successfully!')
        else:
            messages.error(request, 'Failed to generate investment advice. Please try again later.')
            
        return redirect('view_advice', advice_id=advice.id)
    except FinancialProfile.DoesNotExist:
        messages.error(request, 'Financial profile not found. Please create one first.')
        return redirect('create_profile')


@login_required
def view_advice(request, advice_id):
    """View specific investment advice"""
    advice = get_object_or_404(InvestmentAdvice, id=advice_id, user=request.user)
    
    # Get chat history for this user
    chat_messages = ChatMessage.objects.filter(
        user=request.user
    ).order_by('timestamp')
    
    return render(request, 'advisor/advice_detail.html', {
        'advice': advice,
        'chat_messages': chat_messages,
    })


@login_required
def advice_history(request):
    """View all investment advice history"""
    advice_list = InvestmentAdvice.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'advisor/advice_history.html', {'advice_list': advice_list})


@login_required
@require_POST
@csrf_exempt
def chat_with_advisor(request):
    """API endpoint to chat with the AI financial advisor"""
    try:
        data = json.loads(request.body)
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return JsonResponse({'error': 'Message cannot be empty'}, status=400)
        
        # Get user's financial profile
        profile = FinancialProfile.objects.get(user=request.user)
        
        # Save user message
        ChatMessage.objects.create(
            user=request.user,
            message_type='user',
            content=user_message
        )
        
        # Get recent chat history to provide context and avoid repetition
        recent_messages = ChatMessage.objects.filter(user=request.user).order_by('-timestamp')[:10]
        chat_history = []
        for msg in reversed(list(recent_messages)):
            role = "user" if msg.message_type == 'user' else "assistant"
            # Strip HTML tags for context to avoid confusion
            import re
            content = re.sub(r'<[^>]*>', '', msg.content)
            chat_history.append({"role": role, "content": content})
        
        # Format currency values for context
        from .utils import format_currency
        # Use annual_income if available, otherwise calculate from monthly_income
        if hasattr(profile, 'annual_income') and profile.annual_income:
            annual_income = format_currency(float(profile.annual_income))
        else:
            annual_income = format_currency(float(profile.monthly_income * 12))
            
        # Use savings if available, otherwise use monthly_savings * 12 as an estimate
        if hasattr(profile, 'savings') and profile.savings:
            savings = format_currency(float(profile.savings))
        else:
            savings = format_currency(float(profile.monthly_savings * 12))
        
        # Format chat history as text
        chat_history_text = ""
        if chat_history:
            chat_history_text = "Recent conversation history:\n"
            for msg in chat_history:
                chat_history_text += f"{msg['role'].upper()}: {msg['content']}\n"
        
        # Create a chat prompt that includes context from the user's profile
        # Using the format similar to the Flask example
        chat_prompt = f"""
        Context: This user has submitted a financial profile with the following details:
        - Name: {profile.name}
        - Age: {profile.age}
        - Occupation: {profile.occupation}
        - Annual Income: {annual_income}
        - Savings: {savings}
        - Risk Tolerance: {profile.get_risk_tolerance_display()}
        - Investment Goal: {profile.get_investment_goal_display()}
        
        {chat_history_text}
        
        User asks: {user_message}
        
        Provide a helpful, accurate, and personalized response addressing their question about finances or investments. 
        Keep the response concise yet informative. Use clear formatting with bullet points or numbered lists if applicable.
        """
        
        # Query Ollama with the chat prompt
        from .utils import query_ollama
        response = query_ollama(chat_prompt)
        
        # Format the response for better readability
        from .utils import format_investment_advice
        formatted_response = format_investment_advice(response)
        
        # Strip the outer div tags to avoid nesting when the template applies |safe
        if formatted_response.startswith('<div class="advice-content">') and formatted_response.endswith('</div>'):
            # Remove the outer div tags but keep the inner content
            formatted_content = formatted_response[28:-6]  # 28 is the length of '<div class="advice-content">' and -6 for '</div>'
        else:
            formatted_content = formatted_response
        
        # Save advisor response - store the formatted HTML version without the wrapper div
        ChatMessage.objects.create(
            user=request.user,
            message_type='advisor',
            content=formatted_content
        )
        
        # Get current timestamp for display
        from datetime import datetime
        timestamp = datetime.now().isoformat()
        
        return JsonResponse({
            'response_html': formatted_content,
            'raw_response': response,
            'timestamp': timestamp
        })
    
    except FinancialProfile.DoesNotExist:
        return JsonResponse({'error': 'Financial profile not found'}, status=404)
    
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@login_required
@csrf_exempt
def get_chat_history(request):
    """Get the chat history for the user"""
    try:
        # Get user's chat messages
        chat_messages = ChatMessage.objects.filter(user=request.user).order_by('timestamp')
        
        history = []
        for msg in chat_messages:
            # For advisor messages, the content is already formatted HTML
            # For user messages, the content is plain text
            if msg.message_type == 'advisor':
                # The content is already HTML, just pass it through
                formatted_content = msg.content
            else:
                # User message is plain text
                formatted_content = msg.content
                
            history.append({
                'type': msg.message_type,
                'content': msg.content,
                'formatted_content': formatted_content,
                'timestamp': msg.timestamp.strftime('%Y-%m-%d %H:%M:%S')
            })
        
        return JsonResponse({'history': history})
    
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


def generate_investment_advice(user, profile):
    """Generate investment advice based on financial profile"""
    # Prepare profile data for the AI
    profile_data = {
        'name': profile.name,
        'age': profile.age,
        'occupation': profile.occupation,
        'annual_income': float(profile.annual_income) if hasattr(profile, 'annual_income') and profile.annual_income else float(profile.monthly_income * 12),
        'savings': float(profile.savings) if hasattr(profile, 'savings') and profile.savings else float(profile.monthly_savings * 12),
        'risk_tolerance': profile.get_risk_tolerance_display(),
        'investment_goal': profile.get_investment_goal_display() if hasattr(profile, 'get_investment_goal_display') else 'Other Goals',
    }
    
    # Get current date and time for more varied advice
    from datetime import datetime
    current_date = datetime.now().strftime("%B %d, %Y")
    current_time = datetime.now().strftime("%H:%M")
    
    # Get count of previous advice to reference
    previous_advice_count = InvestmentAdvice.objects.filter(user=user).count()
    
    # Generate a focus area based on user's age and goals
    focus_areas = {
        'retirement': ['long-term wealth building', 'retirement planning', 'pension optimization'],
        'wealth': ['wealth accumulation', 'portfolio diversification', 'high-growth investments'],
        'education': ['education funding', 'systematic investment for education', 'tax-efficient education savings'],
        'home': ['real estate investment', 'mortgage planning', 'down payment strategies'],
        'other': ['financial independence', 'passive income streams', 'balanced portfolio management']
    }
    
    # Select a focus area based on the user's investment goal
    import random
    goal_key = profile.investment_goal.lower()
    focus_area = random.choice(focus_areas.get(goal_key, focus_areas['other']))
    
    # Select a market perspective based on the current time (just to add variety)
    market_perspectives = [
        "In the current market conditions",
        "Given recent economic trends",
        "With today's market volatility",
        "Considering the present economic climate",
        "In light of recent market developments"
    ]
    market_perspective = random.choice(market_perspectives)
    
    # Use the base format_prompt function and then enhance it with additional context
    from .utils import format_prompt
    base_prompt = format_prompt(profile_data)
    
    # Enhance the prompt with more specific instructions for variety
    enhanced_prompt = f"""
    {base_prompt}
    
    Additional Context:
    - Current Date: {current_date}
    - Current Time: {current_time}
    - Market Context: {market_perspective}
    - Focus Area: {focus_area}
    - Advice Count: {'First advice' if previous_advice_count == 0 else f'Advice #{previous_advice_count+1}'}
    
    Please provide:
    1. A personalized investment strategy with specific focus on {focus_area}
    2. Recommended investments based on their risk tolerance and current market conditions
    3. Expected returns and timeframes
    4. At least one unique recommendation you haven't given before
    
    Format your response with clear headings and bullet points.
    """
    
    # Query Ollama for advice
    from .utils import query_ollama
    raw_advice_content = query_ollama(enhanced_prompt)
    
    # Format the advice content using our utility function
    from .utils import format_investment_advice
    formatted_content = format_investment_advice(raw_advice_content)
    
    # Strip the outer div tags to avoid nesting when the template applies |safe
    # This is important because the template already has a div with class="advice-content"
    if formatted_content.startswith('<div class="advice-content">') and formatted_content.endswith('</div>'):
        # Remove the outer div tags but keep the inner content
        formatted_content = formatted_content[28:-6]  # 28 is the length of '<div class="advice-content">' and -6 for '</div>'
    
    # Create a more descriptive title that includes the focus area
    title = f"Investment Advice: {focus_area.title()} for {profile.name}"
    
    # Save the advice with the formatted content
    advice = InvestmentAdvice.objects.create(
        user=user,
        profile=profile,
        title=title,
        content=formatted_content
    )
    
    return advice


def page_not_found(request, exception):
    """404 error handler"""
    return render(request, 'advisor/404.html', status=404)


def server_error(request):
    """500 error handler"""
    return render(request, 'advisor/500.html', status=500)
