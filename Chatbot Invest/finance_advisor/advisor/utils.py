import requests
import re
import markdown
import bleach
from django.utils.safestring import mark_safe
from django.conf import settings

def format_prompt(user_data):
    """
    Format a prompt for the Ollama API based on user financial data
    This is similar to the Flask example but adapted for the Django model structure
    """
    return f"""
    Provide financial advice for a person with the following details:
    - Name: {user_data.get('name')}
    - Age: {user_data.get('age')}
    - Occupation: {user_data.get('occupation')}
    - Annual Income: {user_data.get('annual_income')}
    - Savings: {user_data.get('savings')}
    - Risk Tolerance: {user_data.get('risk_tolerance')}
    - Investment Goal: {user_data.get('investment_goal')}
    """

def query_ollama(prompt):
    """
    Query the Ollama API with a prompt and return the response
    If Ollama is not available, generate mock advice instead
    """
    url = settings.OLLAMA_URL
    payload = {
        "model": settings.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    
    # Print debug information
    print(f"Attempting to connect to Ollama at {url} with model {settings.OLLAMA_MODEL}")
    
    try:
        # Try to connect to Ollama with an extended timeout (1 minute)
        response = requests.post(url, json=payload, timeout=60)  # Extended timeout to 1 minute for model loading
        response.raise_for_status()
        result = response.json().get("response", "")
        if result:
            print("Successfully received response from Ollama")
            return result
        else:
            print("No response key in Ollama output, using mock advice")
            return generate_mock_investment_advice(prompt)
            
    except requests.exceptions.ConnectionError as e:
        # Return mock advice instead of error message
        error_msg = f"Connection error when trying to reach Ollama: {e}"
        print(error_msg)
        print("Using mock investment advice instead")
        return generate_mock_investment_advice(prompt)
        
    except requests.exceptions.Timeout:
        error_msg = "Request to Ollama timed out"
        print(error_msg)
        print("Using mock investment advice instead")
        return generate_mock_investment_advice(prompt)
        
    except requests.exceptions.HTTPError as e:
        # Return mock advice for HTTP errors
        error_msg = f"HTTP error when trying to reach Ollama: {e}"
        print(error_msg)
        print("Using mock investment advice instead")
        return generate_mock_investment_advice(prompt)
        
    except Exception as e:
        error_msg = f"Unexpected error when calling Ollama: {e}"
        print(error_msg)
        print("Using mock investment advice instead")
        return generate_mock_investment_advice(prompt)


def format_investment_advice(raw_response):
    """
    Format the Ollama response for better readability and modern display
    Enhanced version based on the Flask example
    """
    # Step 1: Handle any escaped HTML or nested advice-content divs
    if '&lt;div class="advice-content"&gt;' in raw_response:
        # This is already escaped HTML, unescape it first
        import html
        raw_response = html.unescape(raw_response)
    
    # Check if the response already has our wrapper div and remove it to prevent nesting
    if '<div class="advice-content">' in raw_response:
        # Extract the content from inside the div
        match = re.search(r'<div class="advice-content">(.*?)</div>', raw_response, re.DOTALL)
        if match:
            raw_response = match.group(1)
    
    # Step 2: Handle bold text formatting
    formatted = re.sub(r'(\*\*.*?\*\*)', r'\1\n', raw_response)
    
    # Step 3: Add paragraph breaks for numbered items
    formatted = re.sub(r'(\d+\.\s)', r'\n\1', formatted)
    
    # Step 4: Convert Markdown to HTML for rich formatting
    html_content = markdown.markdown(formatted, extensions=['tables', 'nl2br'])
    
    # Step 5: Sanitize HTML to prevent XSS
    allowed_tags = ['p', 'h1', 'h2', 'h3', 'h4', 'b', 'i', 'strong', 'em', 
                    'ul', 'ol', 'li', 'br', 'table', 'thead', 'tbody', 
                    'tr', 'th', 'td', 'hr', 'div', 'span']
    allowed_attrs = {'*': ['class', 'style']}
    clean_html = bleach.clean(html_content, tags=allowed_tags, attributes=allowed_attrs)
    
    # Step 6: Add custom styling classes for better display
    clean_html = clean_html.replace('<h1>', '<h1 class="advice-heading">')
    clean_html = clean_html.replace('<h2>', '<h2 class="advice-heading">')
    clean_html = clean_html.replace('<h3>', '<h3 class="advice-heading">')
    clean_html = clean_html.replace('<p>', '<p class="advice-paragraph">')
    clean_html = clean_html.replace('<ul>', '<ul class="advice-list" style="margin-bottom: 16px;">')
    clean_html = clean_html.replace('<ol>', '<ol class="advice-list" style="margin-bottom: 16px;">')
    
    # Step 7: Wrap the entire content in a div for styling
    final_html = f'<div class="advice-content">{clean_html}</div>'
    
    return mark_safe(final_html)


def format_currency(value, include_rupee_symbol=True):
    """
    Format numerical values into Indian currency format (crore, lakh)
    """
    if value is None or value == '':
        return ''
        
    try:
        value = float(value)
    except (ValueError, TypeError):
        return str(value)
        
    # Format as crore for values >= 10,000,000
    if value >= 10000000:
        formatted = f"{value/10000000:.2f} crore"
    # Format as lakh for values >= 100,000
    elif value >= 100000:
        formatted = f"{value/100000:.2f} lakh"
    # Format as regular number for smaller values
    else:
        formatted = f"{value:,.2f}"
        
    if include_rupee_symbol:
        return f"₹{formatted}"
    return formatted


def parse_currency_value(value_str):
    """
    Parse currency values with Indian notations (cr/crore, lakh, etc.)
    Returns the value in rupees (float)
    """
    if not value_str or isinstance(value_str, (int, float)):
        return value_str
        
    value_str = str(value_str).strip().lower()
    
    # Handle crore notation
    if 'cr' in value_str or 'crore' in value_str:
        value_str = value_str.replace('cr', '').replace('crore', '').strip()
        try:
            return float(value_str) * 10000000  # 1 crore = 10,000,000
        except ValueError:
            return value_str
            
    # Handle lakh notation
    if 'l' in value_str or 'lakh' in value_str:
        value_str = value_str.replace('l', '').replace('lakh', '').strip()
        try:
            return float(value_str) * 100000  # 1 lakh = 100,000
        except ValueError:
            return value_str
            
    # Return as is if no special notation found
    try:
        return float(value_str)
    except ValueError:
        return value_str


def generate_mock_investment_advice(prompt):
    """
    Generate dynamic mock investment advice based on the prompt
    Used when Ollama is not available
    """
    import random
    from datetime import datetime
    
    # Extract information from the prompt
    prompt_lower = prompt.lower()
    
    # Extract risk profile from prompt
    risk_profile = "moderate"
    if "conservative" in prompt_lower or "low risk" in prompt_lower:
        risk_profile = "conservative"
    elif "aggressive" in prompt_lower or "high risk" in prompt_lower:
        risk_profile = "aggressive"
    
    # Extract financial goals or topics from the prompt
    topics = {
        "retirement": ["retirement planning", "pension", "retirement fund", "retire"],
        "stocks": ["stock", "equity", "shares", "market"],
        "mutual_funds": ["mutual fund", "sip", "systematic"],
        "real_estate": ["real estate", "property", "home", "house"],
        "tax": ["tax", "taxation", "tax-saving", "tax benefit"],
        "debt": ["debt", "bond", "fixed income", "deposit"],
        "gold": ["gold", "precious metal", "commodity"],
        "international": ["international", "global", "foreign", "overseas"],
        "crypto": ["crypto", "bitcoin", "ethereum", "blockchain"],
        "emergency": ["emergency", "contingency", "rainy day", "liquid"],
    }
    
    # Determine which topics are mentioned in the prompt
    mentioned_topics = []
    for topic, keywords in topics.items():
        if any(keyword in prompt_lower for keyword in keywords):
            mentioned_topics.append(topic)
    
    # If no specific topics are mentioned, pick random ones based on risk profile
    if not mentioned_topics:
        if risk_profile == "conservative":
            potential_topics = ["retirement", "debt", "emergency", "tax"]
        elif risk_profile == "aggressive":
            potential_topics = ["stocks", "international", "real_estate", "crypto"]
        else:  # moderate
            potential_topics = ["mutual_funds", "stocks", "debt", "gold"]
        
        # Pick 1-2 random topics
        mentioned_topics = random.sample(potential_topics, min(2, len(potential_topics)))
    
    # Check if this is a question or a request for advice
    is_question = "?" in prompt or any(q in prompt_lower for q in ["what", "how", "why", "when", "where", "which", "can", "should"])
    
    # Current date and time for more varied responses
    current_date = datetime.now().strftime("%B %d, %Y")
    
    # Generate a unique response identifier to ensure variety
    response_id = random.randint(1000, 9999)
    
    # Create dynamic responses based on topics and question type
    if is_question:
        # Generate a response to a specific question
        response_parts = []
        
        # Add appropriate greeting and introduction
        response_parts.append(f"<h3 class=\"advice-heading\">Financial Advice - {current_date}</h3>")
        response_parts.append(f"<p class=\"advice-paragraph\">Thank you for your question about {', '.join(mentioned_topics)}. Here's my perspective:</p>")
        
        # Generate specific advice for each mentioned topic
        for topic in mentioned_topics:
            if topic == "retirement":
                response_parts.append(f"<h4>Retirement Planning</h4>")
                if risk_profile == "conservative":
                    response_parts.append("<p>For retirement with a conservative approach, consider allocating 70% to fixed income instruments like government securities and high-rated bonds. The remaining 30% can be in large-cap equity funds for long-term growth.</p>")
                elif risk_profile == "aggressive":
                    response_parts.append("<p>With an aggressive risk profile, you might consider a 70-30 split favoring equity investments for retirement. Focus on a mix of mid-cap and large-cap funds, with some international exposure for diversification.</p>")
                else:  # moderate
                    response_parts.append("<p>For a balanced retirement approach, consider a 50-50 split between equity and debt. Index funds can form the core of your equity portfolio, while corporate bonds can provide stability.</p>")
            
            elif topic == "stocks":
                response_parts.append(f"<h4>Stock Market Investments</h4>")
                if risk_profile == "conservative":
                    response_parts.append("<p>Given your conservative profile, limit direct stock exposure to 20% of your portfolio. Focus on dividend-yielding blue-chip companies with stable earnings history.</p>")
                elif risk_profile == "aggressive":
                    response_parts.append("<p>With your aggressive approach, you can allocate up to 70% in stocks. Consider a mix of established companies and growth stocks in emerging sectors like renewable energy and technology.</p>")
                else:  # moderate
                    response_parts.append("<p>For a moderate investor, a 40-50% allocation to stocks is appropriate. Focus on a diversified portfolio of large-caps with some mid-cap exposure for growth potential.</p>")
            
            elif topic == "mutual_funds":
                response_parts.append(f"<h4>Mutual Fund Strategy</h4>")
                if risk_profile == "conservative":
                    response_parts.append("<p>Consider debt-oriented hybrid funds and large-cap funds with a track record of stable returns. Liquid funds are good for short-term goals.</p>")
                elif risk_profile == "aggressive":
                    response_parts.append("<p>Look at sectoral funds, mid and small-cap funds, and international funds for higher growth potential. Maintain systematic investment plans (SIPs) to average out market volatility.</p>")
                else:  # moderate
                    response_parts.append("<p>Balanced advantage funds and multi-cap funds can form the core of your portfolio. Consider index funds for cost-effective market exposure.</p>")
            
            elif topic == "real_estate":
                response_parts.append(f"<h4>Real Estate Investments</h4>")
                response_parts.append("<p>REITs (Real Estate Investment Trusts) offer a liquid way to invest in real estate with lower capital requirements than direct property purchases. They typically offer dividend yields of 3-5%.</p>")
                if risk_profile != "conservative":
                    response_parts.append("<p>For physical real estate, consider residential properties in growing tier-2 cities for better rental yields compared to metropolitan areas.</p>")
            
            elif topic == "tax":
                response_parts.append(f"<h4>Tax-Efficient Investing</h4>")
                response_parts.append("<p>ELSS (Equity Linked Saving Schemes) offer tax deductions under Section 80C with a relatively short lock-in period of 3 years compared to other tax-saving instruments.</p>")
                response_parts.append("<p>Consider debt funds held for over 3 years for indexation benefits, which can significantly reduce your tax liability compared to fixed deposits.</p>")
            
            elif topic == "debt":
                response_parts.append(f"<h4>Fixed Income Strategy</h4>")
                if risk_profile == "conservative":
                    response_parts.append("<p>Focus on government securities, AAA-rated bonds, and banking & PSU debt funds for safety. Ladder your fixed deposits to manage interest rate risk.</p>")
                else:
                    response_parts.append("<p>Consider corporate bond funds and strategic debt funds for potentially higher yields. Keep an eye on credit quality and duration based on interest rate outlook.</p>")
            
            elif topic == "gold":
                response_parts.append(f"<h4>Gold Investments</h4>")
                response_parts.append("<p>Sovereign Gold Bonds offer the dual benefit of gold price appreciation and a fixed interest rate of 2.5% per annum. They're more tax-efficient than physical gold.</p>")
                response_parts.append("<p>Limit gold allocation to 5-15% of your portfolio as a hedge against inflation and market volatility.</p>")
            
            elif topic == "international":
                response_parts.append(f"<h4>International Diversification</h4>")
                response_parts.append("<p>Consider funds that invest in US markets for exposure to global technology giants not available in Indian markets.</p>")
                if risk_profile != "conservative":
                    response_parts.append("<p>Emerging market funds can offer higher growth potential but come with additional currency and geopolitical risks.</p>")
            
            elif topic == "crypto":
                response_parts.append(f"<h4>Cryptocurrency Considerations</h4>")
                response_parts.append("<p>Cryptocurrencies are highly volatile and speculative. If you're interested, limit exposure to 1-5% of your portfolio based on your risk tolerance.</p>")
                response_parts.append("<p>Consider dollar-cost averaging rather than lump-sum investments given the high volatility.</p>")
            
            elif topic == "emergency":
                response_parts.append(f"<h4>Emergency Fund Strategy</h4>")
                response_parts.append("<p>Maintain 6-12 months of expenses in highly liquid instruments like savings accounts and liquid funds.</p>")
                response_parts.append("<p>Consider a sweep-in fixed deposit linked to your savings account for better interest rates while maintaining liquidity.</p>")
        
        # Add a unique recommendation based on the response ID
        unique_recommendations = [
            "<p>Consider factor-based investing through smart-beta ETFs that offer a blend of active and passive strategies at lower costs than traditional active funds.</p>",
            "<p>Look into target-date funds that automatically adjust asset allocation as you approach your financial goal date.</p>",
            "<p>Explore the possibility of investing in municipal bonds which offer tax-free interest income.</p>",
            "<p>Consider value-averaging as an alternative to regular SIPs, where you adjust your investment amount to meet a predetermined growth rate.</p>",
            "<p>Investigate fractional property investments through platforms that allow you to own a percentage of premium real estate.</p>",
            "<p>Look into inflation-indexed bonds that provide protection against rising prices by adjusting returns based on inflation rates.</p>",
        ]
        response_parts.append("<h4>Unique Strategy to Consider</h4>")
        response_parts.append(unique_recommendations[response_id % len(unique_recommendations)])
        
        # Combine all parts into a single response
        return "\n".join(response_parts)
    
    # If not a question, use the standard advice templates but with some customization
    advice = {
        "conservative": """
<h2 class="advice-heading">Personalized Investment Advice - Conservative Profile</h2>

<p class="advice-paragraph">Based on your conservative risk profile, here are my recommendations:</p>

<ol class="advice-list">
  <li>
    <strong>Fixed Income Investments (60-70%)</strong>
    <ul>
      <li>Government bonds and treasury bills</li>
      <li>AAA-rated corporate bonds</li>
      <li>Fixed deposits in major banks</li>
      <li>Debt mutual funds with high-quality holdings</li>
    </ul>
  </li>
  
  <li>
    <strong>Equity Investments (15-25%)</strong>
    <ul>
      <li>Large-cap mutual funds focused on stable blue-chip companies</li>
      <li>Index funds tracking major indices like Nifty 50</li>
      <li>Dividend-yielding stocks from established sectors</li>
    </ul>
  </li>
  
  <li>
    <strong>Alternative Investments (5-10%)</strong>
    <ul>
      <li>Gold ETFs or sovereign gold bonds</li>
      <li>REITs (Real Estate Investment Trusts) with stable income properties</li>
    </ul>
  </li>
  
  <li>
    <strong>Cash and Equivalents (5-10%)</strong>
    <ul>
      <li>Liquid funds</li>
      <li>Short-term fixed deposits</li>
      <li>Savings accounts with competitive interest rates</li>
    </ul>
  </li>
</ol>

<p class="advice-paragraph"><strong>Key Considerations:</strong></p>
<ul class="advice-list">
  <li>Focus on capital preservation and steady income</li>
  <li>Maintain emergency fund covering 6-9 months of expenses</li>
  <li>Review portfolio quarterly and rebalance annually</li>
  <li>Consider tax-efficient investments like ELSS for tax planning</li>
</ul>

<p class="advice-paragraph">This conservative allocation aims to provide stability and income while minimizing volatility.</p>
""",
        "moderate": """
<h2 class="advice-heading">Personalized Investment Advice - Moderate Risk Profile</h2>

<p class="advice-paragraph">Based on your moderate risk profile, here are my balanced recommendations:</p>

<ol class="advice-list">
  <li>
    <strong>Equity Investments (40-50%)</strong>
    <ul>
      <li>Diversified large and mid-cap mutual funds</li>
      <li>Index funds tracking major indices</li>
      <li>Select growth stocks in promising sectors</li>
      <li>International equity funds (5-10% allocation)</li>
    </ul>
  </li>
  
  <li>
    <strong>Fixed Income (30-40%)</strong>
    <ul>
      <li>Government and corporate bonds</li>
      <li>Short to medium duration debt funds</li>
      <li>Fixed deposits with laddering strategy</li>
    </ul>
  </li>
  
  <li>
    <strong>Alternative Investments (10-15%)</strong>
    <ul>
      <li>REITs and InvITs for real estate exposure</li>
      <li>Gold ETFs or sovereign gold bonds</li>
      <li>Balanced advantage funds</li>
    </ul>
  </li>
  
  <li>
    <strong>Cash and Equivalents (5-10%)</strong>
    <ul>
      <li>Liquid funds for emergency needs</li>
      <li>Short-term deposits</li>
    </ul>
  </li>
</ol>

<p class="advice-paragraph"><strong>Key Strategies:</strong></p>
<ul class="advice-list">
  <li>Implement systematic investment plans (SIPs) for equity investments</li>
  <li>Consider tax-efficient options like ELSS for equity portion</li>
  <li>Maintain emergency fund covering 4-6 months of expenses</li>
  <li>Review portfolio quarterly and rebalance semi-annually</li>
</ul>

<p class="advice-paragraph">This balanced approach aims to provide growth potential while managing downside risk through diversification.</p>
""",
        "aggressive": """
<h2 class="advice-heading">Personalized Investment Advice - Aggressive Growth Profile</h2>

<p class="advice-paragraph">Based on your aggressive risk profile, here are my growth-oriented recommendations:</p>

<ol class="advice-list">
  <li>
    <strong>Equity Investments (65-75%)</strong>
    <ul>
      <li>Diversified mid and small-cap funds</li>
      <li>Sectoral and thematic funds in high-growth areas</li>
      <li>Direct equity in growth companies</li>
      <li>International equity funds (15-20% allocation)</li>
    </ul>
  </li>
  
  <li>
    <strong>Fixed Income (10-20%)</strong>
    <ul>
      <li>Strategic bond funds</li>
      <li>Credit risk funds with higher yields</li>
      <li>Short-duration debt for liquidity</li>
    </ul>
  </li>
  
  <li>
    <strong>Alternative Investments (10-15%)</strong>
    <ul>
      <li>REITs and InvITs</li>
      <li>Commodity ETFs</li>
      <li>Structured products with capital appreciation focus</li>
      <li>Private equity funds (if accessible)</li>
    </ul>
  </li>
  
  <li>
    <strong>Cash and Equivalents (0-5%)</strong>
    <ul>
      <li>Minimal cash holdings</li>
      <li>Liquid funds only for immediate needs</li>
    </ul>
  </li>
</ol>

<p class="advice-paragraph"><strong>Key Strategies:</strong></p>
<ul class="advice-list">
  <li>Implement systematic investment plans (SIPs) with step-up feature</li>
  <li>Consider tactical asset allocation based on market conditions</li>
  <li>Maintain higher exposure to emerging sectors like technology, renewable energy</li>
  <li>Review portfolio monthly and rebalance quarterly</li>
</ul>

<p class="advice-paragraph">This aggressive allocation aims to maximize long-term growth potential while accepting higher short-term volatility.</p>
"""
    }
    
    # Add customization to the standard templates based on mentioned topics
    base_advice = advice[risk_profile]
    
    # Add a custom section based on the mentioned topics
    if mentioned_topics:
        custom_sections = []
        
        # Add a date stamp for freshness
        custom_sections.append(f"<p class=\"advice-paragraph\"><em>Investment outlook as of {current_date}</em></p>")
        
        # Add topic-specific advice sections
        for topic in mentioned_topics:
            if topic == "retirement":
                custom_sections.append("<h3 class=\"advice-heading\">Retirement Focus</h3>")
                custom_sections.append("<p>For retirement planning, consider a systematic withdrawal plan (SWP) from your mutual fund investments during retirement years. This provides regular income while keeping the remaining corpus invested.</p>")
            
            elif topic == "tax":
                custom_sections.append("<h3 class=\"advice-heading\">Tax Optimization</h3>")
                custom_sections.append("<p>Consider tax-loss harvesting by selling investments that have experienced losses to offset capital gains tax on your profitable investments.</p>")
            
            # Add more topic-specific sections as needed
        
        # Insert the custom sections before the closing advice-content div
        if custom_sections:
            custom_content = "\n".join(custom_sections)
            # Add the custom content near the end of the advice
            parts = base_advice.rsplit("</ul>", 1)
            if len(parts) > 1:
                base_advice = parts[0] + "</ul>\n" + custom_content + parts[1]
            else:
                base_advice += "\n" + custom_content
    
    # Add a unique recommendation based on the response ID for variety
    unique_recommendations = [
        "<p><strong>Unique Strategy #{0}:</strong> Consider factor-based investing through smart-beta ETFs that offer a blend of active and passive strategies at lower costs.</p>",
        "<p><strong>Unique Strategy #{0}:</strong> Look into target-date funds that automatically adjust asset allocation as you approach your financial goal date.</p>",
        "<p><strong>Unique Strategy #{0}:</strong> Explore the possibility of investing in municipal bonds which offer tax-free interest income.</p>",
        "<p><strong>Unique Strategy #{0}:</strong> Consider value-averaging as an alternative to regular SIPs, where you adjust your investment amount to meet a predetermined growth rate.</p>",
        "<p><strong>Unique Strategy #{0}:</strong> Investigate fractional property investments through platforms that allow you to own a percentage of premium real estate.</p>",
        "<p><strong>Unique Strategy #{0}:</strong> Look into inflation-indexed bonds that provide protection against rising prices by adjusting returns based on inflation rates.</p>",
    ]
    
    unique_rec = unique_recommendations[response_id % len(unique_recommendations)].format(response_id)
    parts = base_advice.rsplit("</p>", 1)
    if len(parts) > 1:
        base_advice = parts[0] + "</p>\n" + unique_rec + parts[1]
    else:
        base_advice += "\n" + unique_rec
    
    return base_advice

def generate_investment_advice_prompt(profile):
    """Generate a comprehensive prompt for the Ollama model based on user's financial profile"""
    
    prompt = f"""As an expert financial advisor, provide personalized investment advice based on the following client profile:

Personal Information:
- Age: {profile.age}
- Occupation: {profile.occupation}
- Family Size: {profile.family_size}

Financial Situation:
- Monthly Income: ${profile.monthly_income}
- Monthly Expenses: ${profile.monthly_expenses}
- Monthly Savings: ${profile.monthly_savings}
- Current Debts: ${profile.current_debts} (Interest Rate: {profile.debt_interest_rate}%)

Investment Profile:
- Risk Tolerance: {profile.get_risk_tolerance_display()}
- Emotional Stability: {profile.get_emotional_stability_display()}
- Investment Knowledge: {profile.get_investment_knowledge_display()}
- Previous Investment Experience: {'Yes' if profile.has_investment_experience else 'No'}
- Previous Investments: {profile.previous_investments}

Financial Goals:
Short-term (1-3 years):
- Goals: {profile.short_term_goals}
- Required Amount: ${profile.short_term_goal_amount}

Medium-term (5-10 years):
- Goals: {profile.medium_term_goals}
- Required Amount: ${profile.medium_term_goal_amount}

Long-term (10+ years):
- Goals: {profile.long_term_goals}
- Required Amount: ${profile.long_term_goal_amount}

Additional Information:
- Other Assets: {profile.other_assets}
- Retirement Plans: {profile.retirement_plans}

Please provide:
1. A comprehensive investment strategy that aligns with the client's risk tolerance and goals
2. Specific investment recommendations for each time horizon
3. Asset allocation suggestions
4. Risk management strategies
5. Regular review and rebalancing recommendations
6. Any specific concerns or considerations based on the client's profile

Format the response in a clear, structured manner with sections and bullet points where appropriate."""

    return prompt
