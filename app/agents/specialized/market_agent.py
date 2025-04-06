from app.agents.base_agent import LLMAgent
from datetime import datetime

class MarketAgent(LLMAgent):
    """Agent for agricultural market intelligence and price information"""
    
    def __init__(self, name, llm_service):
        """Initialize market agent with LLM service"""
        super().__init__(name, llm_service)
    
    def process(self, query, context=None):
        """
        Process market-related queries
        
        Args:
            query (str): The user query or request
            context (dict, optional): Additional context for processing
            
        Returns:
            dict: Formatted response with market information
        """
        # Extract parameters from context if available
        params = {}
        if context and "extracted_params" in context:
            params = context["extracted_params"]
        
        # Get user context if available
        user_context = context.get("user_context", {}) if context else {}
        
        # Determine the type of market query
        query_intent = self._determine_market_intent(query, params)
        
        # Generate appropriate response based on intent
        if query_intent == "current_price":
            response, confidence = self._generate_current_price_response(query, params, user_context)
        elif query_intent == "price_forecast":
            response, confidence = self._generate_price_forecast_response(query, params, user_context)
        elif query_intent == "market_selection":
            response, confidence = self._generate_market_selection_response(query, params, user_context)
        elif query_intent == "selling_strategy":
            response, confidence = self._generate_selling_strategy_response(query, params, user_context)
        elif query_intent == "market_trends":
            response, confidence = self._generate_market_trends_response(query, params, user_context)
        else:
            # General market query
            response, confidence = self._generate_general_market_response(query, params, user_context)
        
        # Return formatted response
        return self.format_response(
            response,
            confidence,
            {"intent": query_intent, "crop_type": params.get("crop_type"), "location": params.get("location")}
        )
    
    def _determine_market_intent(self, query, params):
        """Determine the intent of the market-related query"""
        query_lower = query.lower()
        
        # Check for current price intent
        if any(term in query_lower for term in ["current price", "market price", "rate", "how much is", "selling for", "mandi price"]):
            return "current_price"
        
        # Check for price forecast intent
        if any(term in query_lower for term in ["future price", "forecast", "predict", "trend", "will price", "next month", "next week"]):
            return "price_forecast"
        
        # Check for market selection intent
        if any(term in query_lower for term in ["which market", "best market", "where to sell", "better price", "nearby market"]):
            return "market_selection"
        
        # Check for selling strategy intent
        if any(term in query_lower for term in ["when to sell", "best time", "selling strategy", "hold", "wait", "should i sell"]):
            return "selling_strategy"
        
        # Check for market trends intent
        if any(term in query_lower for term in ["trend", "pattern", "historical", "market movement", "price history", "analysis"]):
            return "market_trends"
        
        # Default to general
        return "general"
    
    def _generate_current_price_response(self, query, params, user_context):
        """Generate current price information"""
        # Extract crop type or use default
        crop_type = params.get("crop_type") or user_context.get("primary_crops") or "agricultural products"
        
        # Extract location
        location = params.get("location") or user_context.get("location") or "your region"
        
        # Extract quality grade if available
        quality = params.get("quality") or "average quality"
        
        # Get current date for reference
        current_date = datetime.now().strftime("%d %B, %Y")
        
        # Create current price prompt
        prompt = f"""
        You are an agricultural market intelligence expert with specific knowledge of Indian markets.
        
        Provide current market price information for {crop_type} in {location} as of {current_date}.
        Quality grade: {quality}
        
        Farmer's Query: {query}
        
        Include in your response:
        1. Current price range for {crop_type} in {location} (approximate if exact data not known)
        2. Comparison to last month's prices (higher/lower/stable)
        3. Factors currently affecting the price
        4. Price variations based on quality grades if applicable
        5. Nearby markets with potentially different prices
        
        Give a balanced assessment of current market conditions.
        Prices should be realistic for Indian markets and denominated in Rupees per appropriate unit (kg/quintal).
        If precise data would require real-time market information, provide reasonable estimates based on seasonal patterns.
        """
        
        # Generate response
        response = self.generate_response(prompt)
        
        # Confidence level based on specificity
        confidence = 0.7  # Limited by not having real-time market data
        
        return response, confidence
    
    def _generate_price_forecast_response(self, query, params, user_context):
        """Generate price forecast information"""
        # Extract crop type or use default
        crop_type = params.get("crop_type") or user_context.get("primary_crops") or "agricultural products"
        
        # Extract location
        location = params.get("location") or user_context.get("location") or "your region"
        
        # Extract time period if available
        time_period = params.get("time_period") or "the coming weeks and months"
        
        # Get current date for reference
        current_date = datetime.now().strftime("%d %B, %Y")
        
        # Create price forecast prompt
        prompt = f"""
        You are an agricultural market analyst specializing in price forecasting for Indian agricultural markets.
        
        Provide a price forecast for {crop_type} in {location} for {time_period} from {current_date}.
        
        Farmer's Query: {query}
        
        Include in your response:
        1. Likely price trends for {crop_type} in {time_period}
        2. Key factors that could influence prices (supply, demand, weather, policies)
        3. Price volatility assessment (stable, moderate volatility, highly volatile)
        4. Potential high and low price points
        5. Confidence level of the forecast (acknowledging limitations)
        
        Base your forecast on typical seasonal patterns, current market conditions, and historical trends.
        Acknowledge the uncertainty in price predictions and provide cautious ranges rather than specific figures.
        Frame the forecast in terms useful for a farmer's decision-making.
        """
        
        # Generate response
        response = self.generate_response(prompt)
        
        # Confidence level for forecasts should be modest
        confidence = 0.6  # Lower due to inherent uncertainty in forecasting
        
        return response, confidence
    
    def _generate_market_selection_response(self, query, params, user_context):
        """Generate market selection recommendations"""
        # Extract crop type or use default
        crop_type = params.get("crop_type") or user_context.get("primary_crops") or "your produce"
        
        # Extract location
        location = params.get("location") or user_context.get("location") or "your area"
        
        # Extract quantity if available
        quantity = params.get("quantity") or user_context.get("harvest_size") or "your harvest"
        
        # Create market selection prompt
        prompt = f"""
        You are an agricultural marketing specialist with deep knowledge of Indian agricultural market systems.
        
        Advise on the best markets to sell {crop_type} from {location} with a quantity of approximately {quantity}.
        
        Farmer's Query: {query}
        
        Include in your response:
        1. Types of markets available (local mandis, larger wholesale markets, direct buyers)
        2. Pros and cons of each market option
        3. Factors to consider when selecting a market (distance, transportation cost, price volatility)
        4. Market fee structures and regulations to be aware of
        5. Alternative marketing channels (processing units, contracts, cooperatives)
        
        Focus on practical advice that considers the realities faced by Indian farmers.
        Include information about digital marketing platforms and government initiatives if relevant.
        Provide guidance on how to assess whether a market is offering fair prices.
        """
        
        # Generate response
        response = self.generate_response(prompt)
        
        # Confidence based on specificity
        confidence = 0.75
        
        return response, confidence
    
    def _generate_selling_strategy_response(self, query, params, user_context):
        """Generate selling strategy recommendations"""
        # Extract crop type or use default
        crop_type = params.get("crop_type") or user_context.get("primary_crops") or "your produce"
        
        # Extract location
        location = params.get("location") or user_context.get("location") or "your region"
        
        # Extract storage capability if available
        storage_capability = user_context.get("storage_capability") or "typical storage facilities"
        
        # Extract financial situation if available
        financial_situation = user_context.get("financial_situation") or "typical financial considerations"
        
        # Get current season
        current_season = self._determine_current_season()
        
        # Create selling strategy prompt
        prompt = f"""
        You are an agricultural marketing strategist advising Indian farmers on optimizing their sales timing.
        
        Provide strategic advice on when and how to sell {crop_type} from {location} during {current_season}.
        Storage capability: {storage_capability}
        Financial situation: {financial_situation}
        
        Farmer's Query: {query}
        
        Include in your response:
        1. Current market situation and expected short-term price movements
        2. Risks of holding vs. selling immediately
        3. Strategies for staggered selling to manage risk
        4. Importance of quality maintenance during storage if holding
        5. Financial considerations (immediate cash needs vs. potential gains from waiting)
        
        Provide balanced advice that considers both market opportunities and the farmer's risk tolerance.
        Acknowledge the uncertainty in market predictions while giving practical actionable guidance.
        Consider both traditional and innovative selling approaches.
        """
        
        # Generate response
        response = self.generate_response(prompt)
        
        # Confidence based on specificity
        confidence = 0.7
        
        return response, confidence
    
    def _generate_market_trends_response(self, query, params, user_context):
        """Generate market trends information"""
        # Extract crop type or use default
        crop_type = params.get("crop_type") or user_context.get("primary_crops") or "agricultural commodities"
        
        # Extract location
        location = params.get("location") or user_context.get("location") or "Indian markets"
        
        # Extract time period if available
        time_period = params.get("time_period") or "recent months"
        
        # Create market trends prompt
        prompt = f"""
        You are an agricultural market analyst specializing in Indian commodity markets.
        
        Analyze market trends for {crop_type} in {location} over {time_period}.
        
        Farmer's Query: {query}
        
        Include in your response:
        1. Overall price movement patterns (rising, falling, stable, cyclical)
        2. Seasonal factors affecting {crop_type} prices
        3. Supply and demand dynamics in {location} and broader markets
        4. Impact of government policies, import/export scenarios, and other external factors
        5. Comparison with historical patterns for this time of year
        
        Present a balanced analysis of market forces, considering both macro and local factors.
        Include relevant information about how these trends might affect small and medium farmers.
        Focus on insights that would be useful for planning planting, harvesting, or selling decisions.
        """
        
        # Generate response
        response = self.generate_response(prompt)
        
        # Confidence level for trend analysis
        confidence = 0.75
        
        return response, confidence
    
    def _generate_general_market_response(self, query, params, user_context):
        """Generate general market information response"""
        # Extract crop type if available
        crop_type = params.get("crop_type") or user_context.get("primary_crops") or "agricultural products"
        
        # Extract location
        location = params.get("location") or user_context.get("location") or "India"
        
        # Create general market info prompt
        prompt = f"""
        You are an agricultural market intelligence expert with extensive knowledge of Indian agricultural markets.
        
        Provide helpful information about markets for {crop_type} in {location}.
        
        Farmer's Query: {query}
        
        Include relevant information such as:
        1. Current market dynamics for {crop_type}
        2. Key factors influencing agricultural prices in {location}
        3. Marketing channels available to farmers
        4. Best practices for market participation
        5. Resources for accessing up-to-date market information
        
        Focus on practical information relevant to Indian farmers.
        Consider the specific challenges and opportunities in Indian agricultural marketing systems.
        """
        
        # Generate response
        response = self.generate_response(prompt)
        
        # Default confidence for general responses
        confidence = 0.7
        
        return response, confidence
    
    def _determine_current_season(self):
        """Determine the current agricultural season in India"""
        # Simple month-based approach
        current_month = datetime.now().month
        
        if 6 <= current_month <= 9:
            return "Kharif (Monsoon) season"
        elif 10 <= current_month <= 11 or current_month <= 2:
            return "Rabi (Winter) season"
        else:
            return "Zaid (Summer) season"
