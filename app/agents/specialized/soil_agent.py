from app.agents.base_agent import LLMAgent

class SoilAgent(LLMAgent):
    """Agent for soil analysis and management recommendations"""
    
    def __init__(self, name, llm_service):
        """Initialize soil agent with LLM service"""
        super().__init__(name, llm_service)
    
    def process(self, query, context=None):
        """
        Process soil-related queries
        
        Args:
            query (str): The user query or request
            context (dict, optional): Additional context for processing
            
        Returns:
            dict: Formatted response with soil recommendations
        """
        # Extract parameters from context if available
        params = {}
        if context and "extracted_params" in context:
            params = context["extracted_params"]
        
        # Get user context if available
        user_context = context.get("user_context", {}) if context else {}
        
        # Determine the type of soil query
        query_intent = self._determine_soil_intent(query, params)
        
        # Generate appropriate response based on intent
        if query_intent == "analysis":
            response, confidence = self._generate_soil_analysis(query, params, user_context)
        elif query_intent == "improvement":
            response, confidence = self._generate_soil_improvement(query, params, user_context)
        elif query_intent == "fertilizer":
            response, confidence = self._generate_fertilizer_recommendation(query, params, user_context)
        elif query_intent == "testing":
            response, confidence = self._generate_soil_testing_advice(query, params, user_context)
        else:
            # General soil query
            response, confidence = self._generate_general_soil_response(query, params, user_context)
        
        # Return formatted response
        return self.format_response(
            response,
            confidence,
            {"intent": query_intent, "soil_type": params.get("soil_type")}
        )
    
    def _determine_soil_intent(self, query, params):
        """Determine the intent of the soil-related query"""
        query_lower = query.lower()
        
        # Check for soil analysis intent
        if any(term in query_lower for term in ["analyze", "analysis", "soil type", "soil quality", "fertility"]):
            return "analysis"
        
        # Check for soil improvement intent
        if any(term in query_lower for term in ["improve", "enhancement", "amendment", "better soil", "increase organic"]):
            return "improvement"
        
        # Check for fertilizer recommendation intent
        if any(term in query_lower for term in ["fertilizer", "fertilize", "manure", "nutrient", "compost"]):
            return "fertilizer"
        
        # Check for soil testing intent
        if any(term in query_lower for term in ["test", "testing", "lab", "sample", "check soil"]):
            return "testing"
        
        # Default to general
        return "general"
    
    def _generate_soil_analysis(self, query, params, user_context):
        """Generate soil analysis response"""
        # Extract soil type or use default
        soil_type = params.get("soil_type") or user_context.get("soil_type") or "unknown soil type"
        
        # Extract location
        location = params.get("location") or user_context.get("location") or "India"
        
        # Extract crop if available
        crop_type = params.get("crop_type") or user_context.get("primary_crops") or "crops in general"
        
        # Create soil analysis prompt
        prompt = f"""
        You are a soil scientist with expertise in Indian agricultural soils.
        
        Provide a detailed analysis for {soil_type} in {location}, considering cultivation of {crop_type}.
        
        Farmer's Query: {query}
        
        Include in your response:
        1. Key characteristics of this soil type
        2. Typical nutrient profile and pH range
        3. Water retention characteristics
        4. Suitability for different crops
        5. Common challenges with this soil type
        
        Provide practical insights specific to Indian agricultural conditions and traditional knowledge.
        """
        
        # Generate response
        response = self.generate_response(prompt)
        
        # High confidence for analysis with known soil type
        confidence = 0.9 if params.get("soil_type") else 0.7
        
        return response, confidence
    
    def _generate_soil_improvement(self, query, params, user_context):
        """Generate soil improvement recommendations"""
        # Extract soil type or use default
        soil_type = params.get("soil_type") or user_context.get("soil_type") or "your soil"
        
        # Extract location
        location = params.get("location") or user_context.get("location") or "India"
        
        # Extract specific concerns
        concern = params.get("concern") or "soil health"
        
        # Create soil improvement prompt
        prompt = f"""
        You are a soil management expert specializing in sustainable agriculture practices in India.
        
        Provide detailed recommendations for improving {soil_type} in {location}, focusing on {concern}.
        
        Farmer's Query: {query}
        
        Include practical methods such as:
        1. Organic matter addition techniques
        2. Cover cropping strategies
        3. Mulching approaches
        4. Appropriate tillage practices
        5. Crop rotation benefits for soil health
        
        Focus on cost-effective, locally available solutions that Indian farmers can implement.
        Include both traditional knowledge and modern sustainable practices.
        """
        
        # Generate response
        response = self.generate_response(prompt)
        
        # Confidence based on specificity
        confidence = 0.85 if params.get("soil_type") else 0.7
        
        return response, confidence
    
    def _generate_fertilizer_recommendation(self, query, params, user_context):
        """Generate fertilizer recommendations"""
        # Extract soil type or use default
        soil_type = params.get("soil_type") or user_context.get("soil_type") or "your soil"
        
        # Extract crop type
        crop_type = params.get("crop_type") or user_context.get("primary_crops") or "your crops"
        
        # Extract location
        location = params.get("location") or user_context.get("location") or "your region"
        
        # Create fertilizer recommendation prompt
        prompt = f"""
        You are an agricultural nutrient management specialist in India.
        
        Provide fertilizer recommendations for {crop_type} grown in {soil_type} in {location}.
        
        Farmer's Query: {query}
        
        Include in your response:
        1. Balanced NPK (Nitrogen, Phosphorus, Potassium) requirements
        2. Timing of fertilizer application
        3. Organic fertilizer options and their benefits
        4. Chemical fertilizer options and proper usage
        5. Signs of nutrient deficiencies to watch for
        
        Emphasize sustainable practices that maintain soil health while optimizing crop yields.
        Include traditional Indian farming knowledge where relevant.
        """
        
        # Generate response
        response = self.generate_response(prompt)
        
        # Confidence based on specificity
        confidence = 0.8 if (params.get("soil_type") and params.get("crop_type")) else 0.65
        
        return response, confidence
    
    def _generate_soil_testing_advice(self, query, params, user_context):
        """Generate soil testing advice"""
        # Extract location
        location = params.get("location") or user_context.get("location") or "your region"
        
        # Create soil testing advice prompt
        prompt = f"""
        You are an agricultural extension specialist in India with expertise in soil testing.
        
        Provide practical advice on soil testing for farmers in {location}.
        
        Farmer's Query: {query}
        
        Include information on:
        1. How to properly collect soil samples
        2. When is the best time to test soil
        3. Which parameters are most important to test
        4. Where to send samples in India (government labs, universities)
        5. How to interpret basic soil test results
        6. Approximate costs and timeline
        
        Focus on practical, actionable advice that farmers can implement themselves.
        Include information on government soil testing programs available in India.
        """
        
        # Generate response
        response = self.generate_response(prompt)
        
        # Standard confidence for testing advice
        confidence = 0.75
        
        return response, confidence
    
    def _generate_general_soil_response(self, query, params, user_context):
        """Generate general soil information response"""
        # Extract soil type if available
        soil_type = params.get("soil_type") or user_context.get("soil_type") or "various soil types"
        
        # Extract location
        location = params.get("location") or user_context.get("location") or "India"
        
        # Create general soil info prompt
        prompt = f"""
        You are a soil scientist with deep knowledge of soils in India.
        
        Provide helpful information about {soil_type} in {location}.
        
        Farmer's Query: {query}
        
        Include relevant information such as:
        1. General characteristics of these soils
        2. Common crops that grow well in these soils
        3. Typical challenges farmers face with these soils
        4. Basic management practices for maintaining soil health
        
        Provide accurate, practical information specific to Indian agricultural conditions.
        """
        
        # Generate response
        response = self.generate_response(prompt)
        
        # Default confidence for general responses
        confidence = 0.7
        
        return response, confidence
