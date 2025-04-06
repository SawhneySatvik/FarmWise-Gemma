from app.agents.base_agent import LLMAgent
from datetime import datetime, timedelta

class IrrigationAgent(LLMAgent):
    """Agent for irrigation and water management advice"""
    
    def __init__(self, name, llm_service):
        """Initialize irrigation agent with LLM service"""
        super().__init__(name, llm_service)
    
    def process(self, query, context=None):
        """
        Process irrigation-related queries
        
        Args:
            query (str): The user query or request
            context (dict, optional): Additional context for processing
            
        Returns:
            dict: Formatted response with irrigation recommendations
        """
        # Extract parameters from context if available
        params = {}
        if context and "extracted_params" in context:
            params = context["extracted_params"]
        
        # Get user context if available
        user_context = context.get("user_context", {}) if context else {}
        
        # Determine the type of irrigation query
        query_intent = self._determine_irrigation_intent(query, params)
        
        # Generate appropriate response based on intent
        if query_intent == "schedule":
            response, confidence = self._generate_irrigation_schedule(query, params, user_context)
        elif query_intent == "system":
            response, confidence = self._generate_irrigation_system_advice(query, params, user_context)
        elif query_intent == "water_conservation":
            response, confidence = self._generate_water_conservation_advice(query, params, user_context)
        elif query_intent == "troubleshooting":
            response, confidence = self._generate_irrigation_troubleshooting(query, params, user_context)
        else:
            # General irrigation query
            response, confidence = self._generate_general_irrigation_response(query, params, user_context)
        
        # Return formatted response
        return self.format_response(
            response,
            confidence,
            {"intent": query_intent, "crop_type": params.get("crop_type")}
        )
    
    def _determine_irrigation_intent(self, query, params):
        """Determine the intent of the irrigation-related query"""
        query_lower = query.lower()
        
        # Check for irrigation schedule intent
        if any(term in query_lower for term in ["schedule", "when to water", "how often", "frequency", "timing"]):
            return "schedule"
        
        # Check for irrigation system intent
        if any(term in query_lower for term in ["system", "drip", "sprinkler", "flood", "furrow", "method", "technique"]):
            return "system"
        
        # Check for water conservation intent
        if any(term in query_lower for term in ["save water", "conserve", "efficient", "reduce water", "water saving"]):
            return "water_conservation"
        
        # Check for troubleshooting intent
        if any(term in query_lower for term in ["problem", "issue", "not working", "clogged", "leaking", "fix", "repair"]):
            return "troubleshooting"
        
        # Default to general
        return "general"
    
    def _generate_irrigation_schedule(self, query, params, user_context):
        """Generate irrigation schedule recommendations"""
        # Extract crop type or use default
        crop_type = params.get("crop_type") or user_context.get("primary_crops") or "your crops"
        
        # Extract soil type or use default
        soil_type = params.get("soil_type") or user_context.get("soil_type") or "average soil"
        
        # Extract location and weather conditions
        location = params.get("location") or user_context.get("location") or "India"
        weather = params.get("weather") or "current weather conditions"
        
        # Extract irrigation method if available
        irrigation_method = user_context.get("irrigation_type") or "your irrigation system"
        
        # Get current season
        current_season = self._determine_current_season()
        
        # Create irrigation schedule prompt
        prompt = f"""
        You are an irrigation specialist with expertise in water management for Indian agriculture.
        
        Provide a detailed irrigation schedule for {crop_type} grown in {soil_type} in {location} during {current_season}.
        Weather conditions: {weather}
        Irrigation method: {irrigation_method}
        
        Farmer's Query: {query}
        
        Include in your response:
        1. Recommended irrigation frequency (days between watering)
        2. Suggested water quantity per irrigation
        3. Best time of day for irrigation
        4. Signs to monitor to adjust the schedule (plant appearance, soil moisture)
        5. How to adjust the schedule based on rainfall
        
        Provide practical, water-efficient recommendations suitable for Indian farming conditions.
        Include traditional knowledge and modern irrigation science.
        """
        
        # Generate response
        response = self.generate_response(prompt)
        
        # Confidence based on specificity
        confidence = 0.85 if (params.get("crop_type") and (params.get("weather") or params.get("soil_type"))) else 0.7
        
        return response, confidence
    
    def _generate_irrigation_system_advice(self, query, params, user_context):
        """Generate irrigation system recommendations"""
        # Extract crop type or use default
        crop_type = params.get("crop_type") or user_context.get("primary_crops") or "your crops"
        
        # Extract farm size if available
        farm_size = user_context.get("farm_size") or "your farm"
        
        # Extract location
        location = params.get("location") or user_context.get("location") or "India"
        
        # Extract water source if available
        water_source = params.get("water_source") or "available water sources"
        
        # Create irrigation system prompt
        prompt = f"""
        You are an agricultural irrigation engineer specializing in systems for Indian farmers.
        
        Provide recommendations on irrigation systems for {crop_type} on a {farm_size} acre farm in {location}.
        Water source: {water_source}
        
        Farmer's Query: {query}
        
        Include in your response:
        1. Suitable irrigation systems for these conditions (drip, sprinkler, furrow, etc.)
        2. Comparative advantages and disadvantages of each system
        3. Approximate setup costs and ongoing maintenance needs
        4. Water and energy efficiency considerations
        5. Government subsidy programs available in India (if any)
        
        Focus on practical, cost-effective solutions appropriate for Indian agricultural conditions.
        Consider both small-scale and larger implementations depending on farm size.
        """
        
        # Generate response
        response = self.generate_response(prompt)
        
        # Confidence based on specificity
        confidence = 0.8
        
        return response, confidence
    
    def _generate_water_conservation_advice(self, query, params, user_context):
        """Generate water conservation recommendations"""
        # Extract crop type or use default
        crop_type = params.get("crop_type") or user_context.get("primary_crops") or "your crops"
        
        # Extract location and weather
        location = params.get("location") or user_context.get("location") or "India"
        weather = params.get("weather") or "local weather conditions"
        
        # Extract irrigation method if available
        irrigation_method = user_context.get("irrigation_type") or "your current irrigation method"
        
        # Create water conservation prompt
        prompt = f"""
        You are a water management specialist focusing on conservation in Indian agriculture.
        
        Provide detailed water conservation strategies for {crop_type} cultivation in {location}.
        Current irrigation method: {irrigation_method}
        Weather conditions: {weather}
        
        Farmer's Query: {query}
        
        Include practical water-saving techniques such as:
        1. Irrigation timing and frequency adjustments
        2. Mulching methods to reduce evaporation
        3. Soil management to improve water retention
        4. Plant spacing and arrangement strategies
        5. Rainwater harvesting and storage options
        6. Low-cost moisture monitoring approaches
        
        Focus on affordable, implementable solutions for Indian farmers that save water while maintaining yields.
        Include both traditional water conservation knowledge and modern approaches.
        """
        
        # Generate response
        response = self.generate_response(prompt)
        
        # Standard confidence for conservation advice
        confidence = 0.8
        
        return response, confidence
    
    def _generate_irrigation_troubleshooting(self, query, params, user_context):
        """Generate irrigation troubleshooting advice"""
        # Extract irrigation method if available
        irrigation_method = user_context.get("irrigation_type") or params.get("irrigation_type") or "irrigation system"
        
        # Try to extract the specific problem from the query
        problem = self._extract_irrigation_problem(query)
        
        # Create troubleshooting prompt
        prompt = f"""
        You are an irrigation system troubleshooting expert with experience in Indian farming systems.
        
        Provide practical solutions for problems with {irrigation_method} irrigation systems.
        Specific issue: {problem}
        
        Farmer's Query: {query}
        
        Include in your response:
        1. Potential causes of the problem
        2. Step-by-step troubleshooting process
        3. Common fixes for the issue
        4. When to seek professional help
        5. Preventative maintenance tips to avoid future problems
        
        Focus on practical solutions that farmers can implement themselves with minimal resources.
        Consider the challenges of rural Indian settings where parts and tools may be limited.
        """
        
        # Generate response
        response = self.generate_response(prompt)
        
        # Confidence based on problem specificity
        confidence = 0.8 if problem else 0.7
        
        return response, confidence
    
    def _generate_general_irrigation_response(self, query, params, user_context):
        """Generate general irrigation information response"""
        # Extract crop type if available
        crop_type = params.get("crop_type") or user_context.get("primary_crops") or "crops in general"
        
        # Extract location
        location = params.get("location") or user_context.get("location") or "India"
        
        # Create general irrigation info prompt
        prompt = f"""
        You are an irrigation specialist with extensive knowledge of water management in Indian agriculture.
        
        Provide helpful information about irrigation for {crop_type} in {location}.
        
        Farmer's Query: {query}
        
        Include relevant information such as:
        1. Basic principles of effective irrigation
        2. Common irrigation methods used for these crops in India
        3. Factors affecting irrigation decisions
        4. Signs of under-watering and over-watering
        5. Water source considerations
        
        Provide practical, science-based information specific to Indian agricultural conditions.
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
    
    def _extract_irrigation_problem(self, query):
        """Extract specific irrigation problem from the query"""
        query_lower = query.lower()
        
        # Common irrigation problems
        problems = {
            "clogging": ["clog", "block", "stuck", "not flowing", "jammed"],
            "leaking": ["leak", "drip", "seep", "water loss", "broken pipe"],
            "pressure issues": ["pressure", "flow rate", "low pressure", "high pressure", "not enough force"],
            "uneven distribution": ["uneven", "patchy", "some areas", "not reaching", "dry spots"],
            "system failure": ["not working", "stopped", "failed", "broken", "damaged"],
            "water quality": ["dirty water", "sediment", "contamination", "salt", "quality"]
        }
        
        # Check for problems
        for problem, terms in problems.items():
            if any(term in query_lower for term in terms):
                return problem
        
        return "general irrigation issues"
