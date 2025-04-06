from app.agents.base_agent import LLMAgent
from datetime import datetime

class CropAgent(LLMAgent):
    """Agent for crop recommendations and management"""
    
    def __init__(self, name, llm_service):
        """Initialize crop agent with LLM service"""
        super().__init__(name, llm_service)
    
    def process(self, query, context=None):
        """
        Process crop-related queries
        
        Args:
            query (str): The user query or request
            context (dict, optional): Additional context for processing
            
        Returns:
            dict: Formatted response with crop recommendations
        """
        # Extract parameters from context if available
        params = {}
        if context and "extracted_params" in context:
            params = context["extracted_params"]
        
        # Get user context if available
        user_context = context.get("user_context", {}) if context else {}
        
        # Determine the type of crop query
        query_intent = self._determine_crop_intent(query, params)
        
        # Generate appropriate response based on intent
        if query_intent == "selection":
            response, confidence = self._generate_crop_selection(query, params, user_context)
        elif query_intent == "management":
            response, confidence = self._generate_crop_management(query, params, user_context)
        elif query_intent == "yield_optimization":
            response, confidence = self._generate_yield_optimization(query, params, user_context)
        elif query_intent == "variety":
            response, confidence = self._generate_variety_recommendation(query, params, user_context)
        elif query_intent == "rotation":
            response, confidence = self._generate_crop_rotation_advice(query, params, user_context)
        else:
            # General crop query
            response, confidence = self._generate_general_crop_response(query, params, user_context)
        
        # Return formatted response
        return self.format_response(
            response,
            confidence,
            {"intent": query_intent, "crop_type": params.get("crop_type")}
        )
    
    def _determine_crop_intent(self, query, params):
        """Determine the intent of the crop-related query"""
        query_lower = query.lower()
        
        # Check for crop selection intent
        if any(term in query_lower for term in ["which crop", "what crop", "suggest crop", "recommend crop", "best crop", "suitable crop"]):
            return "selection"
        
        # Check for crop management intent
        if any(term in query_lower for term in ["how to grow", "how to cultivate", "manage", "cultivation", "growing"]):
            return "management"
        
        # Check for yield optimization intent
        if any(term in query_lower for term in ["increase yield", "better yield", "higher yield", "improve production", "maximize"]):
            return "yield_optimization"
        
        # Check for variety recommendation intent
        if any(term in query_lower for term in ["variety", "cultivar", "hybrid", "which seed", "best seed"]):
            return "variety"
        
        # Check for crop rotation intent
        if any(term in query_lower for term in ["rotation", "after", "before", "sequence", "alternate"]):
            return "rotation"
        
        # Default to general
        return "general"
    
    def _generate_crop_selection(self, query, params, user_context):
        """Generate crop selection recommendations"""
        # Extract location or use default
        location = params.get("location") or user_context.get("location") or "your region"
        
        # Extract soil type if available
        soil_type = params.get("soil_type") or user_context.get("soil_type") or "your soil"
        
        # Extract season or determine current
        season = params.get("season") or self._determine_current_season()
        
        # Create crop selection prompt
        prompt = f"""
        You are an agricultural scientist specializing in Indian crop selection.
        
        Recommend suitable crops for planting in {location} during {season} on {soil_type}.
        
        Farmer's Query: {query}
        
        Include in your response:
        1. Top 3-5 crops well-suited for these conditions
        2. Key advantages of each recommended crop
        3. Approximate growing period and expected yield
        4. Basic resource requirements (water, fertilizer, labor)
        5. Market potential (demand and price trends)
        
        Consider climate patterns, soil compatibility, water availability, and economic value.
        Focus on crops that are realistically viable for Indian farmers in this region.
        Include both traditional staples and potentially profitable alternatives.
        """
        
        # Generate response
        response = self.generate_response(prompt)
        
        # Confidence based on specificity
        confidence = 0.85 if params.get("location") else 0.7
        
        return response, confidence
    
    def _generate_crop_management(self, query, params, user_context):
        """Generate crop management recommendations"""
        # Extract crop type or use default
        crop_type = params.get("crop_type") or self._extract_crop_from_query(query) or "the crop"
        
        # Extract location
        location = params.get("location") or user_context.get("location") or "your region"
        
        # Extract growing stage if mentioned
        growing_stage = params.get("growing_stage") or self._extract_growing_stage(query) or "all growth stages"
        
        # Create crop management prompt
        prompt = f"""
        You are an agricultural extension specialist with expertise in Indian crop management practices.
        
        Provide detailed guidance on managing {crop_type} at {growing_stage} in {location}.
        
        Farmer's Query: {query}
        
        Include in your response:
        1. Key management activities for this growth stage
        2. Optimal timing and methods for these activities
        3. Input requirements (fertilizers, water, etc.)
        4. Potential challenges and how to address them
        5. Signs of healthy crop development to look for
        
        Focus on practical, actionable advice that Indian farmers can implement with available resources.
        Blend traditional farming knowledge with modern scientific practices suitable for Indian conditions.
        """
        
        # Generate response
        response = self.generate_response(prompt)
        
        # Confidence based on specificity
        confidence = 0.9 if params.get("crop_type") else 0.7
        
        return response, confidence
    
    def _generate_yield_optimization(self, query, params, user_context):
        """Generate yield optimization recommendations"""
        # Extract crop type or use default
        crop_type = params.get("crop_type") or self._extract_crop_from_query(query) or "your crop"
        
        # Extract location
        location = params.get("location") or user_context.get("location") or "your region"
        
        # Extract current practices if available
        current_practices = user_context.get("farming_practices") or "current farming practices"
        
        # Create yield optimization prompt
        prompt = f"""
        You are an agricultural productivity expert specializing in yield optimization for Indian farms.
        
        Provide detailed recommendations to increase yield of {crop_type} in {location}, considering {current_practices}.
        
        Farmer's Query: {query}
        
        Include in your response:
        1. Key yield-limiting factors to address
        2. Evidence-based practices to maximize productivity
        3. Optimal input management (seeds, fertilizers, water, etc.)
        4. Critical timing of interventions during the crop cycle
        5. Appropriate technologies and techniques for Indian conditions
        
        Focus on practical strategies that are cost-effective and accessible to Indian farmers.
        Balance traditional knowledge with modern agricultural science.
        Emphasize sustainable practices that maintain long-term soil health and farm viability.
        """
        
        # Generate response
        response = self.generate_response(prompt)
        
        # Confidence based on specificity
        confidence = 0.85 if params.get("crop_type") else 0.7
        
        return response, confidence
    
    def _generate_variety_recommendation(self, query, params, user_context):
        """Generate crop variety recommendations"""
        # Extract crop type or use default
        crop_type = params.get("crop_type") or self._extract_crop_from_query(query) or "the crop"
        
        # Extract location
        location = params.get("location") or user_context.get("location") or "your region"
        
        # Extract specific requirements if mentioned
        requirements = params.get("requirements") or self._extract_variety_requirements(query) or "general purpose"
        
        # Create variety recommendation prompt
        prompt = f"""
        You are an agricultural research scientist specializing in Indian crop varieties.
        
        Recommend suitable varieties of {crop_type} for {location} with focus on {requirements}.
        
        Farmer's Query: {query}
        
        Include in your response:
        1. Top 3-5 recommended varieties with their characteristics
        2. Suitability for local conditions (climate, soil, pests)
        3. Maturity period and expected yield potential
        4. Special features (drought tolerance, disease resistance, etc.)
        5. Seed availability in India and approximate cost
        
        Focus on varieties that have been tested and proven in Indian conditions.
        Include both traditional varieties and improved cultivars/hybrids where appropriate.
        Consider both yield potential and risk factors for each recommendation.
        """
        
        # Generate response
        response = self.generate_response(prompt)
        
        # Confidence based on specificity
        confidence = 0.9 if params.get("crop_type") else 0.7
        
        return response, confidence
    
    def _generate_crop_rotation_advice(self, query, params, user_context):
        """Generate crop rotation recommendations"""
        # Extract crop type or use default
        crop_type = params.get("crop_type") or self._extract_crop_from_query(query) or "your current crop"
        
        # Extract location
        location = params.get("location") or user_context.get("location") or "your region"
        
        # Extract soil type if available
        soil_type = params.get("soil_type") or user_context.get("soil_type") or "your soil"
        
        # Create crop rotation prompt
        prompt = f"""
        You are an agricultural systems specialist with expertise in crop rotation for Indian farming systems.
        
        Recommend appropriate crop rotation sequences involving {crop_type} in {location} on {soil_type}.
        
        Farmer's Query: {query}
        
        Include in your response:
        1. Suitable crops to rotate with {crop_type} and their sequence
        2. Benefits of the recommended rotation pattern
        3. Timing considerations for the transition between crops
        4. Soil fertility impacts and management between rotations
        5. Pest and disease management benefits of the rotation
        
        Focus on rotation systems that are practical for Indian farmers.
        Consider both soil health benefits and economic viability of the rotation.
        Include both traditional rotation knowledge and scientific principles.
        """
        
        # Generate response
        response = self.generate_response(prompt)
        
        # Confidence based on specificity
        confidence = 0.85 if params.get("crop_type") else 0.7
        
        return response, confidence
    
    def _generate_general_crop_response(self, query, params, user_context):
        """Generate general crop information response"""
        # Extract crop type if available
        crop_type = params.get("crop_type") or self._extract_crop_from_query(query) or "crops in general"
        
        # Extract location
        location = params.get("location") or user_context.get("location") or "India"
        
        # Create general crop info prompt
        prompt = f"""
        You are an agricultural expert with comprehensive knowledge of crop production in India.
        
        Provide helpful information about {crop_type} cultivation in {location}.
        
        Farmer's Query: {query}
        
        Include relevant information such as:
        1. General characteristics and requirements of {crop_type}
        2. Typical growing season and cycle in {location}
        3. Basic cultivation practices and considerations
        4. Common challenges and general solutions
        5. Economic aspects and market potential
        
        Provide balanced, practical information relevant to Indian farming conditions.
        Include both traditional farming wisdom and modern agricultural science.
        """
        
        # Generate response
        response = self.generate_response(prompt)
        
        # Default confidence for general responses
        confidence = 0.7
        
        return response, confidence
    
    def _extract_crop_from_query(self, query):
        """Extract specific crop mentioned in the query"""
        query_lower = query.lower()
        
        # Common crops in India
        crops = [
            "rice", "wheat", "maize", "corn", "sorghum", "bajra", "jowar", "ragi",
            "barley", "pulses", "chickpea", "gram", "lentil", "pigeon pea", "arhar",
            "moong", "urad", "soybean", "groundnut", "peanut", "mustard", "rapeseed",
            "sunflower", "sesame", "cotton", "jute", "sugarcane", "potato", "onion",
            "tomato", "chili", "brinjal", "okra", "cauliflower", "cabbage", "cucumber",
            "pumpkin", "bitter gourd", "mango", "banana", "orange", "guava", "grape"
        ]
        
        # Check for crops in query
        for crop in crops:
            if crop in query_lower:
                return crop
        
        # Also check for crop types
        crop_types = {
            "cereal": ["cereal", "grain"],
            "pulse": ["pulse", "legume", "dal"],
            "oilseed": ["oilseed", "oil crop"],
            "vegetable": ["vegetable", "veggie"],
            "fruit": ["fruit", "orchard crop"],
            "cash crop": ["cash crop", "commercial crop"]
        }
        
        for crop_type, terms in crop_types.items():
            if any(term in query_lower for term in terms):
                return crop_type
        
        return None
    
    def _extract_growing_stage(self, query):
        """Extract growing stage mentioned in the query"""
        query_lower = query.lower()
        
        # Define growth stages
        stages = {
            "germination": ["germination", "sprouting", "seedling", "just planted", "emerging"],
            "vegetative": ["vegetative", "growing", "leafing", "development"],
            "flowering": ["flowering", "blooming", "reproductive"],
            "fruiting": ["fruiting", "fruit development", "pod development", "grain filling"],
            "maturation": ["maturation", "ripening", "maturing", "harvest ready"]
        }
        
        # Check for stages in query
        for stage, terms in stages.items():
            if any(term in query_lower for term in terms):
                return stage + " stage"
        
        return None
    
    def _extract_variety_requirements(self, query):
        """Extract variety requirements from the query"""
        query_lower = query.lower()
        
        # Define common requirements
        requirements = {
            "high yield": ["high yield", "more yield", "productive", "high production"],
            "drought tolerance": ["drought", "water scarcity", "dry", "less water", "drought resistant"],
            "disease resistance": ["disease", "resistant", "immunity", "pest resistant"],
            "short duration": ["short duration", "early maturing", "quick", "fast growing"],
            "quality": ["quality", "taste", "flavor", "nutritious", "protein", "oil content"]
        }
        
        # Check for requirements in query
        extracted_reqs = []
        for req, terms in requirements.items():
            if any(term in query_lower for term in terms):
                extracted_reqs.append(req)
        
        if extracted_reqs:
            return ", ".join(extracted_reqs)
        
        return None
    
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
