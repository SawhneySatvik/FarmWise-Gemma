from app.agents.base_agent import LLMAgent

class FeedAgent(LLMAgent):
    """Agent for livestock feed and nutrition management recommendations"""
    
    def __init__(self, name, llm_service):
        """Initialize feed agent with LLM service"""
        super().__init__(name, llm_service)
    
    def process(self, query, context=None):
        """
        Process livestock feed-related queries
        
        Args:
            query (str): The user query or request
            context (dict, optional): Additional context for processing
            
        Returns:
            dict: Formatted response with feed recommendations
        """
        # Extract parameters from context if available
        params = {}
        if context and "extracted_params" in context:
            params = context["extracted_params"]
        
        # Get user context if available
        user_context = context.get("user_context", {}) if context else {}
        
        # Determine the type of feed query
        query_intent = self._determine_feed_intent(query, params)
        
        # Generate appropriate response based on intent
        if query_intent == "ration_formulation":
            response, confidence = self._generate_ration_formulation(query, params, user_context)
        elif query_intent == "diet_adjustment":
            response, confidence = self._generate_diet_adjustment(query, params, user_context)
        elif query_intent == "nutrition_deficiency":
            response, confidence = self._generate_nutrition_deficiency_advice(query, params, user_context)
        elif query_intent == "local_feed":
            response, confidence = self._generate_local_feed_advice(query, params, user_context)
        elif query_intent == "feed_storage":
            response, confidence = self._generate_feed_storage_advice(query, params, user_context)
        else:
            # General feed query
            response, confidence = self._generate_general_feed_response(query, params, user_context)
        
        # Return formatted response
        return self.format_response(
            response,
            confidence,
            {"intent": query_intent, "livestock_type": params.get("livestock_type")}
        )
    
    def _determine_feed_intent(self, query, params):
        """Determine the intent of the feed-related query"""
        query_lower = query.lower()
        
        # Check for ration formulation intent
        if any(term in query_lower for term in ["ration", "formulate", "diet", "feed mix", "balanced diet", "feed formula"]):
            return "ration_formulation"
        
        # Check for diet adjustment intent
        if any(term in query_lower for term in ["adjust", "change", "modify", "improve", "enhance", "optimize", "upgrade feed"]):
            return "diet_adjustment"
        
        # Check for nutrition deficiency intent
        if any(term in query_lower for term in ["deficiency", "lacking", "insufficient", "symptoms", "health issue", "not eating", "weak"]):
            return "nutrition_deficiency"
        
        # Check for local feed resources intent
        if any(term in query_lower for term in ["local feed", "available feed", "low cost", "cheap", "affordable", "alternative feed"]):
            return "local_feed"
        
        # Check for feed storage intent
        if any(term in query_lower for term in ["store", "storage", "preserve", "keep", "spoilage", "shelf life", "silage", "hay"]):
            return "feed_storage"
        
        # Default to general
        return "general"
    
    def _generate_ration_formulation(self, query, params, user_context):
        """Generate livestock ration formulation recommendations"""
        # Extract livestock type or use default
        livestock_type = params.get("livestock_type") or self._extract_livestock_from_query(query) or "your livestock"
        
        # Extract production purpose if available
        purpose = params.get("purpose") or self._extract_purpose_from_query(query) or "general production"
        
        # Extract age/stage if available
        life_stage = params.get("life_stage") or self._extract_life_stage_from_query(query) or "adult"
        
        # Create ration formulation prompt
        prompt = f"""
        You are a livestock nutrition specialist with expertise in feed formulation for Indian farming systems.
        
        Provide a detailed feed ration formulation for {livestock_type} used for {purpose} at {life_stage} stage.
        
        Farmer's Query: {query}
        
        Include in your response:
        1. Essential nutrient requirements for this specific animal and purpose
        2. Suggested ingredients with approximate proportions
        3. Daily feeding amounts and schedule
        4. Critical minerals and vitamins to include
        5. Locally available feed alternatives in India
        
        Focus on practical formulations that can be implemented by Indian farmers.
        Include both commercial and farm-made feed options.
        Consider the economic aspects of different feed formulations.
        """
        
        # Generate response
        response = self.generate_response(prompt)
        
        # Confidence based on specificity
        confidence = 0.85 if (params.get("livestock_type") and params.get("purpose")) else 0.7
        
        return response, confidence
    
    def _generate_diet_adjustment(self, query, params, user_context):
        """Generate recommendations for adjusting livestock diet"""
        # Extract livestock type or use default
        livestock_type = params.get("livestock_type") or self._extract_livestock_from_query(query) or "your livestock"
        
        # Extract reason for adjustment if available
        reason = params.get("reason") or self._extract_reason_from_query(query) or "improved performance"
        
        # Extract current diet if available
        current_diet = params.get("current_diet") or "their current diet"
        
        # Create diet adjustment prompt
        prompt = f"""
        You are a livestock nutrition expert specializing in diet optimization for Indian farming conditions.
        
        Provide recommendations for adjusting the diet of {livestock_type} for {reason}, currently fed with {current_diet}.
        
        Farmer's Query: {query}
        
        Include in your response:
        1. Analysis of potential issues with the current diet
        2. Specific adjustments to make (ingredients to add, remove, or change proportions)
        3. Gradual transition plan to avoid digestive upset
        4. Expected benefits from these dietary changes
        5. Cost-benefit considerations of the adjustments
        
        Focus on practical changes that are accessible to Indian farmers.
        Provide specific dietary adjustments rather than general feeding principles.
        Consider seasonal availability of feed ingredients in India.
        """
        
        # Generate response
        response = self.generate_response(prompt)
        
        # Confidence based on specificity
        confidence = 0.8 if self._extract_livestock_from_query(query) else 0.7
        
        return response, confidence
    
    def _generate_nutrition_deficiency_advice(self, query, params, user_context):
        """Generate advice for addressing livestock nutritional deficiencies"""
        # Extract livestock type or use default
        livestock_type = params.get("livestock_type") or self._extract_livestock_from_query(query) or "your livestock"
        
        # Extract symptoms if available
        symptoms = params.get("symptoms") or self._extract_symptoms_from_query(query) or "the symptoms you're observing"
        
        # Create nutrition deficiency prompt
        prompt = f"""
        You are a veterinary nutritionist specializing in livestock health and nutrition in India.
        
        Provide advice on potential nutritional deficiencies in {livestock_type} showing these symptoms: {symptoms}.
        
        Farmer's Query: {query}
        
        Include in your response:
        1. Possible nutrient deficiencies based on the symptoms
        2. Feed adjustments to address these deficiencies
        3. Supplement options available in India
        4. Immediate interventions for severe cases
        5. Long-term dietary changes to prevent recurrence
        
        Focus on practical solutions accessible to Indian farmers.
        Differentiate between symptoms requiring veterinary intervention and those that can be addressed through nutrition.
        Include both commercial and natural/traditional remedies.
        """
        
        # Generate response
        response = self.generate_response(prompt)
        
        # Confidence based on specificity
        confidence = 0.75 if self._extract_symptoms_from_query(query) else 0.65
        
        return response, confidence
    
    def _generate_local_feed_advice(self, query, params, user_context):
        """Generate advice on using locally available feed resources"""
        # Extract livestock type or use default
        livestock_type = params.get("livestock_type") or self._extract_livestock_from_query(query) or "your livestock"
        
        # Extract location information
        location = params.get("location") or user_context.get("location") or "your region"
        
        # Create local feed resources prompt
        prompt = f"""
        You are a livestock feed specialist with expertise in alternative and local feed resources in India.
        
        Provide information on using locally available feed resources for {livestock_type} in {location}.
        
        Farmer's Query: {query}
        
        Include in your response:
        1. Locally available feed resources commonly found in this region
        2. Nutritional value of these resources for {livestock_type}
        3. Proper processing methods to improve digestibility and safety
        4. Appropriate inclusion rates in the diet
        5. Potential limitations or toxicity concerns
        
        Focus on economical and sustainable feed alternatives.
        Include crop residues, by-products, and unconventional feed resources.
        Consider seasonal availability and storage options.
        Provide specific preparation methods where relevant.
        """
        
        # Generate response
        response = self.generate_response(prompt)
        
        # Confidence based on specificity
        confidence = 0.8 if params.get("livestock_type") else 0.7
        
        return response, confidence
    
    def _generate_feed_storage_advice(self, query, params, user_context):
        """Generate advice on feed storage and preservation"""
        # Extract feed type if available
        feed_type = params.get("feed_type") or self._extract_feed_type_from_query(query) or "feed resources"
        
        # Extract location for contextual advice
        location = params.get("location") or user_context.get("location") or "your region"
        
        # Create feed storage prompt
        prompt = f"""
        You are a feed management specialist with expertise in storage and preservation techniques for Indian conditions.
        
        Provide detailed advice on storing and preserving {feed_type} in {location}.
        
        Farmer's Query: {query}
        
        Include in your response:
        1. Proper storage conditions (moisture, temperature, ventilation)
        2. Preservation methods suitable for Indian climate
        3. Prevention of mold, pests, and nutrient degradation
        4. Low-cost storage structures or containers
        5. Signs of spoilage to monitor and safety considerations
        
        Focus on practical methods suitable for Indian farming conditions.
        Consider both traditional and modern storage approaches.
        Include season-specific storage challenges and solutions.
        Provide specific guidelines for different storage durations.
        """
        
        # Generate response
        response = self.generate_response(prompt)
        
        # Confidence level
        confidence = 0.8
        
        return response, confidence
    
    def _generate_general_feed_response(self, query, params, user_context):
        """Generate general livestock feed information response"""
        # Extract livestock type if available
        livestock_type = params.get("livestock_type") or self._extract_livestock_from_query(query) or "livestock in general"
        
        # Create general feed info prompt
        prompt = f"""
        You are a livestock nutrition expert with deep knowledge of feeding practices in Indian agricultural systems.
        
        Provide helpful information about feeding and nutrition for {livestock_type}.
        
        Farmer's Query: {query}
        
        Include relevant information such as:
        1. Basic nutritional requirements for {livestock_type}
        2. Common feed ingredients and their benefits
        3. Feeding practices for optimal health and production
        4. Economic aspects of feed management
        5. Common feeding mistakes to avoid
        
        Provide practical, science-based advice relevant to Indian farming conditions.
        Balance traditional feeding knowledge with modern nutritional science.
        """
        
        # Generate response
        response = self.generate_response(prompt)
        
        # Default confidence for general responses
        confidence = 0.7
        
        return response, confidence
    
    def _extract_livestock_from_query(self, query):
        """Extract livestock type mentioned in the query"""
        query_lower = query.lower()
        
        # Common livestock in India
        livestock = {
            "cattle": ["cow", "cattle", "bull", "ox", "buffalo", "dairy animal"],
            "goat": ["goat", "bakri"],
            "sheep": ["sheep", "lamb", "ewe", "ram"],
            "poultry": ["chicken", "hen", "rooster", "bird", "poultry", "duck", "fowl"],
            "pig": ["pig", "swine", "hog"],
            "fish": ["fish", "aquaculture", "fishery"]
        }
        
        # Check for livestock in query
        for animal, terms in livestock.items():
            if any(term in query_lower for term in terms):
                return animal
        
        return None
    
    def _extract_purpose_from_query(self, query):
        """Extract production purpose mentioned in the query"""
        query_lower = query.lower()
        
        # Common production purposes
        purposes = {
            "dairy": ["milk", "dairy", "milking", "lactation"],
            "meat": ["meat", "beef", "mutton", "pork", "broiler", "fattening"],
            "dual purpose": ["dual purpose", "both milk and meat"],
            "draft": ["draft", "work", "plowing", "labor"],
            "breeding": ["breeding", "reproduction", "pregnant", "pregnancy"],
            "egg": ["egg", "layer", "laying"],
            "wool": ["wool", "fiber"]
        }
        
        # Check for purposes in query
        for purpose, terms in purposes.items():
            if any(term in query_lower for term in terms):
                return purpose
        
        return None
    
    def _extract_life_stage_from_query(self, query):
        """Extract animal life stage mentioned in the query"""
        query_lower = query.lower()
        
        # Common life stages
        stages = {
            "calf/kid/lamb/chick": ["calf", "kid", "lamb", "chick", "newborn", "baby", "young"],
            "growing": ["growing", "grower", "juvenile", "young", "adolescent"],
            "adult": ["adult", "mature", "full-grown"],
            "lactating": ["lactating", "milking", "in milk", "dairy"],
            "pregnant": ["pregnant", "gestation", "expecting"],
            "dry": ["dry", "non-lactating", "non-milking"],
            "old": ["old", "aged", "elderly", "senior"]
        }
        
        # Check for stages in query
        for stage, terms in stages.items():
            if any(term in query_lower for term in terms):
                return stage
        
        return None
    
    def _extract_reason_from_query(self, query):
        """Extract reason for diet adjustment from query"""
        query_lower = query.lower()
        
        # Common reasons for adjustment
        reasons = {
            "increase production": ["increase production", "more milk", "more eggs", "more meat", "better yield", "higher output"],
            "weight gain": ["weight gain", "fatten", "growth", "increase weight", "body weight"],
            "health improvement": ["health", "improve health", "recovery", "sick", "disease", "immunity"],
            "cost reduction": ["cost", "cheaper", "reduce cost", "save money", "economical", "budget"],
            "seasonal change": ["season", "summer", "winter", "monsoon", "seasonal", "weather change"],
            "life stage change": ["growing", "pregnancy", "lactation", "dry period", "old age", "young", "adult"]
        }
        
        # Check for reasons in query
        for reason, terms in reasons.items():
            if any(term in query_lower for term in terms):
                return reason
        
        return None
    
    def _extract_symptoms_from_query(self, query):
        """Extract symptoms mentioned in the query"""
        query_lower = query.lower()
        
        # Common symptom keywords
        symptom_indicators = [
            "symptom", "sign", "showing", "appears", "look like", "seeing", "notice",
            "observe", "found", "weak", "thin", "not eating", "loss", "poor", "problem",
            "sick", "ill", "unhealthy", "condition", "disorder"
        ]
        
        # Check if query contains symptom indicators
        if any(indicator in query_lower for indicator in symptom_indicators):
            # Extract the part of the query that likely contains symptom description
            # This is a simplified approach - in production would use more sophisticated NLP
            for indicator in symptom_indicators:
                if indicator in query_lower:
                    start_idx = query_lower.find(indicator)
                    # Extract a reasonable chunk after the indicator
                    end_idx = min(start_idx + 100, len(query_lower))
                    symptom_text = query[start_idx:end_idx]
                    return symptom_text
        
        return None
    
    def _extract_feed_type_from_query(self, query):
        """Extract feed type mentioned in the query"""
        query_lower = query.lower()
        
        # Common feed types
        feed_types = {
            "hay": ["hay", "dried grass", "dried fodder"],
            "silage": ["silage", "fermented", "ensiled"],
            "straw": ["straw", "bhusa", "stubble", "crop residue"],
            "concentrate": ["concentrate", "feed mix", "compound feed", "pellet", "mash"],
            "green fodder": ["green fodder", "grass", "fresh forage", "grazing", "chaff", "chara"],
            "grain": ["grain", "cereal", "maize", "corn", "wheat", "barley", "oat"],
            "by-products": ["by-product", "bran", "oil cake", "molasses", "bagasse"]
        }
        
        # Check for feed types in query
        for feed_type, terms in feed_types.items():
            if any(term in query_lower for term in terms):
                return feed_type
        
        return None
