from app.agents.base_agent import LLMAgent

class PestDiseaseAgent(LLMAgent):
    """Agent for pest and disease identification and management recommendations"""
    
    def __init__(self, name, llm_service):
        """Initialize pest and disease agent with LLM service"""
        super().__init__(name, llm_service)
    
    def process(self, query, context=None):
        """
        Process pest and disease related queries
        
        Args:
            query (str): The user query or request
            context (dict, optional): Additional context for processing
            
        Returns:
            dict: Formatted response with pest/disease recommendations
        """
        # Extract parameters from context if available
        params = {}
        if context and "extracted_params" in context:
            params = context["extracted_params"]
        
        # Get user context if available
        user_context = context.get("user_context", {}) if context else {}
        
        # Determine the type of pest/disease query
        query_intent = self._determine_pest_disease_intent(query, params)
        
        # Generate appropriate response based on intent
        if query_intent == "identification":
            response, confidence = self._generate_identification_response(query, params, user_context)
        elif query_intent == "prevention":
            response, confidence = self._generate_prevention_response(query, params, user_context)
        elif query_intent == "treatment":
            response, confidence = self._generate_treatment_response(query, params, user_context)
        elif query_intent == "organic":
            response, confidence = self._generate_organic_solution_response(query, params, user_context)
        elif query_intent == "lifecycle":
            response, confidence = self._generate_lifecycle_response(query, params, user_context)
        else:
            # General pest/disease query
            response, confidence = self._generate_general_pd_response(query, params, user_context)
        
        # Return formatted response
        return self.format_response(
            response,
            confidence,
            {"intent": query_intent, "crop_type": params.get("crop_type"), "pest_disease": params.get("pest_disease")}
        )
    
    def _determine_pest_disease_intent(self, query, params):
        """Determine the intent of the pest/disease-related query"""
        query_lower = query.lower()
        
        # Check for identification intent
        if any(term in query_lower for term in ["identify", "what is", "what pest", "what disease", "diagnosis", "symptoms", "recognize"]):
            return "identification"
        
        # Check for prevention intent
        if any(term in query_lower for term in ["prevent", "avoid", "protect", "before", "stop from", "keep away"]):
            return "prevention"
        
        # Check for treatment intent
        if any(term in query_lower for term in ["treat", "cure", "control", "manage", "get rid", "solution", "spray", "chemical"]):
            return "treatment"
        
        # Check for organic solutions intent
        if any(term in query_lower for term in ["organic", "natural", "without chemical", "non-chemical", "traditional", "home remedy"]):
            return "organic"
        
        # Check for lifecycle/biology intent
        if any(term in query_lower for term in ["lifecycle", "life cycle", "biology", "breed", "reproduce", "eggs", "growth", "live"]):
            return "lifecycle"
        
        # Default to general
        return "general"
    
    def _generate_identification_response(self, query, params, user_context):
        """Generate pest/disease identification response"""
        # Extract crop type or use default
        crop_type = params.get("crop_type") or user_context.get("primary_crops") or "your crops"
        
        # Extract symptoms if mentioned
        symptoms = self._extract_symptoms_from_query(query) or "the symptoms you're observing"
        
        # Extract pest/disease if already mentioned
        pest_disease = self._extract_pest_disease_from_query(query) or params.get("pest_disease")
        
        # Create identification prompt
        if pest_disease:
            # If pest/disease is already mentioned, provide information about it
            prompt = f"""
            You are a plant pathologist and entomologist specializing in agricultural pests and diseases in India.
            
            Provide detailed identification information about {pest_disease} affecting {crop_type}.
            
            Farmer's Query: {query}
            
            Include in your response:
            1. Detailed description of {pest_disease} and its characteristics
            2. Distinctive symptoms and signs on {crop_type}
            3. Conditions that favor this pest/disease
            4. Look-alike issues that might be confused with it
            5. Critical times when identification is most important
            
            Focus on visual cues and diagnostic features that Indian farmers can use for field identification.
            Include both scientific information and traditional recognition methods used by farmers.
            Emphasize practical identification techniques that don't require special equipment.
            """
        else:
            # If symptoms are provided, help identify the pest/disease
            prompt = f"""
            You are a plant pathologist and entomologist specializing in agricultural pests and diseases in India.
            
            Based on these symptoms: {symptoms}, identify possible pests or diseases affecting {crop_type}.
            
            Farmer's Query: {query}
            
            Include in your response:
            1. Top 2-3 possible pests/diseases matching these symptoms
            2. Distinguishing features of each possibility
            3. Additional observations the farmer should make to confirm identification
            4. Images or visual cues to look for (described in detail)
            5. Urgency of action needed based on these symptoms
            
            Focus on common pests and diseases affecting these crops in India.
            Explain how the farmer can differentiate between similar-looking problems.
            Prioritize identification of serious issues that require immediate attention.
            """
        
        # Generate response
        response = self.generate_response(prompt)
        
        # Confidence based on specificity
        if pest_disease:
            confidence = 0.85
        elif symptoms:
            confidence = 0.7
        else:
            confidence = 0.6
        
        return response, confidence
    
    def _generate_prevention_response(self, query, params, user_context):
        """Generate pest/disease prevention recommendations"""
        # Extract crop type or use default
        crop_type = params.get("crop_type") or user_context.get("primary_crops") or "your crops"
        
        # Extract specific pest/disease if mentioned
        pest_disease = self._extract_pest_disease_from_query(query) or params.get("pest_disease") or "common pests and diseases"
        
        # Create prevention prompt
        prompt = f"""
        You are an integrated pest management (IPM) specialist with expertise in Indian agricultural systems.
        
        Provide detailed prevention strategies for {pest_disease} in {crop_type}.
        
        Farmer's Query: {query}
        
        Include in your response:
        1. Cultural practices that prevent infestation/infection
        2. Resistant varieties available in India
        3. Preventive biological controls appropriate for Indian conditions
        4. Preventive spray schedules (if applicable)
        5. Monitoring techniques to detect early signs
        
        Focus on preventive approaches that are practical for Indian farming conditions.
        Balance traditional preventive methods with modern scientific approaches.
        Emphasize cost-effective solutions that can be implemented with locally available resources.
        """
        
        # Generate response
        response = self.generate_response(prompt)
        
        # Confidence based on specificity
        confidence = 0.8 if self._extract_pest_disease_from_query(query) else 0.7
        
        return response, confidence
    
    def _generate_treatment_response(self, query, params, user_context):
        """Generate pest/disease treatment recommendations"""
        # Extract crop type or use default
        crop_type = params.get("crop_type") or user_context.get("primary_crops") or "your crops"
        
        # Extract specific pest/disease if mentioned
        pest_disease = self._extract_pest_disease_from_query(query) or params.get("pest_disease") or "the pest/disease issue"
        
        # Create treatment prompt
        prompt = f"""
        You are an agricultural plant protection specialist with expertise in pest and disease management in India.
        
        Provide effective treatment options for controlling {pest_disease} in {crop_type}.
        
        Farmer's Query: {query}
        
        Include in your response:
        1. Chemical control options available in India (specific products and active ingredients)
        2. Application methods and timing
        3. Safety precautions and waiting periods
        4. Biological control alternatives
        5. Integrated management approach combining different methods
        
        Focus on treatments that are legally approved for use in India.
        Provide specific dilution rates and application techniques.
        Include both conventional and alternative treatment approaches.
        """
        
        # Generate response
        response = self.generate_response(prompt)
        
        # Confidence based on specificity
        confidence = 0.85 if self._extract_pest_disease_from_query(query) else 0.7
        
        return response, confidence
    
    def _generate_organic_solution_response(self, query, params, user_context):
        """Generate organic pest/disease control recommendations"""
        # Extract crop type or use default
        crop_type = params.get("crop_type") or user_context.get("primary_crops") or "your crops"
        
        # Extract specific pest/disease if mentioned
        pest_disease = self._extract_pest_disease_from_query(query) or params.get("pest_disease") or "pest and disease issues"
        
        # Create organic solutions prompt
        prompt = f"""
        You are an organic farming specialist with expertise in traditional and natural pest management methods used in India.
        
        Provide organic and natural control methods for {pest_disease} affecting {crop_type}.
        
        Farmer's Query: {query}
        
        Include in your response:
        1. Traditional botanical preparations used in India (neem, garlic, etc.)
        2. Home-made organic sprays and their preparation methods
        3. Biological control options (beneficial insects, microorganisms)
        4. Cultural practices that suppress without chemicals
        5. Organic commercial products available in India
        
        Focus on solutions that use locally available materials.
        Provide specific preparation methods and application instructions.
        Include traditional knowledge from Indian farming systems.
        Emphasize methods that have been validated through experience or research.
        """
        
        # Generate response
        response = self.generate_response(prompt)
        
        # Confidence based on specificity
        confidence = 0.8 if self._extract_pest_disease_from_query(query) else 0.7
        
        return response, confidence
    
    def _generate_lifecycle_response(self, query, params, user_context):
        """Generate pest/disease lifecycle information"""
        # Extract specific pest/disease
        pest_disease = self._extract_pest_disease_from_query(query) or params.get("pest_disease")
        
        if not pest_disease:
            # If no specific pest/disease mentioned, provide general lifecycle info
            response, confidence = self._generate_general_pd_response(query, params, user_context)
            return response, confidence
        
        # Create lifecycle prompt
        prompt = f"""
        You are an entomologist and plant pathologist specializing in pest and disease biology in Indian agricultural systems.
        
        Explain the lifecycle and biology of {pest_disease} in the context of agricultural impacts.
        
        Farmer's Query: {query}
        
        Include in your response:
        1. Complete lifecycle stages of {pest_disease}
        2. Duration of each lifecycle stage
        3. Environmental conditions that accelerate or slow development
        4. Most vulnerable stages in the lifecycle
        5. How understanding the lifecycle helps with control timing
        
        Focus on practical lifecycle information relevant to farming decisions.
        Explain how knowledge of the lifecycle can be used for better management.
        Include specific details about how the pest/disease behaves in Indian climatic conditions.
        """
        
        # Generate response
        response = self.generate_response(prompt)
        
        # Confidence based on having specific pest/disease
        confidence = 0.9 if pest_disease else 0.6
        
        return response, confidence
    
    def _generate_general_pd_response(self, query, params, user_context):
        """Generate general pest/disease information response"""
        # Extract crop type if available
        crop_type = params.get("crop_type") or user_context.get("primary_crops") or "crops in general"
        
        # Extract location
        location = params.get("location") or user_context.get("location") or "India"
        
        # Create general pest/disease info prompt
        prompt = f"""
        You are an agricultural pest and disease management specialist with extensive knowledge of Indian farming systems.
        
        Provide helpful information about managing pests and diseases for {crop_type} in {location}.
        
        Farmer's Query: {query}
        
        Include relevant information such as:
        1. Common pest and disease challenges for {crop_type} in this region
        2. Basic principles of integrated pest management (IPM)
        3. Preventive cultural practices that reduce pest/disease pressure
        4. When to seek professional help for identification or management
        5. Resources available to Indian farmers for pest/disease management
        
        Provide balanced information covering both prevention and control.
        Include both traditional knowledge and modern scientific approaches.
        Focus on practical information relevant to Indian farmers.
        """
        
        # Generate response
        response = self.generate_response(prompt)
        
        # Default confidence for general responses
        confidence = 0.7
        
        return response, confidence
    
    def _extract_pest_disease_from_query(self, query):
        """Extract specific pest or disease mentioned in the query"""
        query_lower = query.lower()
        
        # Common pests in India
        pests = [
            "aphid", "thrips", "whitefly", "jassid", "leafhopper", "mealybug", "scale insect",
            "fruit fly", "stem borer", "pod borer", "bollworm", "armyworm", "caterpillar",
            "grasshopper", "locust", "termite", "weevil", "mite", "nematode", "rodent"
        ]
        
        # Common diseases in India
        diseases = [
            "blight", "blast", "rust", "smut", "mildew", "powdery mildew", "downy mildew",
            "wilt", "leaf spot", "anthracnose", "mosaic", "yellow mosaic", "leaf curl",
            "rot", "root rot", "collar rot", "damping off", "bacterial leaf", "viral"
        ]
        
        # Check for specific pests
        for pest in pests:
            if pest in query_lower:
                return pest
        
        # Check for specific diseases
        for disease in diseases:
            if disease in query_lower:
                return disease
        
        # Look for general terms
        general_terms = {
            "pest": ["pest", "insect", "bug"],
            "disease": ["disease", "infection", "fungus", "bacteria", "virus", "pathogen"]
        }
        
        for category, terms in general_terms.items():
            if any(term in query_lower for term in terms):
                return category
        
        return None
    
    def _extract_symptoms_from_query(self, query):
        """Extract symptoms mentioned in the query"""
        query_lower = query.lower()
        
        # Common symptom keywords
        symptom_indicators = [
            "symptom", "sign", "showing", "appears", "look like", "seeing", "notice",
            "observe", "found", "yellowing", "wilting", "spots", "holes", "damage",
            "dying", "stunted", "curling", "rotting", "discolored", "infected"
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
