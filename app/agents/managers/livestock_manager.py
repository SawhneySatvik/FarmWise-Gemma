from app.agents.base_agent import ManagerAgent, LLMAgent
from app.agents.specialized.feed_agent import FeedAgent
from app.agents.specialized.pest_disease_agent import PestDiseaseAgent
from app.agents.specialized.market_agent import MarketAgent

class LivestockManager(ManagerAgent, LLMAgent):
    """Manager agent that oversees livestock-related specialized agents"""
    
    def __init__(self, name, llm_service):
        """
        Initialize LivestockManager
        
        Args:
            name (str): Agent name
            llm_service (LLMService): Service for LLM operations
        """
        # Initialize as both manager and LLM agent
        ManagerAgent.__init__(self, name)
        LLMAgent.__init__(self, name, llm_service)
        
        # Create specialized agents for livestock
        # Note: We reuse some agents from crop management
        self.feed_agent = FeedAgent("FeedAgent", llm_service)
        self.disease_agent = PestDiseaseAgent("LivestockDiseaseAgent", llm_service)  # Reuse pest agent for livestock diseases
        self.market_agent = MarketAgent("LivestockMarketAgent", llm_service)  # Reuse market agent
        
        # Add agents to manager
        self.add_agent(self.feed_agent)
        self.add_agent(self.disease_agent)
        self.add_agent(self.market_agent)
    
    def process(self, query, context=None):
        """
        Process livestock-related queries
        
        Args:
            query (str): The user query or request
            context (dict, optional): Additional context for processing
            
        Returns:
            dict: Response from appropriate livestock agent(s)
        """
        # Log the incoming query
        self.log_activity("Processing livestock query", query[:50] + "..." if len(query) > 50 else query)
        
        # For MVP, we'll use a simple keyword-based approach to route to specialized agents
        query_lower = query.lower()
        
        # Route feed-related queries
        if any(term in query_lower for term in ["feed", "fodder", "nutrition", "diet", "grass", "hay", "silage"]):
            return self.feed_agent.process(query, context)
        
        # Route disease-related queries
        if any(term in query_lower for term in ["disease", "sick", "health", "infection", "parasite", "treatment", "vaccine"]):
            return self.disease_agent.process(query, context)
        
        # Route market-related queries
        if any(term in query_lower for term in ["market", "price", "sell", "buy", "cost", "profit", "demand"]):
            return self.market_agent.process(query, context)
        
        # For general livestock queries, try all agents and select best response
        results = []
        for agent in self.agents:
            try:
                result = agent.process(query, context)
                results.append(result)
            except Exception as e:
                self.log_activity(f"Error with {agent.name}", str(e))
        
        # Filter results by minimum confidence
        valid_results = [r for r in results if r.get("confidence", 0) >= 0.3]
        
        if not valid_results:
            # If no agent is confident, provide a general livestock response using LLM
            return self._generate_general_livestock_response(query, context)
        
        # Sort by confidence
        valid_results.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        
        # Return the highest confidence result
        return valid_results[0]
    
    def _generate_general_livestock_response(self, query, context):
        """Generate a general livestock response when specialized agents aren't confident"""
        # Extract livestock type if present
        livestock_type = self._extract_livestock_type(query)
        
        # Create prompt with livestock context
        livestock_prompt = f"""
        You are a livestock specialist assistant helping Indian farmers with their livestock 
        questions. Provide a helpful response to this query about {livestock_type or 'livestock'}.
        
        QUERY: {query}
        
        Provide specific advice relevant to Indian farming conditions, including:
        1. Appropriate management practices
        2. Common health considerations
        3. Feed recommendations if relevant
        4. Economic considerations if relevant
        
        Base your response on established animal husbandry practices suitable for Indian conditions.
        """
        
        # Generate response
        response = self.generate_response(livestock_prompt, context.get("user_context") if context else None)
        
        return self.format_response(
            response,
            0.8,  # Confidence for general livestock responses
            {"livestock_type": livestock_type, "query_type": "general_livestock"}
        )
    
    def _extract_livestock_type(self, query):
        """Extract the type of livestock from the query"""
        query_lower = query.lower()
        
        livestock_mapping = {
            "cattle": ["cow", "bull", "cattle", "calf", "calves", "bovine"],
            "buffalo": ["buffalo", "buffaloes", "water buffalo"],
            "goat": ["goat", "kid", "buck", "doe"],
            "sheep": ["sheep", "lamb", "ewe", "ram"],
            "poultry": ["chicken", "hen", "rooster", "chick", "poultry", "bird", "duck", "geese"],
            "pig": ["pig", "swine", "hog", "piglet", "sow", "boar"]
        }
        
        for livestock, terms in livestock_mapping.items():
            if any(term in query_lower for term in terms):
                return livestock
        
        return None
