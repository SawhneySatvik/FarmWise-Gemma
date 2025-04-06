from app.agents.base_agent import ManagerAgent, LLMAgent
from app.agents.specialized.crop_agent import CropAgent
from app.agents.specialized.soil_agent import SoilAgent
from app.agents.specialized.weather_agent import WeatherAgent
from app.agents.specialized.irrigation_agent import IrrigationAgent
from app.agents.specialized.pest_disease_agent import PestDiseaseAgent
from app.agents.specialized.market_agent import MarketAgent

class CropManager(ManagerAgent, LLMAgent):
    """Manager agent that oversees crop-related specialized agents"""
    
    def __init__(self, name, llm_service):
        """
        Initialize CropManager
        
        Args:
            name (str): Agent name
            llm_service (LLMService): Service for LLM operations
        """
        # Initialize as both manager and LLM agent
        ManagerAgent.__init__(self, name)
        LLMAgent.__init__(self, name, llm_service)
        
        # Create and add specialized agents
        self.crop_agent = CropAgent("CropAgent", llm_service)
        self.soil_agent = SoilAgent("SoilAgent", llm_service)
        self.weather_agent = WeatherAgent("WeatherAgent", llm_service)
        self.irrigation_agent = IrrigationAgent("IrrigationAgent", llm_service)
        self.pest_disease_agent = PestDiseaseAgent("PestDiseaseAgent", llm_service)
        self.market_agent = MarketAgent("MarketAgent", llm_service)
        
        # Add agents to manager
        self.add_agent(self.crop_agent)
        self.add_agent(self.soil_agent)
        self.add_agent(self.weather_agent)
        self.add_agent(self.irrigation_agent)
        self.add_agent(self.pest_disease_agent)
        self.add_agent(self.market_agent)
    
    def process(self, query, context=None):
        """
        Process query by routing to appropriate specialized agents
        
        Args:
            query (str): The user query or request
            context (dict, optional): Additional context for processing
            
        Returns:
            dict: Response from the most appropriate agent(s)
        """
        # Get query type from context if available
        query_type = context.get("query_type", "general") if context else "general"
        
        # Log the incoming query
        self.log_activity(f"Routing {query_type} query", query[:50] + "..." if len(query) > 50 else query)
        
        # For MVP, we'll use a simple routing strategy
        if query_type == "crop":
            # Direct crop queries to crop agent
            return self.crop_agent.process(query, context)
        elif query_type == "soil":
            # Soil queries to soil agent
            return self.soil_agent.process(query, context)
        elif query_type == "weather":
            # Weather queries to weather agent
            return self.weather_agent.process(query, context)
        elif query_type == "irrigation":
            # Irrigation queries to irrigation agent
            return self.irrigation_agent.process(query, context)
        elif query_type == "pest_disease":
            # Pest and disease queries to pest/disease agent
            return self.pest_disease_agent.process(query, context)
        elif query_type == "market":
            # Market queries to market agent
            return self.market_agent.process(query, context)
        else:
            # For other queries, determine which agents to use based on confidence
            
            # First, process with all agents
            results = []
            for agent in self.agents:
                try:
                    result = agent.process(query, context)
                    results.append(result)
                except Exception as e:
                    self.log_activity(f"Error with {agent.name}", str(e))
            
            # Filter results by minimum confidence
            valid_results = [r for r in results if r.get("confidence", 0) >= 0.4]
            
            if not valid_results:
                # If no agent is confident, provide a general response
                general_response = self.generate_response(query, context.get("user_context") if context else None)
                return self.format_response(
                    general_response,
                    0.7,
                    {"query_type": "general", "contributing_agents": [self.name]}
                )
            
            # Sort by confidence
            valid_results.sort(key=lambda x: x.get("confidence", 0), reverse=True)
            
            # If the top result has high confidence, return it
            if valid_results[0].get("confidence", 0) >= 0.7:
                return valid_results[0]
            
            # Otherwise, combine the top results
            top_results = valid_results[:2]  # Take top 2 results
            combined_content = self._combine_responses(query, [r["content"] for r in top_results])
            
            return self.format_response(
                combined_content,
                max(r.get("confidence", 0) for r in top_results),
                {"contributing_agents": [r["agent"] for r in top_results]}
            )
    
    def _combine_responses(self, query, responses):
        """
        Combine multiple agent responses into a coherent answer
        
        Args:
            query (str): Original query
            responses (list): List of response texts to combine
            
        Returns:
            str: Combined response
        """
        # Create a prompt to combine responses
        combine_prompt = f"""
        The following responses were generated for this farmer's query:
        
        QUERY: {query}
        
        RESPONSE 1: {responses[0]}
        
        RESPONSE 2: {responses[1]}
        
        Combine these responses into a single coherent answer that includes the most relevant 
        information from both. Make the response flow naturally as if it came from a single source.
        """
        
        # Generate combined response
        return self.generate_response(combine_prompt)
