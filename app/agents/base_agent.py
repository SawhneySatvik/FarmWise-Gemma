from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """Abstract base class for all farming agents"""
    
    def __init__(self, name):
        """Initialize base agent with name and empty state"""
        self.name = name
        self.state = {}
    
    @abstractmethod
    def process(self, query, context=None):
        """
        Process a query with optional context
        
        Args:
            query (str): The user query or request
            context (dict, optional): Additional context for processing
            
        Returns:
            dict: Response with results and metadata
        """
        pass
    
    @property
    def agent_type(self):
        """Get the agent type"""
        return self.__class__.__name__
    
    def reset_state(self):
        """Reset the agent's state"""
        self.state = {}
    
    def update_state(self, key, value):
        """Update a specific state value"""
        self.state[key] = value
    
    def get_state(self, key, default=None):
        """Get a specific state value, or default if not found"""
        return self.state.get(key, default)
    
    def format_response(self, content, confidence=1.0, metadata=None):
        """
        Format a standard agent response
        
        Args:
            content (str): Main response content
            confidence (float): Confidence score (0.0 to 1.0)
            metadata (dict, optional): Additional metadata
            
        Returns:
            dict: Formatted response
        """
        response = {
            "agent": self.name,
            "agent_type": self.agent_type,
            "content": content,
            "confidence": confidence
        }
        
        if metadata:
            response["metadata"] = metadata
        
        return response
    
    def log_activity(self, action, details=None):
        """Log agent activity (placeholder for future logging)"""
        # For MVP, just print to console
        # In production, implement proper logging to database or log files
        print(f"[{self.agent_type}:{self.name}] {action}")
        if details:
            print(f"  Details: {details}")


class ManagerAgent(BaseAgent):
    """Base class for manager agents that coordinate specialized agents"""
    
    def __init__(self, name):
        """Initialize manager agent with name and empty agents list"""
        super().__init__(name)
        self.agents = []
    
    def add_agent(self, agent):
        """Add a specialized agent to this manager"""
        self.agents.append(agent)
    
    def process(self, query, context=None):
        """
        Process query by delegating to specialized agents
        
        Args:
            query (str): The user query or request
            context (dict, optional): Additional context for processing
            
        Returns:
            dict: Aggregate response from specialized agents
        """
        # Default implementation delegates to all agents and combines results
        # Specific manager implementations can override this for smarter routing
        results = []
        
        for agent in self.agents:
            try:
                result = agent.process(query, context)
                results.append(result)
            except Exception as e:
                # Log error but continue with other agents
                print(f"Error with agent {agent.name}: {str(e)}")
        
        # Find highest confidence result
        if results:
            best_result = max(results, key=lambda x: x.get("confidence", 0))
            return self.format_response(
                best_result["content"],
                best_result["confidence"],
                {"contributing_agents": [r["agent"] for r in results]}
            )
        else:
            return self.format_response(
                "I'm sorry, I couldn't find a specific answer to your question.",
                0.0,
                {"contributing_agents": []}
            )


class LLMAgent(BaseAgent):
    """Base agent class for LLM-powered agents"""
    
    def __init__(self, name, llm_service):
        """
        Initialize LLM agent with name and LLM service
        
        Args:
            name (str): Name of the agent
            llm_service: Service for LLM generation
        """
        super().__init__(name)
        self.llm_service = llm_service
    
    def generate_response(self, prompt):
        """
        Generate a response using the LLM service
        
        Args:
            prompt (str): The prompt to send to the LLM
            
        Returns:
            str: The generated response
        """
        # The actual LLM call will be handled by the LLM service
        # For MVP, we'll just use a simple completion
        return self.llm_service.generate(prompt)
    
    def extract_entities(self, text):
        """
        Extract relevant entities from text
        
        Args:
            text (str): The text to analyze
            
        Returns:
            dict: Extracted entities
        """
        # This will be implemented with more sophisticated NLP in the future
        # For MVP, we'll use simple keyword matching in subclasses
        return {}
    
    def retrieve_context(self, query, entities=None):
        """
        Retrieve relevant context for a query
        
        Args:
            query (str): The user query
            entities (dict, optional): Previously extracted entities
            
        Returns:
            dict: Retrieved context
        """
        # This will be implemented with RAG in the future
        # For MVP, we rely on LLM knowledge
        return {} 