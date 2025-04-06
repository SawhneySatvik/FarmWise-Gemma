from app.services.llm_service import LLMService
from app.models.user import User
from app.models.knowledge_base import KnowledgeItem
from flask import current_app
import json
from typing import Dict, Any, List, Optional

class FarmingAdvisor:
    """Top-level coordinator agent that manages specialized agents"""
    
    def __init__(self, llm_service: Optional[LLMService] = None):
        """
        Initialize the farming advisor
        
        Args:
            llm_service (LLMService, optional): LLM service to use for text generation
        """
        self.llm_service = llm_service or LLMService()
        
        # Agent mapping for routing
        self.agent_mapping = {
            "crop_selection": self._handle_crop_query,
            "soil_management": self._handle_soil_query,
            "weather_info": self._handle_weather_query,
            "pest_control": self._handle_pest_query,
            "market_price": self._handle_market_query,
            "irrigation": self._handle_irrigation_query,
            "general": self._handle_general_query
        }
    
    def process(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a user query and generate a response
        
        Args:
            query (str): The user query
            context (dict, optional): Additional context information
            
        Returns:
            dict: Response with content and metadata
        """
        if not context:
            context = {}
            
        # Extract parameters from the query
        parameters = self.llm_service.extract_parameters(query)
        
        # Determine which agent to use
        agent_type = parameters.get("query_intent", "general")
        
        # Get relevant knowledge items for context enhancement
        knowledge_items = self._get_relevant_knowledge(query, parameters)
        knowledge_context = self._format_knowledge_context(knowledge_items)
        
        # Create complete context
        full_context = {
            "parameters": parameters,
            "knowledge": knowledge_context,
            "user_context": context.get("user_context", {})
        }
        
        # Route to appropriate handler
        handler = self.agent_mapping.get(agent_type, self._handle_general_query)
        response = handler(query, full_context)
        
        # Format the final response
        return {
            "content": response,
            "metadata": {
                "agent_used": agent_type,
                "parameters": parameters
            },
            "confidence": 0.8  # Hardcoded for now, could be derived from LLM later
        }
    
    def _get_relevant_knowledge(self, query: str, parameters: Dict[str, Any]) -> List[KnowledgeItem]:
        """
        Get relevant knowledge items for the query
        
        Args:
            query (str): The user query
            parameters (dict): Extracted parameters
            
        Returns:
            list: Relevant knowledge items
        """
        try:
            # Get topic from parameters if available
            topic = None
            if parameters.get("query_intent") == "crop_selection" and parameters.get("crop_type"):
                topic = "Crops"
            elif parameters.get("query_intent") == "soil_management":
                topic = "Soil"
            elif parameters.get("query_intent") == "weather_info":
                topic = "Weather"
            elif parameters.get("query_intent") == "pest_control":
                topic = "Pests"
            elif parameters.get("query_intent") == "market_price":
                topic = "Market"
            elif parameters.get("query_intent") == "irrigation":
                topic = "Irrigation"
            
            # Get knowledge items
            if topic:
                # Try to get items by topic and subtopic
                items = KnowledgeItem.get_by_topic(topic, parameters.get("crop_type"))
                if items:
                    return items
                
                # If no items found, get all items for the topic
                items = KnowledgeItem.get_by_topic(topic)
                if items:
                    return items[:3]  # Limit to top 3 items
            
            # If no items found by topic, search by keywords
            return KnowledgeItem.search(query, limit=3)
            
        except Exception as e:
            current_app.logger.error(f"Error getting knowledge items: {str(e)}")
            return []
    
    def _format_knowledge_context(self, knowledge_items: List[KnowledgeItem]) -> str:
        """
        Format knowledge items into a context string for the LLM
        
        Args:
            knowledge_items (list): Knowledge items
            
        Returns:
            str: Formatted context string
        """
        if not knowledge_items:
            return ""
        
        context = "KNOWLEDGE BASE INFORMATION:\n\n"
        for item in knowledge_items:
            context += f"TOPIC: {item.topic}\n"
            context += f"SUBTOPIC: {item.subtopic}\n"
            context += f"CONTENT: {item.content}\n"
            context += f"SOURCE: {item.source}\n\n"
        
        return context
    
    def _create_prompt(self, query: str, context: Dict[str, Any], agent_type: str) -> str:
        """
        Create a prompt for the LLM based on the query and context
        
        Args:
            query (str): User query
            context (dict): Context information
            agent_type (str): Type of agent
            
        Returns:
            str: Formatted prompt
        """
        user_context = context.get("user_context", {})
        knowledge = context.get("knowledge", "")
        parameters = context.get("parameters", {})
        
        # Build basic prompt
        prompt = f"""You are FarmWise, an expert farming assistant for Indian farmers. 
Respond to the following query with practical, region-specific agricultural advice.

FARMER QUERY: {query}

"""
        
        # Add user context if available
        if user_context:
            prompt += "FARMER CONTEXT:\n"
            for key, value in user_context.items():
                if value:
                    prompt += f"- {key}: {value}\n"
            prompt += "\n"
        
        # Add extracted parameters if available
        if parameters:
            prompt += "QUERY PARAMETERS:\n"
            for key, value in parameters.items():
                if value:
                    prompt += f"- {key}: {value}\n"
            prompt += "\n"
        
        # Add knowledge context if available
        if knowledge:
            prompt += f"{knowledge}\n"
        
        # Add agent-specific instructions
        if agent_type == "crop_selection":
            prompt += """Focus on recommending appropriate crops based on the farmer's location, soil type, and current season. 
Consider climate conditions, water availability, and market prospects. Provide specific variety names when possible."""
        elif agent_type == "soil_management":
            prompt += """Provide specific advice on soil health management including fertilizer application, 
soil testing, organic matter incorporation, and addressing soil problems like salinity or acidity."""
        elif agent_type == "weather_info":
            prompt += """Interpret weather information in terms of agricultural impact. 
Advise on how to prepare for upcoming weather conditions and mitigate risks."""
        elif agent_type == "pest_control":
            prompt += """Suggest specific pest and disease management strategies, preferring IPM approaches when possible. 
Include both preventive measures and treatment options with specific product names and application rates where appropriate."""
        elif agent_type == "market_price":
            prompt += """Provide market intelligence with specific price ranges and trends. 
Suggest optimal timing for selling and potential markets that might offer better prices."""
        elif agent_type == "irrigation":
            prompt += """Give precise irrigation advice based on crop water requirements, 
soil type, growth stage, and weather conditions. Include irrigation scheduling and water conservation techniques."""
        
        # Add general guidelines
        prompt += """
RESPONSE GUIDELINES:
1. Be practical and specific - include quantities, timings, and methods
2. Use simple, conversational language
3. Focus on actionable advice
4. Keep your response brief but informative
5. When uncertain, acknowledge limitations
6. Include traditional knowledge when relevant
7. Mention the source of information when possible

RESPONSE:"""
        
        return prompt
    
    def _handle_crop_query(self, query: str, context: Dict[str, Any]) -> str:
        """Handle crop-related queries"""
        prompt = self._create_prompt(query, context, "crop_selection")
        return self.llm_service.generate(prompt)
    
    def _handle_soil_query(self, query: str, context: Dict[str, Any]) -> str:
        """Handle soil-related queries"""
        prompt = self._create_prompt(query, context, "soil_management")
        return self.llm_service.generate(prompt)
    
    def _handle_weather_query(self, query: str, context: Dict[str, Any]) -> str:
        """Handle weather-related queries"""
        prompt = self._create_prompt(query, context, "weather_info")
        return self.llm_service.generate(prompt)
    
    def _handle_pest_query(self, query: str, context: Dict[str, Any]) -> str:
        """Handle pest and disease-related queries"""
        prompt = self._create_prompt(query, context, "pest_control")
        return self.llm_service.generate(prompt)
    
    def _handle_market_query(self, query: str, context: Dict[str, Any]) -> str:
        """Handle market and price-related queries"""
        prompt = self._create_prompt(query, context, "market_price")
        return self.llm_service.generate(prompt)
    
    def _handle_irrigation_query(self, query: str, context: Dict[str, Any]) -> str:
        """Handle irrigation-related queries"""
        prompt = self._create_prompt(query, context, "irrigation")
        return self.llm_service.generate(prompt)
    
    def _handle_general_query(self, query: str, context: Dict[str, Any]) -> str:
        """Handle general farming queries"""
        prompt = self._create_prompt(query, context, "general")
        return self.llm_service.generate(prompt)
