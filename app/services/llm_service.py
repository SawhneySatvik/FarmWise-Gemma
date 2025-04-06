import os
import json
import requests # For Ollama API calls
from flask import current_app
from app.config import Config # Assumes Config holds OLLAMA settings
from typing import Dict, Any
import logging

# --- Configuration Assumption ---
# Assumes app.config.Config has:
# OLLAMA_BASE_URL: str (e.g., 'http://localhost:11434')
# OLLAMA_MODEL: str (e.g., 'gemma3:latest' or 'gemma3:8b')
# OLLAMA_FALLBACK_MODEL: str (Optional, e.g., 'gemma2:9b' or 'llama3:latest')
# OLLAMA_REQUEST_TIMEOUT: int (Optional, e.g., 60)

class LLMService:
    """Service for interacting with local LLMs via Ollama (targeting Gemma models)."""

    def __init__(self):
        """Initialize the LLM service using Ollama configuration."""
        try:
            self.logger = current_app.logger
        except RuntimeError: # Handle running outside Flask app context
            self.logger = logging.getLogger("llm_service_standalone")
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)

        self.ollama_base_url = getattr(Config, 'OLLAMA_BASE_URL', 'http://localhost:11434')
        # Defaulting to a potential Gemma 3 model name
        self.model_name = getattr(Config, 'OLLAMA_MODEL', 'gemma3:latest')
        self.fallback_model_name = getattr(Config, 'OLLAMA_FALLBACK_MODEL', 'gemma2:9b') # Example fallback
        self.request_timeout = getattr(Config, 'OLLAMA_REQUEST_TIMEOUT', 60)

        if not self.ollama_base_url:
            self.logger.warning("No Ollama base URL found (OLLAMA_BASE_URL), using mock responses")
            self.use_mock = True
        else:
            self.use_mock = False
            self.logger.info(f"Initialized Ollama LLM service. Primary: {self.model_name}, Fallback: {self.fallback_model_name}, URL: {self.ollama_base_url}")

    def _ollama_request(self, model: str, prompt: str, generation_config: Dict[str, Any], stream: bool = False) -> Dict[str, Any]:
        """Helper function to make requests to the Ollama API."""
        api_url = f"{self.ollama_base_url}/api/generate"
        headers = {'Content-Type': 'application/json'}

        # Map generation_config to Ollama options (check Ollama docs for specifics)
        options = {
            "temperature": generation_config.get("temperature", 0.7),
            "top_p": generation_config.get("top_p", 0.9),
            "top_k": generation_config.get("top_k", 40),
            # 'num_predict': generation_config.get("max_output_tokens") # Optional mapping
        }

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": options
            # Consider adding "format": "json" for extraction if models reliably support it
        }

        try:
            response = requests.post(
                api_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=self.request_timeout
            )
            response.raise_for_status() # Raise HTTPError for bad responses
            return response.json()
        except requests.exceptions.Timeout:
            self.logger.error(f"Timeout error connecting to Ollama at {api_url} for model {model}")
            raise
        except requests.exceptions.ConnectionError:
            self.logger.error(f"Connection error connecting to Ollama at {api_url}. Is Ollama running?")
            raise
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Ollama request failed for model {model}: {e}")
            if e.response is not None:
                self.logger.error(f"Ollama Response Status: {e.response.status_code}, Body: {e.response.text[:500]}")
            raise
        except json.JSONDecodeError as e:
            # Log the raw response if JSON decoding fails
            raw_response = response.text if 'response' in locals() else 'Response object not available'
            self.logger.error(f"Failed to decode JSON response from Ollama (model: {model}): {e}. Raw response: {raw_response[:500]}")
            raise

    def generate(self, prompt: str) -> str:
        """Generate text using the configured Ollama LLM."""
        if self.use_mock:
            return self._generate_mock(prompt)

        generation_config = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            # "max_output_tokens": 1024, # Mapped to num_predict in _ollama_request if needed
        }

        try:
            # Try primary model
            self.logger.debug(f"Attempting generation with primary model: {self.model_name}")
            response_data = self._ollama_request(self.model_name, prompt, generation_config)
            return response_data.get("response", "").strip()

        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            self.logger.warning(f"Primary Ollama model {self.model_name} failed generation: {e}. Trying fallback.")

            # Try fallback model if configured
            if self.fallback_model_name:
                try:
                    self.logger.debug(f"Attempting generation with fallback model: {self.fallback_model_name}")
                    response_data = self._ollama_request(self.fallback_model_name, prompt, generation_config)
                    return response_data.get("response", "").strip()
                except (requests.exceptions.RequestException, json.JSONDecodeError) as fallback_error:
                    self.logger.error(f"Fallback Ollama model {self.fallback_model_name} also failed: {fallback_error}. Falling back to mock.")
                    return self._generate_mock(prompt)
            else:
                self.logger.error("Primary model failed and no fallback configured. Falling back to mock.")
                return self._generate_mock(prompt)

        except Exception as e:
            # Catch any other unexpected errors
            self.logger.error(f"Unexpected error during Ollama text generation: {str(e)}", exc_info=True)
            return self._generate_mock(prompt)

    def extract_parameters(self, query: str) -> Dict[str, Any]:
        """Extract farming parameters from a user query using Ollama."""
        if self.use_mock:
            return self._extract_parameters_mock(query)

        # Prompt designed to guide the LLM towards JSON output
        extraction_prompt = f"""
        Analyze the following user query about farming. Extract the specified parameters and return them ONLY as a valid JSON object.

        Parameters to extract:
        - query_intent: Classify the main goal (e.g., "crop_selection", "soil_management", "weather_info", "pest_control", "market_price", "irrigation", "livestock_info", "general_advice").
        - crop_type: Specific crop name mentioned (e.g., "wheat", "tomato"). Use null if not mentioned.
        - soil_type: Soil type mentioned (e.g., "sandy", "clay"). Use null if not mentioned.
        - location: Geographic location or region mentioned. Use null if not mentioned.
        - season: Time period like "kharif", "rabi", "summer". Use null if not mentioned.
        - pest_type: Specific pest or disease mentioned (e.g., "aphids", "rust"). Use null if not mentioned.
        - livestock_type: Specific livestock mentioned (e.g., "cow", "poultry"). Use null if not mentioned.
        - additional_context: Any other relevant details or parameters from the query. Use null if none.

        User Query: "{query}"

        JSON Response:
        ```json
        {{
          "query_intent": "...",
          "crop_type": null,
          "soil_type": null,
          "location": null,
          "season": null,
          "pest_type": null,
          "livestock_type": null,
          "additional_context": null
        }}
        ```
        """

        # Use low temperature for more deterministic JSON output
        generation_config = {
            "temperature": 0.1,
            "top_p": 0.95,
            "top_k": 40,
        }

        response_text = None
        try:
            # Try primary model
            self.logger.debug(f"Attempting parameter extraction with primary model: {self.model_name}")
            response_data = self._ollama_request(self.model_name, extraction_prompt, generation_config)
            response_text = response_data.get("response", "").strip()

        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            self.logger.warning(f"Primary Ollama model {self.model_name} failed parameter extraction: {e}. Trying fallback.")
            # Try fallback model if configured
            if self.fallback_model_name:
                try:
                    self.logger.debug(f"Attempting parameter extraction with fallback model: {self.fallback_model_name}")
                    response_data = self._ollama_request(self.fallback_model_name, extraction_prompt, generation_config)
                    response_text = response_data.get("response", "").strip()
                except (requests.exceptions.RequestException, json.JSONDecodeError) as fallback_error:
                    self.logger.error(f"Fallback Ollama model {self.fallback_model_name} also failed extraction: {fallback_error}. Falling back to simple extraction.")
                    return self._simple_parameter_extraction(query, f"Fallback model failed: {fallback_error}")
            else:
                self.logger.error("Primary model failed extraction, no fallback configured. Falling back to simple extraction.")
                return self._simple_parameter_extraction(query, f"Primary model failed: {e}")

        except Exception as e:
            # Catch any other unexpected errors
            self.logger.error(f"Unexpected error during Ollama parameter extraction: {str(e)}", exc_info=True)
            return self._simple_parameter_extraction(query, f"Unexpected error: {str(e)}")

        # If we received a response, attempt to parse JSON
        if response_text:
            try:
                # Prioritize finding JSON within ```json blocks
                if "```json" in response_text:
                    json_str = response_text.split("```json")[1].split("```")[0].strip()
                    return json.loads(json_str)
                # Check for generic ``` blocks containing JSON
                elif "```" in response_text and '{' in response_text and '}' in response_text:
                     potential_json = response_text.split("```")[1].split("```")[0].strip()
                     if potential_json.startswith('{') and potential_json.endswith('}'):
                        return json.loads(potential_json)
                # Fallback: try parsing the entire response, cleaning it first
                first_brace = response_text.find('{')
                last_brace = response_text.rfind('}')
                if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                     json_str = response_text[first_brace:last_brace+1]
                     return json.loads(json_str)

                # If no JSON structure found after attempts
                raise json.JSONDecodeError("No valid JSON object found in the response", response_text, 0)

            except (json.JSONDecodeError, IndexError) as parse_error:
                self.logger.warning(f"Failed to parse JSON from Ollama response: {parse_error}. Falling back to simple extraction.")
                self.logger.debug(f"Raw response causing parse error: {response_text}")
                return self._simple_parameter_extraction(query, response_text)
        else:
             # If response_text is still None after all attempts
             self.logger.error("Failed to get any response from Ollama for parameter extraction.")
             return self._simple_parameter_extraction(query, "No response received from LLM.")


    def _simple_parameter_extraction(self, query: str, model_response_info: str) -> Dict[str, Any]:
        """Fallback method for parameter extraction when JSON parsing fails."""
        self.logger.info(f"Using simple parameter extraction for query: {query}")
        params = {
            "query_intent": "general", "crop_type": None, "soil_type": None,
            "location": None, "season": None, "pest_type": None,
            "livestock_type": None,
            "additional_context": f"LLM response info (non-JSON or error): {model_response_info[:200]}"
        }

        intent_keywords = {
            "crop_selection": ["crop", "plant", "seed", "grow", "variety", "sow"],
            "soil_management": ["soil", "fertilizer", "manure", "compost", "nutrient", "ph"],
            "weather_info": ["weather", "rain", "temperature", "monsoon", "climate"],
            "pest_control": ["pest", "disease", "insect", "fungus", "infection", "control"],
            "market_price": ["price", "market", "sell", "cost", "profit", "mandi"],
            "irrigation": ["water", "irrigation", "moisture", "drip", "sprinkler"],
            "livestock_info": ["cow", "cattle", "buffalo", "goat", "poultry", "chicken", "pig"]
        }
        query_lower = query.lower()
        for intent, keywords in intent_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                params["query_intent"] = intent
                break

        common_crops = ["wheat", "rice", "maize", "corn", "cotton", "sugarcane", "tomato", "onion"]
        for crop in common_crops:
            if crop in query_lower: params["crop_type"] = crop; break
        common_pests = ["rust", "aphid", "borer", "blight", "mildew"]
        for pest in common_pests:
             if pest in query_lower: params["pest_type"] = pest; break
      

        return params

    def _generate_mock(self, prompt: str) -> str:
        """Generate mock responses for development and testing."""
        self.logger.info(f"Using mock response for prompt: {prompt[:50]}...")
        # (Keep the exact mock response logic from the original code)
        keywords = {
            "crop": ["crop", "plant", "seed", "harvest", "grow", "cultivate", "sow"],
            "soil": ["soil", "fertilizer", "nutrient", "compost", "manure"],
            "pest": ["pest", "disease", "insect", "fungus", "bacteria", "control"],
            "water": ["water", "irrigation", "moisture", "drought", "rain"],
            "market": ["market", "price", "sell", "cost", "profit", "mandi"],
            "weather": ["weather", "rain", "temperature", "climate", "monsoon"]
        }
        if "rust" in prompt.lower() and "wheat" in prompt.lower():
            # Return specific mock for wheat rust
            return """To prevent rust disease in wheat, follow these effective strategies:

1. Plant rust-resistant varieties: Use varieties like HD 3086, PBW 725, or WH 1105 that have genetic resistance to rust diseases.
2. Apply fungicides preventively: Spray propiconazole (0.1%) or tebuconazole (0.1%) at the first sign of disease. Early application is crucial for effective control.
3. Ensure proper spacing: Allow adequate space between plants for good air circulation which reduces humidity levels that favor rust development.
4. Practice balanced fertilization: Avoid excessive nitrogen which can make plants more susceptible to rust. Maintain balanced NPK ratios.
5. Rotate crops: Avoid planting wheat after wheat in the same field. This breaks the disease cycle.
6. Monitor fields regularly: Check your wheat crop at least weekly, focusing on lower leaves where rust often appears first. Early detection allows for timely intervention.
7. Remove volunteer wheat: Eliminate any volunteer wheat plants that could harbor rust between growing seasons.
8. Use proper timing: Plant early in the season to allow the crop to mature before rust becomes prevalent during warmer weather."""

        for theme, theme_keywords in keywords.items():
            if any(keyword in prompt.lower() for keyword in theme_keywords):
                if theme == "soil":
                    return """Soil health is fundamental to successful farming. Here are key practices for soil management:

1. Regular soil testing: Test your soil at least once every 2-3 years to understand its nutrient profile, pH level, and organic matter content.
2. Organic matter addition: Apply farmyard manure, compost, or green manures to improve soil structure and nutrient availability. Apply 5-10 tonnes per hectare before land preparation.
3. Crop rotation: Alternate crops with different nutrient needs to prevent soil depletion. Include legumes in your rotation to fix nitrogen naturally.
4. Minimum tillage: Reduce tillage operations to minimize soil disturbance and prevent erosion.
5. Cover crops: Plant cover crops during fallow periods to protect soil from erosion and add organic matter.
6. Balanced fertilization: Apply fertilizers based on soil test recommendations and crop requirements rather than general estimates.
7. Manage soil pH: Most crops prefer slightly acidic to neutral soil (pH 6.0-7.0). Apply lime to acidic soils or gypsum to alkaline soils as needed.
8. Prevent compaction: Avoid heavy machinery on wet soils and consider using controlled traffic farming techniques.

These practices will help maintain soil fertility, improve water infiltration and retention, and ultimately lead to better crop yields."""
                elif theme == "crop":
                     return """Effective crop management involves several key practices to maximize yield and quality:

1. Variety selection: Choose crop varieties suited to your specific climate, soil type, and market demands. Consider disease resistance, maturity period, and yield potential.
2. Seed treatment: Treat seeds with appropriate fungicides and insecticides to protect young seedlings from soil-borne diseases and pests.
3. Optimal planting: Follow recommended seeding rates, planting depth, and spacing. Consider row orientation for maximum sunlight interception.
4. Nutrient management: Apply fertilizers based on soil tests and crop requirements, using split applications for better utilization.
5. Water management: Ensure adequate water supply through proper irrigation scheduling based on crop growth stages and weather conditions.
6. Weed control: Implement integrated weed management combining cultural practices, mechanical methods, and appropriate herbicides.
7. Regular monitoring: Scout fields frequently to detect and address issues with pests, diseases, nutrient deficiencies, or water stress.
8. Timely harvest: Harvest at optimal maturity to maximize yield, quality, and market value.

Combining these practices in a systematic approach will help you achieve consistent crop performance and sustainable production."""
                elif theme == "pest":
                     return """Effective pest and disease management requires an integrated approach:

1. Prevention: Start with resistant varieties, clean planting material, and crop rotation to break pest cycles.
2. Monitoring: Regularly inspect crops for early signs of pests and diseases. Consider setting up pheromone traps for key insect pests.
3. Cultural controls: Adjust planting dates, use trap crops, practice field sanitation, and manage irrigation to create unfavorable conditions for pests.
4. Biological controls: Encourage beneficial insects by maintaining habitat diversity. Consider introducing predators, parasites, or microbial agents.
5. Chemical controls (when necessary):
   - Use selective pesticides that target specific pests
   - Apply at recommended rates and timing
   - Rotate pesticide groups to prevent resistance
   - Follow safety guidelines and pre-harvest intervals
6. Neem-based solutions: For organic management, neem oil (5ml/liter) with a drop of soap as emulsifier is effective against many soft-bodied insects.
7. Prompt intervention: Address pest issues when populations are still low and damage is minimal.
8. Post-harvest management: Properly dispose of crop residues that might harbor pests and diseases.

This integrated approach helps manage pests effectively while minimizing environmental impact and preserving beneficial organisms."""
                # ... (Add other theme mocks: water, market, weather) ...


        # Default generic farming advice
        return """Here are some general best practices for successful farming:

1. Soil health management: Regularly test your soil and maintain organic matter through crop rotations, cover crops, and organic inputs.
2. Water efficiency: Implement water conservation techniques and efficient irrigation methods appropriate for your crops and region.
3. Integrated pest management: Use a combination of cultural, biological, and targeted chemical controls to manage pests while minimizing environmental impact.
4. Crop selection and rotation: Choose crops suited to your local conditions and rotate them properly to break pest cycles and maintain soil health.
5. Farm record keeping: Maintain detailed records of inputs, activities, yields, and finances to inform future decisions.
6. Market awareness: Stay informed about market trends and explore multiple marketing channels to optimize returns.
7. Mechanization appropriate to scale: Select tools and equipment that improve efficiency without creating excessive costs or soil compaction.
8. Continuous learning: Stay updated on agricultural innovations and best practices through extension services, farmer networks, and agricultural education resources.
9. Risk management: Diversify crops, consider insurance options, and develop contingency plans for weather extremes and market fluctuations.
10. Sustainable practices: Balance productivity with environmental stewardship to ensure long-term farm viability.

If you have a more specific farming question, please provide details about your crops, location, and particular concerns for more targeted advice."""


    def _extract_parameters_mock(self, query: str) -> Dict[str, Any]:
        """Extract parameters using a rule-based approach for mock responses."""
        self.logger.info(f"Using mock parameter extraction for query: {query}")
        params = {
            "query_intent": "general", "crop_type": None, "soil_type": None,
            "location": None, "season": None, "pest_type": None,
            "livestock_type": None, "additional_context": None
        }
        # Simple intent detection based on keywords
        intent_keywords = {
            "crop_selection": ["crop", "plant", "seed", "grow", "variety", "sow"],
            "soil_management": ["soil", "fertilizer", "manure", "compost", "nutrient", "ph"],
            "weather_info": ["weather", "rain", "temperature", "monsoon", "climate"],
            "pest_control": ["pest", "disease", "insect", "fungus", "infection", "control"],
            "market_price": ["price", "market", "sell", "cost", "profit", "mandi"],
            "irrigation": ["water", "irrigation", "moisture", "drip", "sprinkler"],
            "livestock_info": ["cow", "cattle", "buffalo", "goat", "poultry", "chicken", "pig"]
        }
        query_lower = query.lower()
        for intent, keywords in intent_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                params["query_intent"] = intent
                break

        # Very basic keyword spotting for other params
        common_crops = ["wheat", "rice", "maize", "corn", "cotton", "sugarcane", "tomato", "onion"]
        for crop in common_crops:
            if crop in query_lower: params["crop_type"] = crop; break
        common_pests = ["rust", "aphid", "borer", "blight", "mildew"]
        for pest in common_pests:
             if pest in query_lower: params["pest_type"] = pest; break
        soil_types = ["sandy", "clay", "loam", "silt", "black", "red", "alluvial", "laterite"]
        for soil in soil_types:
            if soil in query_lower: params["soil_type"] = soil; break
        livestock_types = ["cow", "cattle", "buffalo", "goat", "sheep", "poultry", "chicken", "pig"]
        for livestock in livestock_types:
            if livestock in query_lower: params["livestock_type"] = livestock; break
        states = ["punjab", "haryana", "uttar pradesh", "bihar", "west bengal", "assam",
                 "gujarat", "maharashtra", "karnataka", "tamil nadu", "andhra pradesh",
                 "telangana", "madhya pradesh", "rajasthan"]
        for state in states:
            if state in query_lower: params["location"] = state; break
        seasons = ["kharif", "rabi", "summer", "winter", "monsoon", "spring"]
        for season in seasons:
            if season in query_lower: params["season"] = season; break

        return params
