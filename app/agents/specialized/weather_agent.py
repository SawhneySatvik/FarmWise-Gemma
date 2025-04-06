from app.agents.base_agent import BaseAgent
from app.services.weather_service import WeatherService
import logging

logger = logging.getLogger(__name__)

class WeatherAgent(BaseAgent):
    """Agent for handling weather-related queries"""
    
    def __init__(self):
        """Initialize the weather agent"""
        super().__init__()
        self.weather_service = WeatherService()
        self.name = "weather_agent"
    
    def process(self, query, context=None):
        """
        Process a weather-related query
        
        Args:
            query (str): User query about weather
            context (dict, optional): Additional context for processing
            
        Returns:
            dict: Response data
        """
        try:
            # Extract parameters from query
            parameters = self._extract_parameters(query, context)
            
            # Determine the weather request type
            request_type = self._determine_request_type(query)
            
            # Get the location from parameters or context
            location = parameters.get("location")
            if not location and context and "user_context" in context:
                user_location = context.get("user_context", {}).get("location")
                if user_location:
                    location = user_location
            
            # Default to the user's district or a major city if no location specified
            if not location and context and "user_context" in context:
                district = context.get("user_context", {}).get("district")
                state = context.get("user_context", {}).get("state")
                if district and state:
                    location = f"{district}, {state}"
                elif state:
                    location = state
            
            # If still no location, use a default
            if not location:
                location = "New Delhi"  # Default to capital
            
            # Process the request based on type
            if request_type == "current":
                return self._get_current_weather(location, parameters)
            elif request_type == "forecast":
                return self._get_forecast(location, parameters)
            else:
                return self._get_weather_advice(location, parameters)
                
        except Exception as e:
            logger.error(f"Error in WeatherAgent: {str(e)}")
            return self._generate_error_response(str(e))
    
    def _determine_request_type(self, query):
        """Determine what type of weather information the user is asking for"""
        query_lower = query.lower()
        
        # Check for forecast keywords
        forecast_keywords = ["forecast", "next week", "next few days", "tomorrow", "upcoming", "will it", "expecting"]
        if any(keyword in query_lower for keyword in forecast_keywords):
            return "forecast"
            
        # Check for current weather keywords
        current_keywords = ["current", "today", "now", "right now", "outside", "currently"]
        if any(keyword in query_lower for keyword in current_keywords):
            return "current"
            
        # Default to weather advice for farming
        return "advice"
    
    def _extract_parameters(self, query, context=None):
        """Extract parameters from the query"""
        # Start with basic parameters
        parameters = {
            "location": None,
            "time_frame": "current",  # current, today, tomorrow, week
            "weather_aspect": "general",  # general, rain, temperature, wind, humidity
        }
        
        # Extract location (basic implementation - can be enhanced with NER)
        query_words = query.lower().split()
        
        # Check for location after prepositions
        location_prepositions = ["in", "at", "near", "around", "for"]
        for i, word in enumerate(query_words):
            if word in location_prepositions and i < len(query_words) - 1:
                # Take the next 1-2 words as potential location
                potential_location = " ".join(query_words[i+1:i+3])
                # TODO: Validate if this is a real location
                parameters["location"] = potential_location
                break
        
        # Check for time frame
        if "tomorrow" in query.lower():
            parameters["time_frame"] = "tomorrow"
        elif any(phrase in query.lower() for phrase in ["next week", "coming week", "week ahead"]):
            parameters["time_frame"] = "week"
        elif any(phrase in query.lower() for phrase in ["next few days", "coming days", "days ahead"]):
            parameters["time_frame"] = "few_days"
            
        # Check for specific weather aspects
        if any(word in query.lower() for word in ["rain", "rainfall", "precipitation", "raining", "rainy"]):
            parameters["weather_aspect"] = "rain"
        elif any(word in query.lower() for word in ["temperature", "hot", "cold", "heat", "warm", "cool"]):
            parameters["weather_aspect"] = "temperature"
        elif any(word in query.lower() for word in ["wind", "windy", "breeze"]):
            parameters["weather_aspect"] = "wind"
        elif any(word in query.lower() for word in ["humid", "humidity", "moisture"]):
            parameters["weather_aspect"] = "humidity"
            
        return parameters
    
    def _get_current_weather(self, location, parameters):
        """Get current weather information"""
        # Get weather data
        weather_data = self.weather_service.get_weather_data(location)
        
        if "error" in weather_data:
            return self._generate_error_response(weather_data["error"])
        
        # Format for farmers
        formatted_data = self.weather_service.format_weather_for_farmers(weather_data)
        
        # Generate response based on the requested weather aspect
        weather_aspect = parameters.get("weather_aspect", "general")
        
        if weather_aspect == "rain":
            return self._generate_rain_response(formatted_data, "current")
        elif weather_aspect == "temperature":
            return self._generate_temperature_response(formatted_data, "current")
        elif weather_aspect == "wind":
            return self._generate_wind_response(formatted_data, "current")
        elif weather_aspect == "humidity":
            return self._generate_humidity_response(formatted_data, "current")
        else:
            return self._generate_general_response(formatted_data, "current")
    
    def _get_forecast(self, location, parameters):
        """Get weather forecast information"""
        # Get weather data
        weather_data = self.weather_service.get_weather_data(location)
        
        if "error" in weather_data:
            return self._generate_error_response(weather_data["error"])
        
        # Format for farmers
        formatted_data = self.weather_service.format_weather_for_farmers(weather_data)
        
        # Get time frame
        time_frame = parameters.get("time_frame", "few_days")
        
        # Generate response based on the requested weather aspect
        weather_aspect = parameters.get("weather_aspect", "general")
        
        if weather_aspect == "rain":
            return self._generate_rain_response(formatted_data, time_frame)
        elif weather_aspect == "temperature":
            return self._generate_temperature_response(formatted_data, time_frame)
        elif weather_aspect == "wind":
            return self._generate_wind_response(formatted_data, time_frame)
        elif weather_aspect == "humidity":
            return self._generate_humidity_response(formatted_data, time_frame)
        else:
            return self._generate_general_response(formatted_data, time_frame)
    
    def _get_weather_advice(self, location, parameters):
        """Get weather-based agricultural advice"""
        # Get weather data
        weather_data = self.weather_service.get_weather_data(location)
        
        if "error" in weather_data:
            return self._generate_error_response(weather_data["error"])
        
        # Format for farmers
        formatted_data = self.weather_service.format_weather_for_farmers(weather_data)
        
        # Generate comprehensive advice
        return {
            "content": self._generate_advice_text(formatted_data),
            "confidence": 0.85,
            "metadata": {
                "agent_used": self.name,
                "location": location,
                "weather_data_summary": {
                    "current_temp": formatted_data.get("current", {}).get("temperature", {}).get("current"),
                    "forecast_days": len(formatted_data.get("forecast", [])),
                    "rain_expected": any(day.get("rainfall_probability", 0) > 50 for day in formatted_data.get("forecast", []))
                }
            }
        }
    
    def _generate_general_response(self, weather_data, time_frame):
        """Generate a general weather response"""
        current = weather_data.get("current", {})
        forecast = weather_data.get("forecast", [])
        location = weather_data.get("location", "your location")
        
        if time_frame == "current":
            # Current weather response
            temp = current.get("temperature", {}).get("current")
            condition = current.get("condition", "clear")
            humidity = current.get("humidity")
            
            response = f"Current weather in {location.split(',')[0]}: "
            if temp is not None:
                response += f"{temp}°C, "
            response += f"{condition.lower()}"
            if humidity is not None:
                response += f", humidity {humidity}%"
            
            # Add agricultural relevance
            advice = weather_data.get("agricultural_advice", [])
            if advice:
                response += f"\n\nFarming implications: {advice[0]}"
        
        elif time_frame == "tomorrow" and len(forecast) >= 1:
            # Tomorrow's forecast
            tomorrow = forecast[0]
            temp_max = tomorrow.get("temperature", {}).get("max")
            temp_min = tomorrow.get("temperature", {}).get("min")
            condition = tomorrow.get("condition")
            rain_prob = tomorrow.get("rainfall_probability")
            
            response = f"Tomorrow's forecast for {location.split(',')[0]}: "
            if temp_min is not None and temp_max is not None:
                response += f"{temp_min}-{temp_max}°C, "
            response += f"{condition.lower()}"
            if rain_prob is not None and rain_prob > 0:
                response += f", {rain_prob}% chance of rain"
            
            # Add agricultural relevance
            advice = weather_data.get("agricultural_advice", [])
            if advice and len(advice) > 1:
                response += f"\n\nFarming implications: {advice[1]}"
        
        else:
            # Multi-day forecast
            days_to_include = min(len(forecast), 5)  # Limit to 5 days
            
            response = f"Weather forecast for {location.split(',')[0]}:\n"
            
            for i in range(days_to_include):
                day = forecast[i]
                date = day.get("date", f"Day {i+1}")
                temp_max = day.get("temperature", {}).get("max")
                temp_min = day.get("temperature", {}).get("min")
                condition = day.get("condition")
                rain_prob = day.get("rainfall_probability")
                
                response += f"\n{date}: "
                if temp_min is not None and temp_max is not None:
                    response += f"{temp_min}-{temp_max}°C, "
                response += f"{condition.lower()}"
                if rain_prob is not None and rain_prob > 0:
                    response += f", {rain_prob}% chance of rain"
            
            # Add agricultural relevance
            advice = weather_data.get("agricultural_advice", [])
            if advice:
                response += f"\n\nFarming implications: {'; '.join(advice[:2])}"
        
        return {
            "content": response,
            "confidence": 0.9,
            "metadata": {
                "agent_used": self.name,
                "location": location,
                "time_frame": time_frame
            }
        }
    
    def _generate_rain_response(self, weather_data, time_frame):
        """Generate a rain-specific weather response"""
        forecast = weather_data.get("forecast", [])
        location = weather_data.get("location", "your location")
        
        if time_frame == "current":
            current = weather_data.get("current", {})
            condition = current.get("condition", "").lower()
            rainfall = current.get("rainfall", 0)
            
            if "rain" in condition or rainfall > 0:
                response = f"It is currently raining in {location.split(',')[0]}"
                if rainfall > 0:
                    response += f" with {rainfall}mm of rainfall in the last hour"
            else:
                response = f"It is not currently raining in {location.split(',')[0]}"
            
            # Add rainfall forecast for the next 24 hours
            if forecast and len(forecast) > 0:
                tomorrow = forecast[0]
                rain_prob = tomorrow.get("rainfall_probability", 0)
                total_rainfall = tomorrow.get("total_rainfall", 0)
                
                if rain_prob > 70:
                    response += f". High probability ({rain_prob}%) of rain tomorrow"
                    if total_rainfall > 0:
                        response += f" with approximately {total_rainfall}mm expected"
                elif rain_prob > 30:
                    response += f". Moderate chance ({rain_prob}%) of rain tomorrow"
                else:
                    response += ". No significant rainfall expected tomorrow"
        
        else:
            # Forecast for rain
            days_with_rain = [day for day in forecast if day.get("rainfall_probability", 0) > 30]
            
            if not days_with_rain:
                response = f"No significant rainfall is expected in {location.split(',')[0]} for the next {len(forecast)} days."
            else:
                response = f"Rainfall forecast for {location.split(',')[0]}:\n"
                
                for day in days_with_rain:
                    date = day.get("date", "")
                    rain_prob = day.get("rainfall_probability", 0)
                    total_rainfall = day.get("total_rainfall", 0)
                    
                    response += f"\n{date}: {rain_prob}% chance of rain"
                    if total_rainfall > 0:
                        response += f", approximately {total_rainfall}mm expected"
            
        # Add agricultural relevance
        advice = [adv for adv in weather_data.get("agricultural_advice", []) if "rain" in adv.lower()]
        if advice:
            response += f"\n\nFarming implications: {advice[0]}"
        
        return {
            "content": response,
            "confidence": 0.9,
            "metadata": {
                "agent_used": self.name,
                "location": location,
                "weather_aspect": "rain",
                "time_frame": time_frame
            }
        }
    
    def _generate_temperature_response(self, weather_data, time_frame):
        """Generate a temperature-specific weather response"""
        current = weather_data.get("current", {})
        forecast = weather_data.get("forecast", [])
        location = weather_data.get("location", "your location")
        
        if time_frame == "current":
            # Current temperature
            temp = current.get("temperature", {})
            current_temp = temp.get("current")
            feels_like = temp.get("feels_like")
            
            response = f"Current temperature in {location.split(',')[0]}"
            if current_temp is not None:
                response += f" is {current_temp}°C"
                if feels_like is not None and abs(current_temp - feels_like) > 1:
                    response += f" (feels like {feels_like}°C)"
            else:
                response += " is unavailable"
            
            # Add high/low
            temp_min = temp.get("min")
            temp_max = temp.get("max")
            if temp_min is not None and temp_max is not None:
                response += f". Today's range: {temp_min}-{temp_max}°C"
        
        else:
            # Temperature forecast
            days_to_include = min(len(forecast), 5 if time_frame == "week" else 3)
            
            response = f"Temperature forecast for {location.split(',')[0]}:\n"
            
            for i in range(days_to_include):
                day = forecast[i]
                date = day.get("date", f"Day {i+1}")
                temp = day.get("temperature", {})
                temp_min = temp.get("min")
                temp_max = temp.get("max")
                
                response += f"\n{date}: "
                if temp_min is not None and temp_max is not None:
                    response += f"{temp_min}-{temp_max}°C"
                else:
                    response += "Temperature data unavailable"
        
        # Add agricultural relevance
        advice = [adv for adv in weather_data.get("agricultural_advice", []) if "temperature" in adv.lower()]
        if advice:
            response += f"\n\nFarming implications: {advice[0]}"
        else:
            # Check for high/low temperature advice
            advice = [adv for adv in weather_data.get("agricultural_advice", []) if any(word in adv.lower() for word in ["hot", "cold", "heat", "frost"])]
            if advice:
                response += f"\n\nFarming implications: {advice[0]}"
        
        return {
            "content": response,
            "confidence": 0.9,
            "metadata": {
                "agent_used": self.name,
                "location": location,
                "weather_aspect": "temperature",
                "time_frame": time_frame
            }
        }
    
    def _generate_wind_response(self, weather_data, time_frame):
        """Generate a wind-specific weather response"""
        current = weather_data.get("current", {})
        location = weather_data.get("location", "your location")
        
        wind = current.get("wind", {})
        speed = wind.get("speed")
        direction = wind.get("direction")
        
        response = f"Wind conditions in {location.split(',')[0]}: "
        if speed is not None:
            # Convert m/s to km/h
            speed_kmh = speed * 3.6
            response += f"{speed_kmh:.1f} km/h"
            
            # Describe wind strength
            if speed_kmh < 5:
                response += " (calm)"
            elif speed_kmh < 20:
                response += " (light breeze)"
            elif speed_kmh < 40:
                response += " (moderate wind)"
            elif speed_kmh < 60:
                response += " (strong wind)"
            else:
                response += " (very strong wind)"
                
            # Add direction if available
            if direction is not None:
                wind_direction = self._get_wind_direction(direction)
                response += f" from the {wind_direction}"
        else:
            response += "Wind data unavailable"
        
        # Add agricultural relevance
        advice = [adv for adv in weather_data.get("agricultural_advice", []) if "wind" in adv.lower()]
        if advice:
            response += f"\n\nFarming implications: {advice[0]}"
        
        return {
            "content": response,
            "confidence": 0.9,
            "metadata": {
                "agent_used": self.name,
                "location": location,
                "weather_aspect": "wind",
                "time_frame": time_frame
            }
        }
    
    def _generate_humidity_response(self, weather_data, time_frame):
        """Generate a humidity-specific weather response"""
        current = weather_data.get("current", {})
        location = weather_data.get("location", "your location")
        
        humidity = current.get("humidity")
        temp = current.get("temperature", {}).get("current")
        
        response = f"Humidity conditions in {location.split(',')[0]}: "
        if humidity is not None:
            response += f"{humidity}%"
            
            # Describe humidity level
            if humidity < 30:
                response += " (very dry)"
            elif humidity < 50:
                response += " (dry)"
            elif humidity < 70:
                response += " (moderate)"
            elif humidity < 90:
                response += " (humid)"
            else:
                response += " (very humid)"
            
            # Add temperature context if available
            if temp is not None:
                if temp > 30 and humidity > 70:
                    response += ". The combination of high temperature and humidity may cause heat stress for plants and workers"
                elif temp < 10 and humidity > 90:
                    response += ". High humidity with low temperature increases the risk of fungal diseases"
        else:
            response += "Humidity data unavailable"
        
        return {
            "content": response,
            "confidence": 0.9,
            "metadata": {
                "agent_used": self.name,
                "location": location,
                "weather_aspect": "humidity",
                "time_frame": time_frame
            }
        }
    
    def _generate_advice_text(self, weather_data):
        """Generate comprehensive agricultural advice based on weather data"""
        current = weather_data.get("current", {})
        forecast = weather_data.get("forecast", [])
        advice = weather_data.get("agricultural_advice", [])
        location = weather_data.get("location", "your location").split(',')[0]
        
        # Start with current conditions summary
        temp = current.get("temperature", {}).get("current")
        condition = current.get("condition", "")
        humidity = current.get("humidity")
        
        response = f"Weather analysis for {location}:\n\n"
        response += "Current conditions: "
        if temp is not None:
            response += f"{temp}°C, "
        response += f"{condition.lower()}"
        if humidity is not None:
            response += f", {humidity}% humidity"
        
        # Add forecast summary
        if forecast:
            response += "\n\nForecast summary:\n"
            
            # Check for rain in the forecast
            rain_days = [day for day in forecast[:5] if day.get("rainfall_probability", 0) > 50]
            if rain_days:
                rain_dates = ", ".join([day.get("date", "") for day in rain_days])
                response += f"- Rain expected on: {rain_dates}\n"
            else:
                response += "- No significant rainfall expected in the next few days\n"
            
            # Temperature trend
            if len(forecast) >= 3:
                avg_max = sum(day.get("temperature", {}).get("max", 0) for day in forecast[:3]) / 3
                avg_min = sum(day.get("temperature", {}).get("min", 0) for day in forecast[:3]) / 3
                response += f"- Average temperature in the next 3 days: {avg_min:.1f}-{avg_max:.1f}°C\n"
        
        # Add comprehensive agricultural advice
        response += "\nFarming recommendations based on this weather:\n"
        
        if advice:
            for i, item in enumerate(advice, 1):
                response += f"{i}. {item}\n"
        else:
            # Fallback advice if none was generated
            current_temp = current.get("temperature", {}).get("current")
            if current_temp is not None:
                if current_temp > 35:
                    response += "1. High temperatures: Increase irrigation frequency and monitor for heat stress\n"
                    response += "2. Consider providing shade for sensitive crops\n"
                elif current_temp < 10:
                    response += "1. Low temperatures: Monitor for frost damage, especially for sensitive crops\n"
                    response += "2. Delay sowing of warm-season crops until temperatures rise\n"
            
            # Check forecast for rain
            if forecast and any(day.get("rainfall_probability", 0) > 70 for day in forecast[:3]):
                response += "3. Prepare for rainfall: Ensure proper drainage in your fields\n"
                response += "4. Consider delaying fertilizer application until after the rain\n"
            else:
                response += "3. Dry conditions expected: Ensure adequate irrigation for your crops\n"
                response += "4. Good opportunity for fertilizer application with proper irrigation\n"
        
        return response
    
    def _get_wind_direction(self, degrees):
        """Convert wind direction in degrees to cardinal direction"""
        directions = ["north", "northeast", "east", "southeast",
                     "south", "southwest", "west", "northwest"]
        index = round(degrees / 45) % 8
        return directions[index]
    
    def _generate_error_response(self, error_message):
        """Generate an error response"""
        return {
            "content": f"I'm sorry, I couldn't retrieve the weather information at this time. {error_message}",
            "confidence": 0.5,
            "metadata": {
                "agent_used": self.name,
                "error": error_message
            }
        }
