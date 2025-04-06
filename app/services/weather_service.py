import os
import requests
import logging
from flask import current_app
from app.config import Config
from app.services import location_service

logger = logging.getLogger(__name__)

class WeatherService:
    """Service for fetching weather data from OpenWeatherMap API"""
    
    def __init__(self):
        """Initialize the weather service with API key from config"""
        self.api_key = Config.OPEN_WEATHER_API_KEY
        self.current_weather_url = Config.CURRENT_WEATHER_DATA_URL
        self.forecast_url = Config.FIVE_DAY_FORECAST_DATA_URL
        
        if not self.api_key:
            try:
                current_app.logger.warning("No OpenWeatherMap API key found, weather services will be unavailable")
            except RuntimeError:
                logger.warning("No OpenWeatherMap API key found, weather services will be unavailable")
    
    def get_current_weather(self, lat, lon):
        """
        Fetch current weather data for a given latitude and longitude.
        
        Args:
            lat (float): Latitude
            lon (float): Longitude
            
        Returns:
            dict: Weather data or error message
        """
        if not self.api_key or not self.current_weather_url:
            return {"error": "Weather API key or URL is missing."}

        url = self.current_weather_url.format(lat=lat, lon=lon, API_key=self.api_key)
        
        try:
            response = requests.get(url)
            
            if response.status_code == 200:
                return response.json()
            else:
                error_msg = f"Failed to fetch current weather data. Status Code: {response.status_code}"
                logger.error(error_msg)
                return {"error": error_msg}
        except Exception as e:
            error_msg = f"Error fetching current weather: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}

    def get_forecast(self, lat, lon):
        """
        Fetch 5-day weather forecast data for a given latitude and longitude.
        
        Args:
            lat (float): Latitude
            lon (float): Longitude
            
        Returns:
            dict: Forecast data or error message
        """
        if not self.api_key or not self.forecast_url:
            return {"error": "Weather API key or URL is missing."}

        url = self.forecast_url.format(lat=lat, lon=lon, API_key=self.api_key)
        
        try:
            response = requests.get(url)

            if response.status_code == 200:
                return response.json() 
            else:
                error_msg = f"Failed to fetch forecast data. Status Code: {response.status_code}"
                logger.error(error_msg)
                return {"error": error_msg}
        except Exception as e:
            error_msg = f"Error fetching weather forecast: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}

    def get_weather_data(self, city_name):
        """
        Fetch complete weather data for a given city.
        
        Args:
            city_name (str): Name of the city
            
        Returns:
            dict: Complete weather data including current conditions and forecast
        """
        # Get location coordinates
        location_data = location_service.get_lat_lon(city_name)
        
        if "error" in location_data:
            return location_data
        
        lat = location_data["latitude"]
        lon = location_data["longitude"]
        
        # Get weather data
        current_weather = self.get_current_weather(lat, lon)
        forecast = self.get_forecast(lat, lon)
        
        return {
            "location": location_data["location"],
            "current_weather": current_weather,
            "forecast": forecast
        }
        
    def format_weather_for_farmers(self, weather_data):
        """
        Format weather data in a way that's useful for farmers.
        
        Args:
            weather_data (dict): Raw weather data from the API
            
        Returns:
            dict: Formatted weather data with agricultural relevance
        """
        if "error" in weather_data:
            return weather_data
            
        try:
            result = {
                "location": weather_data.get("location", "Unknown location"),
                "current": self._format_current_weather(weather_data.get("current_weather", {})),
                "forecast": self._format_forecast(weather_data.get("forecast", {})),
                "agricultural_advice": self._generate_agricultural_advice(weather_data)
            }
            return result
        except Exception as e:
            error_msg = f"Error formatting weather data: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def _format_current_weather(self, current_data):
        """Format current weather data"""
        if "error" in current_data:
            return current_data
            
        try:
            main = current_data.get("main", {})
            weather = current_data.get("weather", [{}])[0]
            wind = current_data.get("wind", {})
            
            return {
                "temperature": {
                    "current": self._kelvin_to_celsius(main.get("temp")),
                    "feels_like": self._kelvin_to_celsius(main.get("feels_like")),
                    "min": self._kelvin_to_celsius(main.get("temp_min")),
                    "max": self._kelvin_to_celsius(main.get("temp_max"))
                },
                "humidity": main.get("humidity"),
                "pressure": main.get("pressure"),
                "condition": weather.get("main"),
                "description": weather.get("description"),
                "wind": {
                    "speed": wind.get("speed"),
                    "direction": wind.get("deg")
                },
                "rainfall": current_data.get("rain", {}).get("1h", 0)
            }
        except Exception as e:
            logger.error(f"Error formatting current weather: {str(e)}")
            return {"error": "Unable to format current weather data"}
    
    def _format_forecast(self, forecast_data):
        """Format forecast weather data"""
        if "error" in forecast_data:
            return forecast_data
            
        try:
            formatted_forecast = []
            
            # Get forecast list
            forecast_list = forecast_data.get("list", [])
            
            # Group by day (using date part of dt_txt)
            days = {}
            for item in forecast_list:
                date = item.get("dt_txt", "").split(" ")[0]
                if date:
                    if date not in days:
                        days[date] = []
                    days[date].append(item)
            
            # Format each day's forecast
            for date, items in days.items():
                # Calculate daily min/max
                temps = [item.get("main", {}).get("temp") for item in items if "main" in item]
                min_temp = min(temps) if temps else None
                max_temp = max(temps) if temps else None
                
                # Get weather conditions (mode)
                conditions = [item.get("weather", [{}])[0].get("main") for item in items if "weather" in item and item["weather"]]
                condition = max(set(conditions), key=conditions.count) if conditions else "Unknown"
                
                # Get rainfall probability (max)
                rainfall_probs = [item.get("pop", 0) for item in items]
                max_rainfall_prob = max(rainfall_probs) if rainfall_probs else 0
                
                # Calculate total rainfall
                total_rainfall = 0
                for item in items:
                    if "rain" in item and "3h" in item["rain"]:
                        total_rainfall += item["rain"]["3h"]
                
                formatted_forecast.append({
                    "date": date,
                    "temperature": {
                        "min": self._kelvin_to_celsius(min_temp),
                        "max": self._kelvin_to_celsius(max_temp)
                    },
                    "condition": condition,
                    "rainfall_probability": max_rainfall_prob * 100,  # Convert to percentage
                    "total_rainfall": total_rainfall
                })
            
            return formatted_forecast
        except Exception as e:
            logger.error(f"Error formatting forecast: {str(e)}")
            return {"error": "Unable to format forecast data"}
    
    def _generate_agricultural_advice(self, weather_data):
        """Generate agricultural advice based on weather data"""
        try:
            current = weather_data.get("current_weather", {})
            forecast = weather_data.get("forecast", {})
            
            advice = []
            
            # Current temperature advice
            current_temp = current.get("main", {}).get("temp")
            if current_temp:
                temp_celsius = self._kelvin_to_celsius(current_temp)
                if temp_celsius > 35:
                    advice.append("High temperature alert: Consider providing shade for sensitive crops and increasing irrigation frequency.")
                elif temp_celsius < 10:
                    advice.append("Low temperature alert: Protect frost-sensitive crops with covers if available.")
            
            # Rain forecast advice
            will_rain_soon = False
            heavy_rain_coming = False
            
            forecast_list = forecast.get("list", [])
            for item in forecast_list[:8]:  # Check next 24 hours (3-hour intervals)
                if "rain" in item:
                    will_rain_soon = True
                    if item["rain"].get("3h", 0) > 10:  # More than 10mm in 3 hours
                        heavy_rain_coming = True
                        break
            
            if heavy_rain_coming:
                advice.append("Heavy rainfall expected: Ensure proper drainage in your fields and consider delaying fertilizer application.")
            elif will_rain_soon:
                advice.append("Light rainfall expected: Good conditions for seed sowing and fertilizer application before the rain.")
            else:
                advice.append("No rainfall expected in the next 24 hours: Ensure adequate irrigation for your crops.")
            
            # Wind advice
            wind_speed = current.get("wind", {}).get("speed", 0)
            if wind_speed > 8:  # More than 8 m/s (about 29 km/h)
                advice.append("Strong winds alert: Delay spraying operations and check crop supports/stakes.")
            
            return advice
            
        except Exception as e:
            logger.error(f"Error generating agricultural advice: {str(e)}")
            return ["Weather data available but unable to generate specific agricultural advice."]
    
    def _kelvin_to_celsius(self, kelvin):
        """Convert Kelvin to Celsius"""
        if kelvin is None:
            return None
        return round(kelvin - 273.15, 1)