from .location_service import *
from .weather_service import *

def get_weather_data():
    """
    Fetch weather data for a given city.
    """
    weather_service = WeatherService()
    
    city_name = input("Enter city name: ")
    location_data = get_lat_lon(city_name)
    
    if "error" in location_data:
        return location_data
    
    lat = location_data["latitude"]
    lon = location_data["longitude"]
    
    current_weather = weather_service.get_current_weather(lat, lon)
    five_day_forecast = weather_service.get_five_day_forecast(lat, lon)
    
    return {
        "location": location_data["location"],
        "current_weather": current_weather,
        "five_day_forecast": five_day_forecast
    }