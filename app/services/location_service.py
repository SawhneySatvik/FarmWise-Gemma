from geopy.geocoders import Nominatim
import logging

logger = logging.getLogger(__name__)

def get_lat_lon(city_name):
    """
    Extract latitude and longitude for a given city using Geopy.
    
    Args:
        city_name (str): Name of the city to geocode
        
    Returns:
        dict: Dictionary with latitude, longitude and location address
    """
    geolocator = Nominatim(user_agent="FarmWise-LocationService")
    
    try:
        location = geolocator.geocode(city_name)
        if location:
            return {
                "latitude": location.latitude, 
                "longitude": location.longitude, 
                "location": location.address
            }
        else:
            logger.warning(f"City not found: {city_name}")
            return {"error": "City not found"}
    except Exception as e:
        logger.error(f"Geolocation error for {city_name}: {str(e)}")
        return {"error": f"Geolocation error: {str(e)}"}

# print(get_lat_lon("New Delhi"))
# print("=========================")
# print(get_lat_lon("Chennai"))
