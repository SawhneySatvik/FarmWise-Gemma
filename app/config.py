import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration settings"""
    
    # Flask Configuration
    SECRET_KEY = os.environ.get("SECRET_KEY", "default_secret_key")
    JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "default_jwt_secret_key")
    DEBUG = os.environ.get("DEBUG", "False").lower() == "true"
    
    # Database Configuration
    SQLALCHEMY_DATABASE_URI = os.environ.get('SQLALCHEMY_DATABASE_URI', 'sqlite:///database/farming_database.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # LLM Configuration
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
    LLM_MODEL = os.environ.get('LLM_MODEL', 'gemini-1.5-pro')
    
    # OpenWeatherMap API Configuration
    OPEN_WEATHER_API_KEY = os.environ.get('OPEN_WEATHER_API_KEY')
    CURRENT_WEATHER_DATA_URL = os.environ.get(
        'CURRENT_WEATHER_DATA_URL', 
        'https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_key}'
    )
    FIVE_DAY_FORECAST_DATA_URL = os.environ.get(
        'FIVE_DAY_FORECAST_DATA_URL',
        'https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_key}'
    )
    
    # Logging Configuration
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')