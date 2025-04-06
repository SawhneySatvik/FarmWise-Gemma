import requests
import json
from datetime import datetime, timedelta
import random
import os
from app.config import Config

class MarketService:
    """Service for fetching and analyzing agricultural market data"""
    
    def __init__(self):
        """Initialize market service"""
        # For MVP, we'll use mock data
        # In production, connect to Indian government APIs or third-party data sources
        self.markets = self._load_market_data()
        self.crop_prices = self._load_crop_prices()
    
    def get_markets_by_location(self, state=None, district=None):
        """Get agricultural markets near a location"""
        if not state:
            # Return a sample of markets if no location specified
            return {"markets": random.sample(self.markets, min(5, len(self.markets)))}
        
        # Filter by state
        filtered_markets = [m for m in self.markets if m["state"].lower() == state.lower()]
        
        # Further filter by district if provided
        if district:
            filtered_markets = [m for m in filtered_markets if m["district"].lower() == district.lower()]
        
        return {"markets": filtered_markets}
    
    def get_crop_prices(self, crop_name, state=None):
        """Get current and historical prices for a crop"""
        crop_name = crop_name.lower()
        
        # Check if crop exists in our data
        if crop_name not in self.crop_prices:
            return {"error": f"Price data for {crop_name} not available"}
        
        # Get price data
        price_data = self.crop_prices[crop_name]
        
        # Filter by state if provided
        if state:
            state = state.lower()
            if state in price_data["prices_by_state"]:
                current_price = price_data["prices_by_state"][state]["current"]
                historical = price_data["prices_by_state"][state]["historical"]
            else:
                # Fall back to national average
                current_price = price_data["national_avg"]["current"]
                historical = price_data["national_avg"]["historical"]
        else:
            # Use national average
            current_price = price_data["national_avg"]["current"]
            historical = price_data["national_avg"]["historical"]
        
        # Prepare response
        response = {
            "crop": crop_name,
            "unit": price_data["unit"],
            "current_price": current_price,
            "historical_prices": historical,
            "price_trend": self._analyze_price_trend(historical),
            "last_updated": datetime.now().strftime("%Y-%m-%d")
        }
        
        return response
    
    def get_market_trends(self, crop_name=None):
        """Get market trends and forecasts"""
        if crop_name:
            # Get specific crop trend
            crop_name = crop_name.lower()
            if crop_name not in self.crop_prices:
                return {"error": f"Trend data for {crop_name} not available"}
            
            # Get price data and forecast
            price_data = self.crop_prices[crop_name]
            forecast = self._generate_price_forecast(crop_name)
            
            return {
                "crop": crop_name,
                "current_price": price_data["national_avg"]["current"],
                "monthly_trend": self._calculate_monthly_trend(price_data["national_avg"]["historical"]),
                "forecast": forecast,
                "market_factors": self._get_market_factors(crop_name)
            }
        else:
            # Get overall market trends for top crops
            top_crops = list(self.crop_prices.keys())[:5]
            trends = {}
            
            for crop in top_crops:
                price_data = self.crop_prices[crop]
                forecast = self._generate_price_forecast(crop)
                
                trends[crop] = {
                    "current_price": price_data["national_avg"]["current"],
                    "price_change_30d": self._calculate_price_change(price_data["national_avg"]["historical"], 30),
                    "forecast_trend": forecast["trend"]
                }
            
            return {"market_trends": trends, "as_of": datetime.now().strftime("%Y-%m-%d")}
    
    def get_crop_demand(self, crop_name):
        """Get demand information for a crop"""
        crop_name = crop_name.lower()
        
        # Mock demand data
        demand_levels = {
            "rice": "high",
            "wheat": "medium",
            "cotton": "high",
            "sugarcane": "medium",
            "maize": "high",
            "pulses": "high",
            "millets": "medium",
            "oilseeds": "high",
            "vegetables": "high",
            "fruits": "medium"
        }
        
        demand_level = demand_levels.get(crop_name, "unknown")
        
        # Generate export data
        export_trend = "increasing" if demand_level == "high" else ("stable" if demand_level == "medium" else "decreasing")
        
        return {
            "crop": crop_name,
            "domestic_demand": demand_level,
            "export_demand": demand_level,
            "export_trend": export_trend,
            "top_export_markets": ["Bangladesh", "UAE", "Saudi Arabia", "Nepal", "Malaysia"] if demand_level != "low" else [],
            "demand_forecast": self._generate_demand_forecast(demand_level)
        }
    
    def _analyze_price_trend(self, historical_prices):
        """Analyze the price trend from historical data"""
        if len(historical_prices) < 2:
            return "stable"
        
        # Calculate average change
        changes = [
            (historical_prices[i]["price"] - historical_prices[i-1]["price"]) / historical_prices[i-1]["price"]
            for i in range(1, len(historical_prices))
        ]
        avg_change = sum(changes) / len(changes)
        
        # Determine trend
        if avg_change > 0.03:
            return "increasing"
        elif avg_change < -0.03:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_price_change(self, historical_prices, days):
        """Calculate price change over specified days"""
        if not historical_prices or len(historical_prices) < 2:
            return 0
        
        # Find price closest to N days ago
        target_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        # Find the closest date
        closest_old_price = None
        for price in historical_prices:
            if price["date"] <= target_date:
                closest_old_price = price["price"]
                break
        
        if not closest_old_price:
            closest_old_price = historical_prices[-1]["price"]
        
        # Get current price
        current_price = historical_prices[0]["price"]
        
        # Calculate percentage change
        change = ((current_price - closest_old_price) / closest_old_price) * 100
        return round(change, 1)
    
    def _calculate_monthly_trend(self, historical_prices):
        """Calculate monthly averages for the last 12 months"""
        if not historical_prices:
            return []
        
        # Group by month
        monthly_prices = {}
        for price in historical_prices:
            month_key = price["date"][:7]  # YYYY-MM
            if month_key not in monthly_prices:
                monthly_prices[month_key] = []
            monthly_prices[month_key].append(price["price"])
        
        # Calculate averages
        monthly_avg = [
            {"month": month, "avg_price": sum(prices) / len(prices)}
            for month, prices in monthly_prices.items()
        ]
        
        # Sort by month
        monthly_avg.sort(key=lambda x: x["month"], reverse=True)
        
        # Keep only last 12 months
        return monthly_avg[:12]
    
    def _generate_price_forecast(self, crop_name):
        """Generate a price forecast for the next 3 months"""
        if crop_name not in self.crop_prices:
            return {"error": "Crop not found"}
        
        price_data = self.crop_prices[crop_name]
        current_price = price_data["national_avg"]["current"]
        
        # Mock forecast based on seasonal patterns
        current_month = datetime.now().month
        
        # Different crops have different seasonal patterns
        if crop_name in ["rice", "wheat", "maize"]:
            # Staples tend to have less fluctuation
            variations = [0.02, 0.03, 0.04]
        elif crop_name in ["vegetables", "fruits"]:
            # Perishables have higher fluctuation
            variations = [0.05, 0.08, 0.10]
        else:
            # Other crops
            variations = [0.03, 0.05, 0.06]
        
        # Apply seasonal bias
        if 3 <= current_month <= 5:  # Spring
            direction = 1  # Prices tend to rise in spring for many crops
        elif 6 <= current_month <= 8:  # Summer
            direction = -1  # Summer harvest brings prices down
        elif 9 <= current_month <= 11:  # Fall
            direction = -1  # Fall harvest continues lower prices
        else:  # Winter
            direction = 1  # Prices rise in winter
        
        # Generate forecast
        forecast = []
        for i in range(3):
            month = (current_month + i) % 12
            if month == 0:
                month = 12
            
            # Calculate forecasted price
            price_change = variations[i] * direction
            forecasted_price = current_price * (1 + price_change)
            
            forecast.append({
                "month": month,
                "price": round(forecasted_price, 2)
            })
        
        # Determine overall trend
        last_price = forecast[-1]["price"]
        if last_price > current_price * 1.03:
            trend = "increasing"
        elif last_price < current_price * 0.97:
            trend = "decreasing"
        else:
            trend = "stable"
        
        return {
            "forecast": forecast,
            "trend": trend,
            "confidence": "medium"  # In a real system, this would be calculated
        }
    
    def _generate_demand_forecast(self, current_demand):
        """Generate a demand forecast based on current demand"""
        # Mock demand forecast
        forecast = []
        current_month = datetime.now().month
        
        for i in range(3):
            month = (current_month + i) % 12
            if month == 0:
                month = 12
            
            # Determine demand level for this month
            if current_demand == "high":
                demand = "high" if random.random() < 0.8 else "medium"
            elif current_demand == "medium":
                if random.random() < 0.4:
                    demand = "high"
                elif random.random() < 0.4:
                    demand = "low"
                else:
                    demand = "medium"
            else:  # low
                demand = "low" if random.random() < 0.7 else "medium"
            
            forecast.append({
                "month": month,
                "demand": demand
            })
        
        return forecast
    
    def _get_market_factors(self, crop_name):
        """Get factors affecting market prices for a crop"""
        # Mock market factors
        common_factors = [
            "Seasonal demand fluctuations",
            "Government MSP policy",
            "International market prices",
            "Fuel price impacts on transport costs",
            "Storage availability"
        ]
        
        crop_specific_factors = {
            "rice": ["Export regulations", "Monsoon performance", "Government procurement"],
            "wheat": ["Global wheat shortage", "European production", "Public distribution system"],
            "cotton": ["Textile industry demand", "International cotton prices", "Synthetic fiber competition"],
            "sugarcane": ["Ethanol blending policy", "Sugar mill processing capacity", "Export quotas"],
            "maize": ["Poultry feed demand", "Industrial starch consumption", "Ethanol production"],
            "vegetables": ["Cold storage capacity", "Perishability", "Urban consumption patterns"],
            "fruits": ["Export quality standards", "Cold chain infrastructure", "Processing industry demand"],
            "pulses": ["Import policies", "Government buffer stocks", "Protein consumption trends"],
            "oilseeds": ["Edible oil import duties", "Biodiesel demand", "International vegetable oil markets"]
        }
        
        # Get crop-specific factors or default to empty list
        specific_factors = crop_specific_factors.get(crop_name, [])
        
        # Combine factors
        return {
            "common_factors": common_factors,
            "crop_specific_factors": specific_factors
        }
    
    def _load_market_data(self):
        """Load mock market data"""
        return [
            {"id": 1, "name": "Azadpur Mandi", "state": "Delhi", "district": "North Delhi", "type": "Wholesale"},
            {"id": 2, "name": "Ghazipur Mandi", "state": "Delhi", "district": "East Delhi", "type": "Wholesale"},
            {"id": 3, "name": "Devi Ahilya Bai Holkar Mandi", "state": "Madhya Pradesh", "district": "Indore", "type": "Wholesale"},
            {"id": 4, "name": "Yeshwanthpur APMC", "state": "Karnataka", "district": "Bangalore", "type": "Wholesale"},
            {"id": 5, "name": "Bowenpally Market", "state": "Telangana", "district": "Hyderabad", "type": "Wholesale"},
            {"id": 6, "name": "Vashi APMC", "state": "Maharashtra", "district": "Mumbai", "type": "Wholesale"},
            {"id": 7, "name": "Lasalgaon APMC", "state": "Maharashtra", "district": "Nashik", "type": "Wholesale"},
            {"id": 8, "name": "Koyambedu Market", "state": "Tamil Nadu", "district": "Chennai", "type": "Wholesale"},
            {"id": 9, "name": "Gultekdi Market Yard", "state": "Maharashtra", "district": "Pune", "type": "Wholesale"},
            {"id": 10, "name": "Sirhind Mandi", "state": "Punjab", "district": "Fatehgarh Sahib", "type": "Wholesale"}
        ]
    
    def _load_crop_prices(self):
        """Load mock crop price data"""
        # Generate some mock historical data
        def generate_history(base_price, volatility=0.2, days=90):
            history = []
            current_date = datetime.now()
            
            price = base_price
            for i in range(days):
                date = (current_date - timedelta(days=i)).strftime("%Y-%m-%d")
                
                # Random walk with some seasonal pattern
                change = (random.random() - 0.5) * volatility
                
                # Add some seasonality
                season_factor = 0.1 * math.sin(2 * math.pi * i / 180)
                change += season_factor
                
                price *= (1 + change)
                history.append({"date": date, "price": round(price, 2)})
            
            return history
        
        import math
        
        return {
            "rice": {
                "unit": "quintal",
                "national_avg": {
                    "current": 2200,
                    "historical": generate_history(2200, 0.02)
                },
                "prices_by_state": {
                    "punjab": {
                        "current": 2150,
                        "historical": generate_history(2150, 0.03)
                    },
                    "uttar pradesh": {
                        "current": 2050,
                        "historical": generate_history(2050, 0.025)
                    },
                    "west bengal": {
                        "current": 2250,
                        "historical": generate_history(2250, 0.02)
                    }
                }
            },
            "wheat": {
                "unit": "quintal",
                "national_avg": {
                    "current": 1950,
                    "historical": generate_history(1950, 0.03)
                },
                "prices_by_state": {
                    "punjab": {
                        "current": 2000,
                        "historical": generate_history(2000, 0.02)
                    },
                    "haryana": {
                        "current": 1975,
                        "historical": generate_history(1975, 0.025)
                    },
                    "madhya pradesh": {
                        "current": 1900,
                        "historical": generate_history(1900, 0.03)
                    }
                }
            },
            "cotton": {
                "unit": "quintal",
                "national_avg": {
                    "current": 6500,
                    "historical": generate_history(6500, 0.05)
                },
                "prices_by_state": {
                    "gujarat": {
                        "current": 6600,
                        "historical": generate_history(6600, 0.06)
                    },
                    "maharashtra": {
                        "current": 6450,
                        "historical": generate_history(6450, 0.055)
                    }
                }
            },
            "sugarcane": {
                "unit": "quintal",
                "national_avg": {
                    "current": 350,
                    "historical": generate_history(350, 0.01)
                },
                "prices_by_state": {
                    "uttar pradesh": {
                        "current": 345,
                        "historical": generate_history(345, 0.01)
                    },
                    "karnataka": {
                        "current": 360,
                        "historical": generate_history(360, 0.015)
                    }
                }
            },
            "maize": {
                "unit": "quintal",
                "national_avg": {
                    "current": 1850,
                    "historical": generate_history(1850, 0.04)
                },
                "prices_by_state": {
                    "karnataka": {
                        "current": 1900,
                        "historical": generate_history(1900, 0.035)
                    },
                    "bihar": {
                        "current": 1800,
                        "historical": generate_history(1800, 0.045)
                    }
                }
            },
            "pulses": {
                "unit": "quintal",
                "national_avg": {
                    "current": 6000,
                    "historical": generate_history(6000, 0.06)
                },
                "prices_by_state": {
                    "madhya pradesh": {
                        "current": 5900,
                        "historical": generate_history(5900, 0.055)
                    },
                    "maharashtra": {
                        "current": 6100,
                        "historical": generate_history(6100, 0.065)
                    }
                }
            },
            "vegetables": {
                "unit": "quintal",
                "national_avg": {
                    "current": 2000,
                    "historical": generate_history(2000, 0.08)
                },
                "prices_by_state": {
                    "maharashtra": {
                        "current": 2100,
                        "historical": generate_history(2100, 0.09)
                    },
                    "tamil nadu": {
                        "current": 1950,
                        "historical": generate_history(1950, 0.085)
                    }
                }
            },
            "fruits": {
                "unit": "quintal",
                "national_avg": {
                    "current": 3500,
                    "historical": generate_history(3500, 0.07)
                },
                "prices_by_state": {
                    "maharashtra": {
                        "current": 3600,
                        "historical": generate_history(3600, 0.075)
                    },
                    "himachal pradesh": {
                        "current": 3400,
                        "historical": generate_history(3400, 0.08)
                    }
                }
            }
        } 