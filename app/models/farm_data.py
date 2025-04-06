from datetime import datetime
import json
from app.services.database import db

class FarmData(db.Model):
    """Farm data model to store information about farms, crops, and livestock"""
    
    __tablename__ = 'farm_data'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    data_type = db.Column(db.String(50), nullable=False)  # crop, livestock, soil, irrigation, weather, etc.
    data_key = db.Column(db.String(100), nullable=False)
    data_value = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Define relationship with User
    user = db.relationship('User', backref=db.backref('farm_data', lazy=True))
    
    def __init__(self, user_id, data_type, data_key, data_value):
        """Initialize farm data entry"""
        self.user_id = user_id
        self.data_type = data_type
        self.data_key = data_key
        self.data_value = data_value if isinstance(data_value, str) else json.dumps(data_value)
    
    @property
    def value(self):
        """Return the data value, converting from JSON if possible"""
        try:
            return json.loads(self.data_value)
        except (TypeError, json.JSONDecodeError):
            return self.data_value
    
    @value.setter
    def value(self, value):
        """Set the data value, converting to JSON if needed"""
        self.data_value = value if isinstance(value, str) else json.dumps(value)
    
    @classmethod
    def get_user_farm_data(cls, user_id, data_type=None):
        """
        Get all farm data for a user, optionally filtered by type
        
        Args:
            user_id (int): User ID
            data_type (str, optional): Data type filter
            
        Returns:
            dict: Dictionary of farm data
        """
        query = cls.query.filter_by(user_id=user_id)
        
        if data_type:
            query = query.filter_by(data_type=data_type)
            
        result = {}
        for item in query.all():
            if item.data_type not in result:
                result[item.data_type] = {}
            
            result[item.data_type][item.data_key] = item.value
            
        return result
    
    @classmethod
    def get_farm_profile(cls, user_id):
        """
        Get a comprehensive farm profile for the user
        
        Args:
            user_id (int): User ID
            
        Returns:
            dict: Farm profile data
        """
        data = cls.get_user_farm_data(user_id)
        
        # Create a structured profile
        profile = {
            "farm": {
                "size": data.get("farm", {}).get("size"),
                "location": data.get("farm", {}).get("location"),
                "soil_type": data.get("soil", {}).get("type")
            },
            "crops": data.get("crop", {}),
            "livestock": data.get("livestock", {}),
            "irrigation": data.get("irrigation", {}),
            "equipment": data.get("equipment", {})
        }
        
        return profile
    
    @classmethod
    def update_farm_data(cls, user_id, data_type, data_key, data_value):
        """
        Update or create farm data
        
        Args:
            user_id (int): User ID
            data_type (str): Data type
            data_key (str): Data key
            data_value: Value to store
            
        Returns:
            FarmData: The updated or created FarmData instance
        """
        # Check if entry exists
        entry = cls.query.filter_by(
            user_id=user_id,
            data_type=data_type,
            data_key=data_key
        ).first()
        
        if entry:
            # Update existing entry
            entry.value = data_value
            entry.updated_at = datetime.utcnow()
        else:
            # Create new entry
            entry = cls(user_id, data_type, data_key, data_value)
            db.session.add(entry)
        
        db.session.commit()
        return entry
    
    def __repr__(self):
        """String representation of the farm data"""
        return f"<FarmData {self.data_type}:{self.data_key} for user {self.user_id}>"


class CropData(db.Model):
    """Model for storing crop-related data for a farm"""
    
    id = db.Column(db.Integer, primary_key=True)
    farm_id = db.Column(db.Integer, db.ForeignKey('farm_data.id'), nullable=False)
    
    # Crop details
    crop_name = db.Column(db.String(100), nullable=False)
    variety = db.Column(db.String(100))
    area_planted = db.Column(db.Float)  # in acres
    
    # Planting details
    planting_date = db.Column(db.Date)
    expected_harvest_date = db.Column(db.Date)
    
    # Yield information
    previous_yield = db.Column(db.Float)  # in kg/acre
    expected_yield = db.Column(db.Float)  # in kg/acre
    
    # Status
    is_current = db.Column(db.Boolean, default=True)
    status = db.Column(db.String(50))  # e.g., 'planted', 'growing', 'harvested'
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<CropData {self.id} - {self.crop_name}>'


class LivestockData(db.Model):
    """Model for storing livestock-related data for a farm"""
    
    id = db.Column(db.Integer, primary_key=True)
    farm_id = db.Column(db.Integer, db.ForeignKey('farm_data.id'), nullable=False)
    
    # Livestock details
    livestock_type = db.Column(db.String(100), nullable=False)  # e.g., 'cattle', 'goat', 'poultry'
    breed = db.Column(db.String(100))
    count = db.Column(db.Integer)
    
    # Management details
    housing_type = db.Column(db.String(100))
    feed_type = db.Column(db.String(100))
    
    # Production details
    production_purpose = db.Column(db.String(100))  # e.g., 'dairy', 'meat', 'eggs'
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<LivestockData {self.id} - {self.livestock_type}>' 