from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from app.services.database import db

class User(db.Model):
    """User model for storing farmer profiles"""
    
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)
    
    # Profile data
    phone_number = db.Column(db.String(20), nullable=False)
    full_name = db.Column(db.String(100))
    preferred_language = db.Column(db.String(20), default='en')  # Default to English
    
    # Location data
    state = db.Column(db.String(50))
    district = db.Column(db.String(50))
    village = db.Column(db.String(100))
    
    # Farm data
    farm_size = db.Column(db.Float)  # in acres
    farm_size_unit = db.Column(db.String(10), default='acres')
    
    # Chat sessions - removed backref to fix conflict
    chat_sessions = db.relationship('ChatSession')
    
    def __init__(self, username, password, phone_number, email=None, **kwargs):
        """
        Initialize a user
        
        Args:
            username (str): Username
            password (str): Plaintext password
            phone_number (str): Phone number
            email (str, optional): Email address
            **kwargs: Additional user attributes
        """
        self.username = username
        self.phone_number = phone_number
        self.email = email
        self.set_password(password)
        
        # Set any additional attributes from kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def set_password(self, password):
        """
        Set the password hash
        
        Args:
            password (str): Plaintext password
        """
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """
        Check if the password is correct
        
        Args:
            password (str): Plaintext password to check
            
        Returns:
            bool: True if the password is correct
        """
        return check_password_hash(self.password_hash, password)
    
    def update_login(self):
        """Update the last login timestamp"""
        self.last_login = datetime.utcnow()
        db.session.commit()
    
    def get_profile(self):
        """
        Get user profile data
        
        Returns:
            dict: User profile information
        """
        return {
            'username': self.username,
            'email': self.email,
            'full_name': self.full_name,
            'phone_number': self.phone_number,
            'preferred_language': self.preferred_language,
            'location': {
                'state': self.state,
                'district': self.district,
                'village': self.village
            },
            'farm_size': self.farm_size,
            'farm_size_unit': self.farm_size_unit,
            'created_at': self.created_at,
            'last_login': self.last_login
        }
    
    def get_farm_context(self):
        """
        Get farming context for the user
        
        Returns:
            dict: Farming context for AI interactions
        """
        # Get all farm data
        from app.models.farm_data import FarmData
        farm_data = FarmData.get_user_farm_data(self.id)
        
        # Create context with basic user info
        context = {
            'location': f"{self.village}, {self.district}, {self.state}" if self.village and self.district and self.state else None,
            'language_preference': self.preferred_language,
            'farm_size': self.farm_size,
            'primary_crops': None,
            'soil_type': None,
            'has_livestock': False,
            'irrigation_type': None
        }
        
        # Extract key information from farm data
        if 'crop' in farm_data:
            primary_crops = []
            for crop, data in farm_data['crop'].items():
                if isinstance(data, dict) and data.get('is_primary', False):
                    primary_crops.append(crop)
            if primary_crops:
                context['primary_crops'] = ', '.join(primary_crops)
        
        if 'soil' in farm_data and 'type' in farm_data['soil']:
            context['soil_type'] = farm_data['soil']['type']
        
        if 'livestock' in farm_data and farm_data['livestock']:
            context['has_livestock'] = True
        
        if 'irrigation' in farm_data and 'type' in farm_data['irrigation']:
            context['irrigation_type'] = farm_data['irrigation']['type']
        
        return context
    
    @classmethod
    def get_by_username(cls, username):
        """
        Get a user by username
        
        Args:
            username (str): Username to look up
            
        Returns:
            User: User instance or None
        """
        return cls.query.filter_by(username=username).first()
    
    @classmethod
    def get_by_email(cls, email):
        """
        Get a user by email
        
        Args:
            email (str): Email to look up
            
        Returns:
            User: User instance or None
        """
        return cls.query.filter_by(email=email).first()
    
    @classmethod
    def get_by_phone(cls, phone_number):
        """
        Get a user by phone number
        
        Args:
            phone_number (str): Phone number to look up
            
        Returns:
            User: User instance or None
        """
        return cls.query.filter_by(phone_number=phone_number).first()
    
    def __repr__(self):
        """String representation of the user"""
        return f"<User {self.username}>"
