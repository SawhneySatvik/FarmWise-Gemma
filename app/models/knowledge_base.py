from datetime import datetime
from app.services.database import db
import json

class KnowledgeBase(db.Model):
    """Model for storing knowledge base entries for RAG"""
    
    __tablename__ = 'knowledge_base'
    
    id = db.Column(db.Integer, primary_key=True)
    
    # Content categorization
    category = db.Column(db.String(50), nullable=False)  # e.g., 'crop', 'soil', 'weather', 'pest', 'market'
    subcategory = db.Column(db.String(50))  # e.g., 'rice', 'wheat', 'cotton'
    
    # Content
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    
    # Metadata
    source = db.Column(db.String(200))
    source_url = db.Column(db.String(500))
    region_relevance = db.Column(db.String(50))  # e.g., 'north_india', 'andhra_pradesh'
    
    # Embeddings (stored as comma-separated values)
    embedding = db.Column(db.Text)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<KnowledgeBase {self.id} - {self.title}>'
    
    def to_dict(self):
        """Convert to dictionary for API responses"""
        return {
            'id': self.id,
            'category': self.category,
            'subcategory': self.subcategory,
            'title': self.title,
            'content': self.content,
            'source': self.source,
            'source_url': self.source_url,
            'region_relevance': self.region_relevance,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }


class CropGuide(db.Model):
    """Model for storing detailed crop guides"""
    
    id = db.Column(db.Integer, primary_key=True)
    
    # Crop information
    crop_name = db.Column(db.String(100), nullable=False)
    scientific_name = db.Column(db.String(100))
    crop_family = db.Column(db.String(100))
    crop_type = db.Column(db.String(50))  # e.g., 'cereal', 'pulse', 'vegetable', 'fruit'
    
    # Growing conditions
    soil_requirements = db.Column(db.Text)
    water_requirements = db.Column(db.Text)
    temperature_range = db.Column(db.String(50))
    climate_requirements = db.Column(db.Text)
    
    # Cultivation practices
    sowing_season = db.Column(db.String(100))
    harvesting_season = db.Column(db.String(100))
    seed_rate = db.Column(db.String(50))
    spacing = db.Column(db.String(50))
    
    # Management
    fertilizer_requirements = db.Column(db.Text)
    pest_management = db.Column(db.Text)
    disease_management = db.Column(db.Text)
    irrigation_schedule = db.Column(db.Text)
    
    # Economic aspects
    average_yield = db.Column(db.String(50))
    market_information = db.Column(db.Text)
    
    # Metadata
    region_specific = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<CropGuide {self.id} - {self.crop_name}>'


class KnowledgeItem(db.Model):
    """Model for storing agricultural knowledge items for Retrieval-Augmented Generation"""
    
    __tablename__ = 'knowledge_item'
    
    id = db.Column(db.Integer, primary_key=True)
    topic = db.Column(db.String(100), nullable=False, index=True)  # e.g., crops, pests, weather
    subtopic = db.Column(db.String(100), nullable=False, index=True)  # e.g., rice, aphids, monsoon
    content = db.Column(db.Text, nullable=False)  # The actual knowledge content
    source = db.Column(db.String(200))  # Source of information
    item_metadata = db.Column(db.Text)  # JSON metadata (region applicability, confidence, etc.)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __init__(self, topic, subtopic, content, source=None, metadata=None):
        """Initialize knowledge item"""
        self.topic = topic
        self.subtopic = subtopic
        self.content = content
        self.source = source or "Internal database"
        self.item_metadata = json.dumps(metadata or {})
    
    @property
    def metadata_dict(self):
        """Return metadata as a dictionary"""
        try:
            return json.loads(self.item_metadata)
        except (TypeError, json.JSONDecodeError):
            return {}
    
    @metadata_dict.setter
    def metadata_dict(self, value):
        """Set metadata from a dictionary"""
        self.item_metadata = json.dumps(value or {})
    
    @classmethod
    def search(cls, query, limit=10):
        """
        Simple search for knowledge items
        
        Args:
            query (str): Search query
            limit (int): Maximum number of results
            
        Returns:
            list: Matching knowledge items
        """
        # Split query into words
        search_terms = query.lower().split()
        
        # Build query with basic relevance scoring
        # This is a simple approach; a production system would use 
        # proper vector embeddings and semantic search
        results = cls.query.filter(
            db.or_(
                *[cls.topic.ilike(f'%{term}%') for term in search_terms],
                *[cls.subtopic.ilike(f'%{term}%') for term in search_terms],
                *[cls.content.ilike(f'%{term}%') for term in search_terms]
            )
        ).limit(limit).all()
        
        return results
    
    @classmethod
    def get_by_topic(cls, topic, subtopic=None):
        """
        Get knowledge items by topic and optional subtopic
        
        Args:
            topic (str): Main topic
            subtopic (str, optional): Subtopic
            
        Returns:
            list: Matching knowledge items
        """
        query = cls.query.filter_by(topic=topic)
        
        if subtopic:
            query = query.filter_by(subtopic=subtopic)
            
        return query.all()
    
    @classmethod
    def add_item(cls, topic, subtopic, content, source=None, metadata=None):
        """
        Add a new knowledge item
        
        Args:
            topic (str): Main topic
            subtopic (str): Subtopic
            content (str): Knowledge content
            source (str, optional): Source of information
            metadata (dict, optional): Additional metadata
            
        Returns:
            KnowledgeItem: The newly created item
        """
        item = cls(topic, subtopic, content, source, metadata)
        db.session.add(item)
        db.session.commit()
        return item
    
    def update_content(self, new_content, new_source=None, new_metadata=None):
        """
        Update the content of a knowledge item
        
        Args:
            new_content (str): New content
            new_source (str, optional): New source
            new_metadata (dict, optional): New metadata
            
        Returns:
            KnowledgeItem: The updated item
        """
        self.content = new_content
        
        if new_source:
            self.source = new_source
            
        if new_metadata:
            current_metadata = self.metadata_dict
            current_metadata.update(new_metadata)
            self.metadata_dict = current_metadata
            
        self.updated_at = datetime.utcnow()
        db.session.commit()
        return self
    
    def __repr__(self):
        """String representation of the knowledge item"""
        return f"<KnowledgeItem {self.topic}/{self.subtopic}>" 