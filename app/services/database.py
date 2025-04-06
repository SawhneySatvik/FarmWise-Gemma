from flask_sqlalchemy import SQLAlchemy

# Initialize SQLAlchemy extension
db = SQLAlchemy()

def init_db(app):
    """
    Initialize the database
    
    Args:
        app: Flask application instance
    """
    # Configure the database URI
    if not app.config.get("SQLALCHEMY_DATABASE_URI"):
        # Default to SQLite database
        app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///app/database/farming_database.db"
    
    # Disable track modifications for better performance
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    
    # Initialize database with app
    db.init_app(app)
    
    # Create tables
    with app.app_context():
        # Import all models to ensure they are registered with SQLAlchemy
        from app.models.user import User
        from app.models.chat_session import ChatSession
        from app.models.farm_data import FarmData
        from app.models.knowledge_base import KnowledgeItem
        
        # Create tables
        db.create_all()
        
        # Initialize default data if needed
        _initialize_default_data()

def _initialize_default_data():
    """Initialize the database with default data if tables are empty"""
    from app.models.knowledge_base import KnowledgeItem
    
    # Check if knowledge base is empty
    if KnowledgeItem.query.count() == 0:
        # Add some basic knowledge items for testing
        default_items = [
            KnowledgeItem(
                topic="Weather",
                subtopic="Monsoon",
                content="The southwest monsoon in India typically begins in June and lasts until September, bringing the majority of annual rainfall to most of the country.",
                source="Indian Meteorological Department",
                metadata={"region": "All India", "confidence": "high"}
            ),
            KnowledgeItem(
                topic="Crops",
                subtopic="Rice",
                content="Rice requires standing water in the field during most of its growing period. The ideal temperature range is 20-35°C with high humidity.",
                source="Indian Agricultural Research Institute",
                metadata={"region": "All India", "confidence": "high"}
            ),
            KnowledgeItem(
                topic="Pest Management",
                subtopic="Neem Oil",
                content="Neem oil is an effective, natural pesticide for controlling aphids, mites, and many other common pests. Mix 5ml of neem oil with 1 liter of water and add a few drops of dish soap as an emulsifier.",
                source="Traditional Knowledge",
                metadata={"region": "All India", "confidence": "medium"}
            ),
            KnowledgeItem(
                topic="Soil Health",
                subtopic="Organic Matter",
                content="Adding organic matter like compost or farmyard manure improves soil structure, water retention, and nutrient availability. Apply 5-10 tons per hectare before planting.",
                source="Indian Council of Agricultural Research",
                metadata={"region": "All India", "confidence": "high"}
            ),
            KnowledgeItem(
                topic="Market Trends",
                subtopic="MSP",
                content="Minimum Support Price (MSP) is announced by the Government of India twice a year for 23 crops. It ensures farmers receive a minimum price for their produce regardless of market fluctuations.",
                source="Ministry of Agriculture & Farmers Welfare",
                metadata={"region": "All India", "confidence": "high"}
            )
        ]
        
        # Add items to database
        for item in default_items:
            db.session.add(item)
        
        db.session.commit() 