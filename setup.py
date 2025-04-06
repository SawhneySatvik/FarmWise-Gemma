import os
import sys
import argparse
from app import create_app
from app.services.database import db

def setup_database():
    """Initialize database and create tables"""
    app = create_app()
    
    with app.app_context():
        # Import all models to ensure they are registered
        from app.models.user import User
        from app.models.chat_session import ChatSession, ChatMessage
        from app.models.farm_data import FarmData
        from app.models.knowledge_base import KnowledgeItem
        
        # Create database directory if it doesn't exist
        db_path = os.path.dirname(app.config['SQLALCHEMY_DATABASE_URI'].replace('sqlite:///', ''))
        if not os.path.exists(db_path):
            os.makedirs(db_path)
        
        # Create tables
        db.create_all()
        
        print("Database setup complete!")

def create_admin_user():
    """Create an admin user for the application"""
    app = create_app()
    
    with app.app_context():
        from app.models.user import User
        
        # Check if admin already exists
        admin = User.query.filter_by(username='admin').first()
        if admin:
            print("Admin user already exists!")
            return
        
        # Create admin user
        admin = User(
            username='admin',
            password='adminpass',  # This would be hashed by the User model
            phone_number='9999999999',
            email='admin@farmwise.com'
        )
        
        # Set admin flag
        admin.is_admin = True
        
        # Add to database
        db.session.add(admin)
        db.session.commit()
        
        print("Admin user created: admin (phone: 9999999999, password: adminpass)")

def create_sample_data():
    """Create sample data for testing"""
    app = create_app()
    
    with app.app_context():
        from app.models.user import User
        from app.models.knowledge_base import KnowledgeItem
        from app.models.farm_data import FarmData
        
        # Create test user if it doesn't exist
        user = User.query.filter_by(username='testfarmer').first()
        if not user:
            user = User(
                username='testfarmer',
                password='password',
                phone_number='9876543210',
                email='farmer@example.com',
                full_name='Test Farmer',
                preferred_language='en',
                state='Punjab',
                district='Ludhiana',
                village='Test Village',
                farm_size=5.0,
                farm_size_unit='acres'
            )
            db.session.add(user)
            db.session.commit()
            print("Test user created: testfarmer (phone: 9876543210, password: password)")
        
        # Add sample knowledge base items
        if KnowledgeItem.query.count() < 5:
            knowledge_items = [
                KnowledgeItem(
                    topic='Crops',
                    subtopic='Wheat',
                    content='Wheat is a major rabi crop grown in India, typically planted in November-December and harvested in April-May. It requires 4-5 irrigations and moderate temperatures for optimal growth.',
                    source='Indian Agricultural Research Institute',
                    metadata={'region': 'North India', 'confidence': 'high'}
                ),
                KnowledgeItem(
                    topic='Soil',
                    subtopic='Testing',
                    content='Soil testing should be done every 2-3 years to monitor nutrient levels. Collect samples from 15-20 spots in your field at a depth of 15-20 cm. Mix samples thoroughly before sending to a testing lab.',
                    source='Punjab Agricultural University',
                    metadata={'region': 'All India', 'confidence': 'high'}
                ),
                KnowledgeItem(
                    topic='Pests',
                    subtopic='Aphids',
                    content='Aphids are small sap-sucking insects that affect many crops. Use neem oil spray (5ml/liter) as an organic control method. For severe infestations, consider imidacloprid at recommended doses.',
                    source='Traditional Knowledge',
                    metadata={'region': 'All India', 'confidence': 'medium'}
                ),
                KnowledgeItem(
                    topic='Market',
                    subtopic='MSP',
                    content='Minimum Support Price (MSP) is announced by the Government of India twice a year for 23 crops to ensure farmers receive a minimum price for their produce regardless of market fluctuations.',
                    source='Ministry of Agriculture & Farmers Welfare',
                    metadata={'region': 'All India', 'confidence': 'high'}
                ),
                KnowledgeItem(
                    topic='Livestock',
                    subtopic='Dairy',
                    content='For optimal milk production, dairy cows should be fed a balanced ration containing 16-18% protein, along with adequate green fodder, dry fodder, and mineral mixture. Adult cows require about 30-35 liters of water daily.',
                    source='National Dairy Research Institute',
                    metadata={'region': 'All India', 'confidence': 'high'}
                )
            ]
            
            for item in knowledge_items:
                db.session.add(item)
            
            db.session.commit()
            print("Added 5 sample knowledge items")
        
        # Add farm data for test user
        if FarmData.query.filter_by(user_id=user.id).count() < 3:
            farm_data = [
                FarmData(user.id, 'crop', 'wheat', {'is_primary': True, 'area_planted': 2.5, 'planting_date': '2023-11-15'}),
                FarmData(user.id, 'crop', 'rice', {'is_primary': True, 'area_planted': 2.5, 'planting_date': '2023-06-15'}),
                FarmData(user.id, 'soil', 'type', 'clay loam'),
                FarmData(user.id, 'irrigation', 'type', 'tube well'),
                FarmData(user.id, 'livestock', 'cattle', {'count': 3, 'breed': 'Holstein-Friesian cross'})
            ]
            
            for data in farm_data:
                db.session.add(data)
            
            db.session.commit()
            print("Added sample farm data for test user")
        
        print("Sample data setup complete!")

def main():
    """Main function to run setup operations"""
    parser = argparse.ArgumentParser(description='FarmWise Setup Script')
    parser.add_argument('--init-db', action='store_true', help='Initialize database and create tables')
    parser.add_argument('--create-admin', action='store_true', help='Create admin user')
    parser.add_argument('--sample-data', action='store_true', help='Create sample data')
    parser.add_argument('--all', action='store_true', help='Run all setup operations')
    
    args = parser.parse_args()
    
    if args.init_db or args.all:
        setup_database()
    
    if args.create_admin or args.all:
        create_admin_user()
    
    if args.sample_data or args.all:
        create_sample_data()
    
    if len(sys.argv) == 1:
        parser.print_help()

if __name__ == '__main__':
    main() 