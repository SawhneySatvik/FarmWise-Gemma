from flask import Flask, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager
import os
from datetime import timedelta

def create_app(config=None):
    """Create and configure the Flask application"""
    app = Flask(__name__)
    
    # Get absolute path to server directory (one level up from app directory)
    app_dir = os.path.abspath(os.path.dirname(__file__))
    server_dir = os.path.dirname(app_dir)
    
    # Load configuration
    app.config.from_mapping(
        SECRET_KEY=os.environ.get('SECRET_KEY', 'dev-key-for-development-only'),
        JWT_SECRET_KEY=os.environ.get('JWT_SECRET_KEY', 'jwt-dev-key-for-development-only'),
        JWT_ACCESS_TOKEN_EXPIRES=timedelta(days=1),
        SQLALCHEMY_DATABASE_URI=os.environ.get('DATABASE_URI', f'sqlite:///{os.path.join(server_dir, "database", "farming_database.db")}'),
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        UPLOAD_FOLDER=os.path.join(app_dir, 'uploads'),
        MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16 MB max upload
        DEBUG=os.environ.get('FLASK_ENV', 'development') == 'development'
    )
    
    # Override config if provided
    if config:
        app.config.update(config)
    
    # Enable CORS with proper configuration
    CORS(app, resources={r"/api/*": {"origins": "*", "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"], "allow_headers": ["Content-Type", "Authorization"]}})
    
    # Initialize JWT
    jwt = JWTManager(app)
    
    # Ensure uploads directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Ensure database directory exists
    db_path = os.path.dirname(app.config['SQLALCHEMY_DATABASE_URI'].replace('sqlite:///', ''))
    os.makedirs(db_path, exist_ok=True)
    
    # Initialize database
    from app.services.database import init_db
    init_db(app)
    
    # Register routes
    from app.routes.auth import auth_bp
    from app.routes.chat import chat_bp
    from app.routes.admin import admin_bp
    
    app.register_blueprint(auth_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(admin_bp)
    
    # Health check route
    @app.route('/api/health', methods=['GET'])
    def health_check():
        return jsonify({
            'status': 'healthy',
            'version': '0.1.0',
            'environment': os.environ.get('FLASK_ENV', 'development')
        })
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Not found'}), 404
    
    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({'error': 'Bad request'}), 400
    
    @app.errorhandler(500)
    def server_error(error):
        app.logger.error(f'Server error: {error}')
        return jsonify({'error': 'Internal server error'}), 500
    
    # Remove duplicate CORS configuration
    # CORS(app, resources={r"/*": {"origins": "*"}})

    return app
