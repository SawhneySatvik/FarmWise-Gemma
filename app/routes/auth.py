from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from datetime import timedelta
from app.services.database import db
from app.models.user import User

auth_bp = Blueprint('auth', __name__, url_prefix='/api/auth')

@auth_bp.route('/register', methods=['POST'])
def register():
    """Register a new user"""
    data = request.json or {}
    
    # Validate required fields
    required_fields = ['username', 'password', 'phone_number']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing required field: {field}'}), 400
    
    # Check if username already exists
    if User.query.filter_by(username=data['username']).first():
        return jsonify({'error': 'Username already exists'}), 400
    
    # Check if phone number already exists
    if User.query.filter_by(phone_number=data['phone_number']).first():
        return jsonify({'error': 'Phone number already exists'}), 400
    
    # Check if email already exists (if provided)
    if data.get('email') and User.query.filter_by(email=data['email']).first():
        return jsonify({'error': 'Email already exists'}), 400
    
    # Create new user
    user = User(
        username=data['username'],
        password=data['password'],
        phone_number=data['phone_number'],
        email=data.get('email')
    )
    
    # Add optional fields if provided
    optional_fields = [
        'full_name', 'preferred_language',
        'state', 'district', 'village',
        'farm_size', 'farm_size_unit'
    ]
    
    for field in optional_fields:
        if field in data:
            setattr(user, field, data[field])
    
    # Save user to database
    db.session.add(user)
    db.session.commit()
    
    # Generate access token
    access_token = create_access_token(
        identity=str(user.id),
        expires_delta=timedelta(days=1)
    )
    
    return jsonify({
        'message': 'User registered successfully',
        'user': {
            'id': user.id,
            'username': user.username,
            'phone_number': user.phone_number,
            'email': user.email
        },
        'access_token': access_token
    }), 201

@auth_bp.route('/login', methods=['POST'])
def login():
    """Login an existing user"""
    data = request.json or {}
    
    # Check if login is via username, email, or phone
    username = data.get('username')
    email = data.get('email')
    phone_number = data.get('phone_number')
    password = data.get('password')
    
    if not (username or email or phone_number) or not password:
        return jsonify({'error': 'Username/email/phone and password are required'}), 400
    
    # Find user by username, email, or phone number
    user = None
    if username:
        user = User.query.filter_by(username=username).first()
    elif email:
        user = User.query.filter_by(email=email).first()
    elif phone_number:
        user = User.query.filter_by(phone_number=phone_number).first()
    
    # Check if user exists and password is correct
    if not user or not user.check_password(password):
        return jsonify({'error': 'Invalid credentials'}), 401
    
    # Check if user is active
    if not user.is_active:
        return jsonify({'error': 'Account is disabled'}), 403
    
    # Update last login time
    user.update_login()
    
    # Generate access token
    access_token = create_access_token(
        identity=str(user.id),
        expires_delta=timedelta(days=1)
    )
    
    return jsonify({
        'message': 'Login successful',
        'user': {
            'id': user.id,
            'username': user.username,
            'phone_number': user.phone_number,
            'email': user.email
        },
        'access_token': access_token
    })

@auth_bp.route('/profile', methods=['GET'])
@jwt_required()
def get_profile():
    """Get current user's profile"""
    current_user_id = get_jwt_identity()
    
    # Get user
    user = User.query.get(current_user_id)
    
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    # Return user profile
    return jsonify(user.get_profile())

@auth_bp.route('/profile', methods=['PUT'])
@jwt_required()
def update_profile():
    """Update current user's profile"""
    current_user_id = get_jwt_identity()
    
    # Get user
    user = User.query.get(current_user_id)
    
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    # Get data from request
    data = request.json or {}
    
    # Fields that cannot be updated
    forbidden_fields = ['id', 'username', 'phone_number', 'password_hash', 'created_at', 'is_active']
    
    # Update user fields
    for key, value in data.items():
        if key not in forbidden_fields and hasattr(user, key):
            setattr(user, key, value)
    
    # Handle password change separately if provided
    if 'password' in data:
        user.set_password(data['password'])
    
    # Save changes
    db.session.commit()
    
    # Return updated profile
    return jsonify({
        'message': 'Profile updated successfully',
        'user': user.get_profile()
    })

@auth_bp.route('/check-token', methods=['GET'])
@jwt_required()
def check_token():
    """Check if the current token is valid"""
    current_user_id = get_jwt_identity()
    
    # Get user
    user = User.query.get(current_user_id)
    
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    return jsonify({
        'valid': True,
        'user_id': current_user_id,
        'username': user.username
    }) 