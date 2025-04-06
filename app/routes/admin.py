from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from datetime import datetime

from app.services.database import db
from app.models.user import User
from app.models.knowledge_base import KnowledgeBase, CropGuide, KnowledgeItem
from app.models.chat_session import ChatSession

admin_bp = Blueprint('admin', __name__, url_prefix='/api/admin')

# Admin access verification
def verify_admin_access():
    """Verify if the current user has admin access"""
    user_id = get_jwt_identity()
    user = User.query.get(user_id)
    
    # For MVP, just check if user ID is 1 (first user is admin)
    # In production, add a proper role-based system
    return user and user.id == 1

# Admin authentication middleware
def admin_required(fn):
    """Custom decorator to check if user is an admin"""
    @jwt_required()
    def wrapper(*args, **kwargs):
        current_user_id = get_jwt_identity()
        user = User.query.get(current_user_id)
        
        # Check if user exists and is admin
        # Note: Admin field would need to be added to User model
        if not user or not getattr(user, 'is_admin', False):
            return jsonify({'error': 'Admin privileges required'}), 403
        
        return fn(*args, **kwargs)
    
    # Preserve function metadata for Flask
    wrapper.__name__ = fn.__name__
    return wrapper

@admin_bp.route('/stats', methods=['GET'])
@jwt_required()
def get_stats():
    """Get system statistics for admin dashboard"""
    if not verify_admin_access():
        return jsonify({'error': 'Admin access required'}), 403
    
    # Get basic stats
    total_users = User.query.count()
    recent_users = User.query.filter(
        User.created_at >= datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    ).count()
    
    from app.models.chat_session import ChatMessage
    total_sessions = ChatSession.query.count()
    total_messages = ChatMessage.query.count()
    
    # Return stats
    return jsonify({
        'users': {
            'total': total_users,
            'recent': recent_users
        },
        'activity': {
            'sessions': total_sessions,
            'messages': total_messages
        },
        'knowledge_base': {
            'entries': KnowledgeBase.query.count(),
            'crop_guides': CropGuide.query.count()
        }
    }), 200

@admin_bp.route('/knowledge', methods=['GET'])
@admin_required
def get_knowledge_items():
    """Get all knowledge base items with filtering options"""
    # Get query parameters
    topic = request.args.get('topic')
    subtopic = request.args.get('subtopic')
    query = request.args.get('q')
    
    # Build base query
    knowledge_query = KnowledgeItem.query
    
    # Apply filters
    if topic:
        knowledge_query = knowledge_query.filter_by(topic=topic)
    
    if subtopic:
        knowledge_query = knowledge_query.filter_by(subtopic=subtopic)
    
    if query:
        knowledge_query = knowledge_query.filter(
            KnowledgeItem.content.ilike(f'%{query}%')
        )
    
    # Execute query
    items = knowledge_query.all()
    
    # Format response
    result = []
    for item in items:
        result.append({
            'id': item.id,
            'topic': item.topic,
            'subtopic': item.subtopic,
            'content': item.content,
            'source': item.source,
            'metadata': item.metadata_dict,
            'created_at': item.created_at.isoformat(),
            'updated_at': item.updated_at.isoformat()
        })
    
    return jsonify(result)

@admin_bp.route('/knowledge', methods=['POST'])
@admin_required
def add_knowledge_item():
    """Add a new knowledge base item"""
    data = request.json or {}
    
    # Validate required fields
    required_fields = ['topic', 'subtopic', 'content']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing required field: {field}'}), 400
    
    # Create new knowledge item
    item = KnowledgeItem(
        topic=data['topic'],
        subtopic=data['subtopic'],
        content=data['content'],
        source=data.get('source'),
        metadata=data.get('metadata', {})
    )
    
    # Save to database
    db.session.add(item)
    db.session.commit()
    
    return jsonify({
        'message': 'Knowledge item added successfully',
        'id': item.id,
        'topic': item.topic,
        'subtopic': item.subtopic
    }), 201

@admin_bp.route('/knowledge/<int:item_id>', methods=['GET'])
@admin_required
def get_knowledge_item(item_id):
    """Get a specific knowledge base item"""
    item = KnowledgeItem.query.get(item_id)
    
    if not item:
        return jsonify({'error': 'Knowledge item not found'}), 404
    
    return jsonify({
        'id': item.id,
        'topic': item.topic,
        'subtopic': item.subtopic,
        'content': item.content,
        'source': item.source,
        'metadata': item.metadata_dict,
        'created_at': item.created_at.isoformat(),
        'updated_at': item.updated_at.isoformat()
    })

@admin_bp.route('/knowledge/<int:item_id>', methods=['PUT'])
@admin_required
def update_knowledge_item(item_id):
    """Update a knowledge base item"""
    item = KnowledgeItem.query.get(item_id)
    
    if not item:
        return jsonify({'error': 'Knowledge item not found'}), 404
    
    data = request.json or {}
    
    # Update fields
    if 'topic' in data:
        item.topic = data['topic']
    
    if 'subtopic' in data:
        item.subtopic = data['subtopic']
    
    if 'content' in data:
        item.content = data['content']
    
    if 'source' in data:
        item.source = data['source']
    
    if 'metadata' in data:
        item.metadata_dict = data['metadata']
    
    # Save changes
    db.session.commit()
    
    return jsonify({
        'message': 'Knowledge item updated successfully',
        'id': item.id
    })

@admin_bp.route('/knowledge/<int:item_id>', methods=['DELETE'])
@admin_required
def delete_knowledge_item(item_id):
    """Delete a knowledge base item"""
    item = KnowledgeItem.query.get(item_id)
    
    if not item:
        return jsonify({'error': 'Knowledge item not found'}), 404
    
    # Delete item
    db.session.delete(item)
    db.session.commit()
    
    return jsonify({
        'message': 'Knowledge item deleted successfully'
    })

@admin_bp.route('/dashboard', methods=['GET'])
@admin_required
def get_dashboard_data():
    """Get admin dashboard data with system metrics"""
    # Get counts
    user_count = User.query.count()
    active_user_count = User.query.filter_by(is_active=True).count()
    
    # Aggregate chat session data
    total_sessions = ChatSession.query.count()
    active_sessions = ChatSession.query.filter_by(is_active=True).count()
    
    # Calculate messages per session (average)
    from sqlalchemy import func
    from app.models.chat_message import ChatMessage
    
    message_counts = db.session.query(
        ChatMessage.session_id,
        func.count(ChatMessage.id).label('message_count')
    ).group_by(ChatMessage.session_id).all()
    
    avg_messages_per_session = 0
    if message_counts:
        avg_messages_per_session = sum(count for _, count in message_counts) / len(message_counts)
    
    # Return dashboard data
    return jsonify({
        'users': {
            'total': user_count,
            'active': active_user_count
        },
        'chat_sessions': {
            'total': total_sessions,
            'active': active_sessions
        },
        'messages': {
            'avg_per_session': round(avg_messages_per_session, 2)
        },
        'knowledge_base': {
            'items': KnowledgeItem.query.count()
        }
    })

@admin_bp.route('/users', methods=['GET'])
@admin_required
def get_users():
    """Get all users with filtering options"""
    # Get query parameters
    query = request.args.get('q')
    is_active = request.args.get('active')
    
    # Build base query
    user_query = User.query
    
    # Apply filters
    if query:
        user_query = user_query.filter(
            (User.username.ilike(f'%{query}%')) |
            (User.email.ilike(f'%{query}%')) |
            (User.full_name.ilike(f'%{query}%'))
        )
    
    if is_active is not None:
        is_active_bool = is_active.lower() == 'true'
        user_query = user_query.filter_by(is_active=is_active_bool)
    
    # Execute query
    users = user_query.all()
    
    # Format response
    result = []
    for user in users:
        result.append({
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'full_name': user.full_name,
            'created_at': user.created_at.isoformat(),
            'last_login': user.last_login.isoformat() if user.last_login else None,
            'is_active': user.is_active
        })
    
    return jsonify(result)

@admin_bp.route('/users/<int:user_id>', methods=['PUT'])
@admin_required
def update_user(user_id):
    """Update a user (admin action)"""
    user = User.query.get(user_id)
    
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    data = request.json or {}
    
    # Admin can update active status
    if 'is_active' in data:
        user.is_active = data['is_active']
    
    # Admin can reset password
    if 'reset_password' in data and data['reset_password']:
        new_password = data.get('new_password', 'changeme')
        user.set_password(new_password)
    
    # Save changes
    db.session.commit()
    
    return jsonify({
        'message': 'User updated successfully',
        'id': user.id
    })

@admin_bp.route('/crop-guide', methods=['POST'])
@jwt_required()
def add_crop_guide():
    """Add a new crop guide"""
    if not verify_admin_access():
        return jsonify({'error': 'Admin access required'}), 403
    
    data = request.get_json()
    
    # Validate required fields
    if 'crop_name' not in data:
        return jsonify({'error': 'Missing required field: crop_name'}), 400
    
    # Create crop guide
    guide = CropGuide(
        crop_name=data['crop_name'],
        scientific_name=data.get('scientific_name', ''),
        crop_family=data.get('crop_family', ''),
        crop_type=data.get('crop_type', ''),
        soil_requirements=data.get('soil_requirements', ''),
        water_requirements=data.get('water_requirements', ''),
        temperature_range=data.get('temperature_range', ''),
        climate_requirements=data.get('climate_requirements', ''),
        sowing_season=data.get('sowing_season', ''),
        harvesting_season=data.get('harvesting_season', ''),
        seed_rate=data.get('seed_rate', ''),
        spacing=data.get('spacing', ''),
        fertilizer_requirements=data.get('fertilizer_requirements', ''),
        pest_management=data.get('pest_management', ''),
        disease_management=data.get('disease_management', ''),
        irrigation_schedule=data.get('irrigation_schedule', ''),
        average_yield=data.get('average_yield', ''),
        market_information=data.get('market_information', ''),
        region_specific=data.get('region_specific', '')
    )
    
    db.session.add(guide)
    db.session.commit()
    
    return jsonify({
        'message': 'Crop guide added',
        'guide_id': guide.id
    }), 201 