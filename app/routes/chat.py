from flask import Blueprint, request, jsonify, current_app, g
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.services.database import db
from app.models.chat_session import ChatSession, ChatMessage
from app.models.user import User
from app.models.farm_data import FarmData
from app.services.llm_service import LLMService
from app.agents.farming_advisor import FarmingAdvisor
from datetime import datetime

chat_bp = Blueprint('chat', __name__, url_prefix='/api/chat')

# Initialize services
llm_service = LLMService()
farming_advisor = FarmingAdvisor(llm_service)

@chat_bp.route('/sessions', methods=['GET'])
@jwt_required()
def get_sessions():
    """Get all chat sessions for the current user"""
    current_user_id = get_jwt_identity()
    
    # Get parameter for active_only
    active_only = request.args.get('active', 'false').lower() == 'true'
    
    # Get limit parameter with default of 10
    try:
        limit = int(request.args.get('limit', 10))
    except ValueError:
        limit = 10
    
    # Get sessions
    sessions = ChatSession.get_user_sessions(current_user_id, active_only, limit)
    
    # Format response
    result = []
    for session in sessions:
        result.append({
            'id': session.id,
            'name': session.session_name,
            'started_at': session.started_at.isoformat(),
            'updated_at': session.updated_at.isoformat(),
            'is_active': session.is_active,
            'message_count': len(session.messages)
        })
    
    return jsonify(result)

@chat_bp.route('/sessions', methods=['POST'])
@jwt_required()
def create_session():
    """Create a new chat session"""
    current_user_id = get_jwt_identity()
    
    # Get session name from request body
    data = request.json or {}
    session_name = data.get('name')
    
    # Create new session
    session = ChatSession(current_user_id, session_name)
    db.session.add(session)
    db.session.commit()
    
    # Add welcome message
    welcome_message = session.add_message(
        "Hello! I'm your FarmWise farming assistant. How can I help you with your agricultural questions today?",
        "assistant"
    )
    
    return jsonify({
        'id': session.id,
        'name': session.session_name,
        'started_at': session.started_at.isoformat(),
        'updated_at': session.updated_at.isoformat(),
        'is_active': session.is_active,
        'welcome_message': {
            'id': welcome_message.id,
            'content': welcome_message.content,
            'role': welcome_message.role,
            'created_at': welcome_message.created_at.isoformat()
        }
    }), 201

@chat_bp.route('/sessions/<int:session_id>', methods=['GET'])
@jwt_required()
def get_session(session_id):
    """Get details of a specific chat session"""
    current_user_id = get_jwt_identity()
    
    # Get session
    session = ChatSession.query.filter_by(id=session_id, user_id=current_user_id).first()
    
    if not session:
        return jsonify({'error': 'Chat session not found'}), 404
    
    # Get messages
    include_metadata = request.args.get('include_metadata', 'false').lower() == 'true'
    messages = session.get_messages(include_metadata=include_metadata)
    
    # Format response
    result = {
        'id': session.id,
        'name': session.session_name,
        'started_at': session.started_at.isoformat(),
        'updated_at': session.updated_at.isoformat(),
        'is_active': session.is_active,
        'messages': messages
    }
    
    return jsonify(result)

@chat_bp.route('/sessions/<int:session_id>/messages', methods=['POST'])
@jwt_required()
def add_message(session_id):
    """Add a message to a chat session and get AI response"""
    current_user_id = get_jwt_identity()
    
    # Get session
    session = ChatSession.query.filter_by(id=session_id, user_id=current_user_id).first()
    
    if not session:
        return jsonify({'error': 'Chat session not found'}), 404
    
    if not session.is_active:
        return jsonify({'error': 'This chat session is closed'}), 400
    
    # Get message from request body
    data = request.json or {}
    content = data.get('content')
    
    if not content:
        return jsonify({'error': 'Message content is required'}), 400
    
    # Add user message to session
    user_message = session.add_message(content, 'user')
    
    # Get user context for better personalization
    user = User.query.get(current_user_id)
    user_context = user.get_farm_context() if user else {}
    
    # Process the query with the farming advisor
    advisor_response = farming_advisor.process(content, {'user_context': user_context})
    
    # Add AI response to session
    ai_message = session.add_message(
        advisor_response['content'],
        'assistant',
        {
            'agent_used': advisor_response.get('metadata', {}).get('agent_used', 'farming_advisor'),
            'confidence': advisor_response.get('confidence', 0.0),
            'intent': advisor_response.get('metadata', {}).get('intent', 'general')
        }
    )
    
    # Format response
    result = {
        'user_message': {
            'id': user_message.id,
            'content': user_message.content,
            'role': user_message.role,
            'created_at': user_message.created_at.isoformat()
        },
        'ai_message': {
            'id': ai_message.id,
            'content': ai_message.content,
            'role': ai_message.role,
            'created_at': ai_message.created_at.isoformat(),
            'metadata': ai_message.metadata_dict
        }
    }
    
    return jsonify(result)

@chat_bp.route('/sessions/<int:session_id>', methods=['PUT'])
@jwt_required()
def update_session(session_id):
    """Update a chat session (rename or close)"""
    current_user_id = get_jwt_identity()
    
    # Get session
    session = ChatSession.query.filter_by(id=session_id, user_id=current_user_id).first()
    
    if not session:
        return jsonify({'error': 'Chat session not found'}), 404
    
    # Get data from request body
    data = request.json or {}
    
    # Update session name if provided
    if 'name' in data:
        session.session_name = data['name']
    
    # Close session if specified
    if 'is_active' in data and data['is_active'] is False:
        session.is_active = False
    
    # Save changes
    db.session.commit()
    
    # Format response
    result = {
        'id': session.id,
        'name': session.session_name,
        'started_at': session.started_at.isoformat(),
        'updated_at': session.updated_at.isoformat(),
        'is_active': session.is_active
    }
    
    return jsonify(result)

@chat_bp.route('/sessions/<int:session_id>', methods=['DELETE'])
@jwt_required()
def delete_session(session_id):
    """Delete a chat session and all its messages"""
    current_user_id = get_jwt_identity()
    
    # Get session
    session = ChatSession.query.filter_by(id=session_id, user_id=current_user_id).first()
    
    if not session:
        return jsonify({'error': 'Chat session not found'}), 404
    
    # Delete session (and all its messages due to cascade)
    db.session.delete(session)
    db.session.commit()
    
    return jsonify({'message': 'Chat session deleted successfully'}), 200

@chat_bp.route('/query', methods=['POST'])
def direct_query():
    """Handle direct queries without creating a session (for testing/development)"""
    # Get query from request body
    data = request.json or {}
    query = data.get('query')
    
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    # Process the query with the farming advisor
    response = farming_advisor.process(query, {})
    
    return jsonify({
        'query': query,
        'response': response['content'],
        'agent_used': response.get('metadata', {}).get('agent_used', 'farming_advisor'),
        'confidence': response.get('confidence', 0.0),
        'intent': response.get('metadata', {}).get('intent', 'general')
    }) 