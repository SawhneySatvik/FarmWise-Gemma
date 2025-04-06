from datetime import datetime
import json
from app.services.database import db

class ChatSession(db.Model):
    """Model for storing chat sessions between users and the AI system"""
    
    __tablename__ = 'chat_sessions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    session_name = db.Column(db.String(100))  # Optional session name
    started_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    
    # Define relationship with User
    user = db.relationship('User')
    
    # Define relationship with Messages
    messages = db.relationship('ChatMessage', backref='session', lazy=True, cascade='all, delete-orphan')
    
    def __init__(self, user_id, session_name=None):
        """Initialize chat session"""
        self.user_id = user_id
        self.session_name = session_name or f"Session {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}"
    
    def add_message(self, content, role, metadata=None):
        """
        Add a message to the chat session
        
        Args:
            content (str): Message content
            role (str): Message role (user or assistant)
            metadata (dict, optional): Additional metadata
            
        Returns:
            ChatMessage: The created message
        """
        message = ChatMessage(
            session_id=self.id,
            content=content,
            role=role,
            message_metadata=metadata
        )
        db.session.add(message)
        self.updated_at = datetime.utcnow()
        db.session.commit()
        return message
    
    def get_messages(self, limit=None, include_metadata=False):
        """
        Get messages in the chat session
        
        Args:
            limit (int, optional): Maximum number of messages to return
            include_metadata (bool): Whether to include metadata
            
        Returns:
            list: Chat messages
        """
        query = ChatMessage.query.filter_by(session_id=self.id).order_by(ChatMessage.created_at)
        
        if limit:
            query = query.limit(limit)
            
        messages = query.all()
        
        if include_metadata:
            return [
                {
                    "content": message.content,
                    "role": message.role,
                    "created_at": message.created_at,
                    "metadata": message.metadata_dict
                }
                for message in messages
            ]
        else:
            return [
                {
                    "content": message.content,
                    "role": message.role,
                    "created_at": message.created_at
                }
                for message in messages
            ]
    
    def close(self):
        """Close this chat session"""
        self.is_active = False
        self.updated_at = datetime.utcnow()
        db.session.commit()
    
    @classmethod
    def get_user_sessions(cls, user_id, active_only=False, limit=10):
        """
        Get chat sessions for a user
        
        Args:
            user_id (int): User ID
            active_only (bool): Only return active sessions
            limit (int): Maximum number of sessions to return
            
        Returns:
            list: Chat sessions
        """
        query = cls.query.filter_by(user_id=user_id)
        
        if active_only:
            query = query.filter_by(is_active=True)
            
        return query.order_by(cls.updated_at.desc()).limit(limit).all()
    
    def __repr__(self):
        """String representation of the chat session"""
        return f"<ChatSession {self.id} for user {self.user_id}>"


class ChatMessage(db.Model):
    """Model for storing individual chat messages"""
    
    __tablename__ = 'chat_messages'
    
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('chat_sessions.id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    role = db.Column(db.String(20), nullable=False)  # 'user' or 'assistant'
    message_metadata = db.Column(db.Text)  # JSON metadata (e.g., agent used, confidence, etc.)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __init__(self, session_id, content, role, message_metadata=None):
        """Initialize chat message"""
        self.session_id = session_id
        self.content = content
        self.role = role
        self.message_metadata = json.dumps(message_metadata or {})
    
    @property
    def metadata_dict(self):
        """Return metadata as a dictionary"""
        try:
            return json.loads(self.message_metadata)
        except (TypeError, json.JSONDecodeError):
            return {}
    
    @metadata_dict.setter
    def metadata_dict(self, value):
        """Set metadata from a dictionary"""
        self.message_metadata = json.dumps(value or {})
    
    def __repr__(self):
        """String representation of the chat message"""
        return f"<ChatMessage {self.id} ({self.role})>" 