# FarmWise: AI-Powered Smart Farming System

FarmWise is an innovative, AI-driven platform designed to revolutionize farming practices, with a particular focus on empowering Indian farmers. By integrating advanced artificial intelligence, machine learning, and natural language processing, the system provides real-time insights, predictive analytics, and personalized recommendations to optimize crop yields, reduce losses, and promote sustainable agriculture.

## Architecture

The system is built around a hierarchical, agent-based AI platform that orchestrates specialized agents to deliver end-to-end farm management solutions:

- **FarmingAdvisor**: Top-level coordinator that processes farmer queries
- **Specialized Agents**:
  - **CropAgent**: Recommends crops and predicts yields
  - **SoilAgent**: Analyzes soil health and suggests amendments
  - **WeatherAgent**: Provides weather forecasts and impact analysis
  - **IrrigationAgent**: Plans irrigation based on weather and soil data
  - **PestDiseaseAgent**: Assesses pest and disease risks
  - **MarketAgent**: Analyzes price trends and market conditions
  - **FeedAgent**: Optimizes livestock feed and nutrition plans

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/farmwise.git
   cd farmwise/server
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set environment variables (create a `.env` file):

   ```
   SECRET_KEY=your_secret_key
   JWT_SECRET_KEY=your_jwt_secret
   FLASK_ENV=development
   GEMINI_API_KEY=your_gemini_api_key  # If using Google's Gemini API
   ```

5. Initialize the database and create sample data:

   ```bash
   python setup.py --all
   ```

   This will:

   - Create the database tables
   - Add an admin user (username: admin, password: adminpass)
   - Add sample knowledge base entries
   - Create a test farmer account with sample farm data

6. Run the application:
   ```bash
   python run.py
   ```

The server will start at http://localhost:5000 by default.

## API Endpoints

### Authentication

- **POST /api/auth/register**: Register a new user
- **POST /api/auth/login**: Login and get access token
- **GET /api/auth/profile**: Get user profile
- **PUT /api/auth/profile**: Update user profile

### Chat

- **POST /api/chat/session**: Create a new chat session
- **GET /api/chat/sessions**: Get user's chat sessions
- **GET /api/chat/session/{id}**: Get a specific chat session
- **POST /api/chat/session/{id}/message**: Send a message
- **GET /api/chat/session/{id}/messages**: Get messages for a session

### Admin

- **GET /api/admin/users**: Get all users (admin only)
- **GET /api/admin/knowledge**: Get all knowledge base items
- **POST /api/admin/knowledge**: Add a knowledge base item
- **PUT /api/admin/knowledge/{id}**: Update a knowledge base item

## Development

### Project Structure

The server follows a modular structure:

```
server/
├── app/
│   ├── agents/            # AI agents
│   ├── models/            # Database models
│   ├── routes/            # API endpoints
│   ├── services/          # Services (LLM, weather, etc.)
│   └── database/          # Database files
├── setup.py               # Database setup script
└── run.py                 # Application entry point
```

### Database Setup Options

The `setup.py` script provides several options:

```bash
# View help
python setup.py

# Initialize just the database
python setup.py --init-db

# Create admin user
python setup.py --create-admin

# Add sample data
python setup.py --sample-data

# Perform all setup steps
python setup.py --all
```

### LLM Integration

The system integrates with Language Learning Models for natural language processing. Currently supported:

- Google's Gemini (default)
- Support for other models can be added in the LLM service

### Knowledge Base

The system uses a Retrieval-Augmented Generation (RAG) approach, storing domain knowledge in the database:

- Agricultural best practices
- Crop and livestock information
- Regional farming insights

## Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add some amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request
