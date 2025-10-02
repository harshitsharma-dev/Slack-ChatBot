# Slack AI Data Bot MVP

A Slack bot that can answer questions about data stored in a database using AI/LLM capabilities, built with Python Flask backend and React TypeScript frontend.

## 🚀 Features

- **Natural Language Queries**: Ask questions about your data in plain English
- **AI-Powered SQL Generation**: Converts natural language to SQL queries using LLM
- **Slack Integration**: Seamless bot interaction within Slack channels
- **Web Dashboard**: Monitor bot activity and query history
- **Real-time Responses**: Fast and accurate data retrieval

## 🛠 Tech Stack

### Backend
- **Python Flask** - Web framework
- **PostgreSQL** - Database
- **SQLAlchemy** - ORM
- **Slack SDK** - Slack integration
- **Google Gemini** - AI/LLM processing via LangChain

### Frontend
- **React 18** - UI framework
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **Axios** - HTTP client
- **React Query** - Data fetching

### Deployment
- **Render** - Cloud platform for both backend and frontend

## 📋 Prerequisites

- Python 3.9+
- Node.js 16+
- PostgreSQL database
- Slack workspace with admin access
- Google API key for Gemini

## 🔧 Installation & Setup

### Quick Setup (Recommended)
```bash
# Run the automated setup script
python setup.py
```

### Manual Setup

#### 1. Backend Setup
```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
copy .env.example .env  # On Windows
# cp .env.example .env  # On Linux/Mac
# Edit .env with your configuration
```

#### 2. Frontend Setup
```bash
# Install dependencies
npm install

# Environment variables are in .env.local (already created)
```

#### 3. Database Setup
```bash
# Create database using the provided schema
# Run: database_schema.sql in your PostgreSQL database
# Or use the seed_data.sql for sample data
```

## 🔑 Environment Variables

### Backend (.env)
```env
# Database
DATABASE_URL=postgresql://username:password@host:port/database

# Slack
SLACK_BOT_TOKEN=xoxb-your-bot-token
SLACK_SIGNING_SECRET=your-signing-secret

# AI/LLM
GOOGLE_API_KEY=your-google-api-key

# Flask
FLASK_ENV=development
SECRET_KEY=your-secret-key
```

### Frontend (.env.local)
```env
REACT_APP_API_URL=http://localhost:5000
REACT_APP_ENVIRONMENT=development
```

## 🚀 Running the Application

### Development Mode

**Quick Start (Windows):**
```bash
# Start both servers at once
start_dev.bat
```

**Manual Start:**

**Backend:**
```bash
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On Linux/Mac
python app.py
# Server runs on http://localhost:5000
```

**Frontend:**
```bash
npm start
# App runs on http://localhost:3000
```

### Production Mode

**Backend:**
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

**Frontend:**
```bash
npm run build
# Serve the build folder
```

## 📱 Slack App Configuration

1. **Create Slack App**
   - Go to [Slack API](https://api.slack.com/apps)
   - Create new app from scratch
   - Choose your workspace

2. **Configure Bot Permissions**
   - Go to OAuth & Permissions
   - Add scopes: `chat:write`, `app_mentions:read`, `channels:history`

3. **Set Up Event Subscriptions**
   - Enable events
   - Set Request URL: `https://your-backend-url.onrender.com/slack/events`
   - Subscribe to: `app_mention`, `message.channels`

4. **Install App**
   - Install app to workspace
   - Copy Bot User OAuth Token

## 🗄 Database Schema

```sql
-- Users table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Products table
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    category VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Orders table
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    product_id INTEGER REFERENCES products(id),
    quantity INTEGER NOT NULL,
    total_amount DECIMAL(10,2) NOT NULL,
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## 🔄 API Endpoints

- `POST /slack/events` - Slack event webhook
- `GET /api/queries` - Get recent queries
- `POST /api/query` - Manual query testing
- `GET /api/health` - Health check
- `GET /api/stats` - Bot statistics

## 📊 Usage Examples

**Slack Commands:**
- "How many orders were placed today?"
- "What are the top selling products?"
- "Show me users who joined this month"
- "What's the total revenue for electronics category?"

## 🚀 Deployment on Render

### Backend Deployment
1. Connect GitHub repository
2. Choose "Web Service"
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `gunicorn -w 4 -b 0.0.0.0:$PORT app:app`
5. Add environment variables

### Frontend Deployment
1. Connect GitHub repository
2. Choose "Static Site"
3. Set build command: `npm run build`
4. Set publish directory: `build`
5. Add environment variables

### Database Setup
1. Create PostgreSQL database on Render
2. Update DATABASE_URL in backend environment variables

## 🧪 Testing

```bash
# Backend tests
cd backend
pytest

# Frontend tests
cd frontend
npm test
```

## 📝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support, email your-email@example.com or create an issue in the repository.

## 🔮 Roadmap

- [ ] Advanced query optimization
- [ ] Data visualization in responses
- [ ] Multi-workspace support
- [ ] Custom AI model training
- [ ] Analytics dashboard
- [ ] User authentication
- [ ] Query caching
- [ ] Webhook integrations

---

Built with ❤️ for efficient data querying through Slack# Slack-ChatBot
