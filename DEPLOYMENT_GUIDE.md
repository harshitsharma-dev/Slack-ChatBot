# Deployment Guide - Slack AI Data Bot

## 🚀 Quick Deployment Checklist

### Prerequisites
- [ ] GitHub repository created and code pushed
- [ ] Render account created
- [ ] Slack workspace with admin access
- [ ] OpenAI API key obtained

### 1. Database Setup on Render

1. **Create PostgreSQL Database**
   - Go to Render Dashboard
   - Click "New" → "PostgreSQL"
   - Name: `slack-ai-bot-db`
   - Plan: Free (for development)
   - Click "Create Database"

2. **Note Database Details**
   - Copy the connection string (Internal Database URL)
   - Save for backend environment variables

### 2. Backend Deployment

1. **Create Web Service**
   - Go to Render Dashboard
   - Click "New" → "Web Service"
   - Connect your GitHub repository
   - Select the repository

2. **Configure Service**
   - **Name**: `slack-ai-bot-backend`
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn -w 4 -b 0.0.0.0:$PORT app:app`

3. **Environment Variables**
   ```
   DATABASE_URL=<your-postgres-connection-string>
   SLACK_BOT_TOKEN=<your-slack-bot-token>
   SLACK_SIGNING_SECRET=<your-slack-signing-secret>
   OPENAI_API_KEY=<your-openai-api-key>
   FLASK_ENV=production
   SECRET_KEY=<generate-random-secret-key>
   ```

4. **Deploy**
   - Click "Create Web Service"
   - Wait for deployment to complete
   - Note the service URL (e.g., `https://slack-ai-bot-backend.onrender.com`)

### 3. Database Schema Setup

1. **Connect to Database**
   ```bash
   # Using psql (install PostgreSQL client)
   psql <your-database-connection-string>
   ```

2. **Run Schema and Seed Scripts**
   ```sql
   -- Copy and paste contents of database_schema.sql
   -- Then copy and paste contents of seed_data.sql
   ```

### 4. Frontend Deployment

1. **Create Static Site**
   - Go to Render Dashboard
   - Click "New" → "Static Site"
   - Connect your GitHub repository
   - Select the repository

2. **Configure Static Site**
   - **Name**: `slack-ai-bot-frontend`
   - **Build Command**: `npm install && npm run build`
   - **Publish Directory**: `build`

3. **Environment Variables**
   ```
   REACT_APP_API_URL=https://slack-ai-bot-backend.onrender.com
   NODE_VERSION=18.17.0
   ```

4. **Deploy**
   - Click "Create Static Site"
   - Wait for deployment to complete

### 5. Slack App Configuration

1. **Create Slack App**
   - Go to [Slack API](https://api.slack.com/apps)
   - Click "Create New App"
   - Choose "From scratch"
   - App Name: "AI Data Bot"
   - Select your workspace

2. **Configure OAuth & Permissions**
   - Go to "OAuth & Permissions"
   - Add Bot Token Scopes:
     - `app_mentions:read`
     - `channels:history`
     - `chat:write`
     - `im:history`
     - `im:read`
     - `im:write`

3. **Install App to Workspace**
   - Click "Install to Workspace"
   - Authorize the app
   - Copy the "Bot User OAuth Token" (starts with `xoxb-`)

4. **Configure Event Subscriptions**
   - Go to "Event Subscriptions"
   - Enable Events: Toggle ON
   - Request URL: `https://slack-ai-bot-backend.onrender.com/slack/events`
   - Subscribe to bot events:
     - `app_mention`
     - `message.im`

5. **Get Signing Secret**
   - Go to "Basic Information"
   - Copy "Signing Secret"

### 6. Update Environment Variables

Update your backend service on Render with the Slack credentials:
- `SLACK_BOT_TOKEN`: Bot User OAuth Token
- `SLACK_SIGNING_SECRET`: Signing Secret

### 7. Test Deployment

1. **Health Check**
   ```bash
   curl https://slack-ai-bot-backend.onrender.com/api/health
   ```

2. **Test in Slack**
   - Invite the bot to a channel: `/invite @AI Data Bot`
   - Mention the bot: `@AI Data Bot How many orders were placed today?`

## 🔧 Troubleshooting

### Common Issues

1. **Database Connection Error**
   - Verify DATABASE_URL is correct
   - Check if database is running
   - Ensure schema is created

2. **Slack Events Not Received**
   - Verify Request URL in Slack app settings
   - Check SLACK_SIGNING_SECRET is correct
   - Ensure backend service is running

3. **OpenAI API Errors**
   - Verify OPENAI_API_KEY is valid
   - Check API quota and billing
   - Ensure correct model name

4. **Frontend Not Loading**
   - Check REACT_APP_API_URL points to backend
   - Verify CORS settings in backend
   - Check browser console for errors

### Logs and Monitoring

1. **Backend Logs**
   - Go to Render Dashboard → Your Backend Service → Logs
   - Monitor for errors and API calls

2. **Database Logs**
   - Go to Render Dashboard → Your Database → Logs
   - Check connection and query logs

3. **Slack App Logs**
   - Go to Slack API → Your App → Event Subscriptions
   - Check request logs and responses

## 📊 Performance Optimization

### Backend Optimization
- Enable connection pooling for database
- Implement query caching for common requests
- Add request rate limiting
- Use async processing for long-running queries

### Frontend Optimization
- Implement lazy loading for components
- Add service worker for caching
- Optimize bundle size with code splitting
- Use React.memo for expensive components

### Database Optimization
- Add appropriate indexes for common queries
- Implement query result caching
- Use database views for complex aggregations
- Monitor slow queries and optimize

## 🔒 Security Considerations

1. **Environment Variables**
   - Never commit secrets to version control
   - Use Render's environment variable encryption
   - Rotate API keys regularly

2. **API Security**
   - Implement request validation
   - Add rate limiting
   - Use HTTPS only
   - Validate Slack request signatures

3. **Database Security**
   - Use connection pooling with limits
   - Implement SQL injection prevention
   - Regular security updates
   - Monitor for unusual query patterns

## 📈 Monitoring and Maintenance

### Health Checks
- Set up uptime monitoring (e.g., UptimeRobot)
- Monitor API response times
- Track error rates and patterns
- Set up alerts for critical failures

### Regular Maintenance
- Update dependencies monthly
- Monitor database performance
- Review and optimize slow queries
- Update AI model versions as needed

### Scaling Considerations
- Monitor resource usage on Render
- Consider upgrading to paid plans for production
- Implement horizontal scaling if needed
- Use CDN for frontend assets

## 🎯 Success Metrics

### Technical Metrics
- API response time < 3 seconds
- Database query time < 1 second
- 99.9% uptime
- Error rate < 1%

### Business Metrics
- Number of successful queries per day
- User engagement in Slack
- Query accuracy rate
- User satisfaction feedback

---

## 📞 Support

If you encounter issues during deployment:

1. Check the troubleshooting section above
2. Review Render service logs
3. Verify all environment variables
4. Test each component individually
5. Create an issue in the GitHub repository

**Remember**: Keep your API keys secure and never share them publicly!