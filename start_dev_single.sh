#!/bin/bash

echo "Starting development server with hot reload on localhost:5000..."

echo
echo "Building React app for development..."
npm run build

echo
echo "Starting Flask server with React integration..."
echo "Frontend: http://localhost:5000"
echo "Backend API: http://localhost:5000/api/*"
echo "Slack Webhook: http://localhost:5000/slack/events"
echo

python app.py