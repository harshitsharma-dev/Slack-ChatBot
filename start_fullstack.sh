#!/bin/bash

echo "Building React app and starting full-stack server on localhost:5000..."

echo
echo "Step 1: Building React app..."
npm run build

if [ $? -ne 0 ]; then
    echo "Error: React build failed!"
    exit 1
fi

echo
echo "Step 2: Starting Flask server with React integration..."
echo "Both frontend and backend will be available at http://localhost:5000"
echo

python app.py