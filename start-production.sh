#!/bin/bash

# Two-Wheel Bot Production Start Script

echo "Starting Two-Wheel Balancing Robot RL Application..."

# Navigate to application directory
cd /home/orencollaco/GitHub/twowheelbot-rl-web

# Pull latest changes (optional - comment out if you don't want auto-updates)
# git pull origin master

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

# Build the application
echo "Building application..."
npm run build

# Kill any existing process on port 3005
lsof -ti:3005 | xargs kill -9 2>/dev/null

# Start the server
echo "Starting server on port 3005..."
npx serve -s dist -l 3005