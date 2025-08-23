#!/bin/bash

# Two-Wheel Bot Service Installation Script
# Run with: sudo ./install-service.sh

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SERVICE_NAME="twowheelbot"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
APP_DIR="/home/orencollaco/GitHub/twowheelbot-rl-web"
LOG_DIR="/var/log"
PORT=3005

# Function to print colored messages
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   print_error "This script must be run as root (use sudo)"
   exit 1
fi

echo "========================================="
echo "Two-Wheel Bot Service Installation"
echo "========================================="
echo ""

# Step 1: Check if application directory exists
print_status "Checking application directory..."
if [ ! -d "$APP_DIR" ]; then
    print_error "Application directory not found: $APP_DIR"
    exit 1
fi
print_status "Application directory found"

# Step 2: Check if Node.js is installed
print_status "Checking Node.js installation..."
if ! command -v node &> /dev/null; then
    print_error "Node.js is not installed. Please install Node.js first."
    exit 1
fi
NODE_VERSION=$(node -v)
print_status "Node.js $NODE_VERSION found"

# Step 3: Install dependencies and build
print_status "Installing dependencies and building application..."
cd "$APP_DIR"
sudo -u orencollaco npm install
if [ $? -ne 0 ]; then
    print_error "Failed to install dependencies"
    exit 1
fi

sudo -u orencollaco npm run build
if [ $? -ne 0 ]; then
    print_error "Failed to build application"
    exit 1
fi
print_status "Application built successfully"

# Step 4: Create log files with proper permissions
print_status "Creating log files..."
touch "${LOG_DIR}/${SERVICE_NAME}.log"
touch "${LOG_DIR}/${SERVICE_NAME}.error.log"
chown orencollaco:orencollaco "${LOG_DIR}/${SERVICE_NAME}.log"
chown orencollaco:orencollaco "${LOG_DIR}/${SERVICE_NAME}.error.log"
print_status "Log files created"

# Step 5: Create systemd service file
print_status "Creating systemd service file..."
cat > "$SERVICE_FILE" << EOF
[Unit]
Description=Two-Wheel Balancing Robot RL Web Application
After=network.target

[Service]
Type=simple
User=orencollaco
Group=orencollaco
WorkingDirectory=$APP_DIR

# Use npx to run serve
ExecStart=/usr/bin/npx serve -s ${APP_DIR}/dist -l ${PORT} --no-clipboard

# Restart configuration
Restart=always
RestartSec=10
StartLimitInterval=200
StartLimitBurst=5

# Logging
StandardOutput=append:${LOG_DIR}/${SERVICE_NAME}.log
StandardError=append:${LOG_DIR}/${SERVICE_NAME}.error.log

# Environment
Environment="NODE_ENV=production"
Environment="PATH=/usr/bin:/usr/local/bin:/home/orencollaco/.nvm/versions/node/v20.11.1/bin"

# Security settings
PrivateTmp=true
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target
EOF

if [ $? -ne 0 ]; then
    print_error "Failed to create service file"
    exit 1
fi
print_status "Service file created at $SERVICE_FILE"

# Step 6: Reload systemd daemon
print_status "Reloading systemd daemon..."
systemctl daemon-reload
if [ $? -ne 0 ]; then
    print_error "Failed to reload systemd daemon"
    exit 1
fi
print_status "Systemd daemon reloaded"

# Step 7: Stop any existing service
print_status "Checking for existing service..."
if systemctl is-active --quiet "$SERVICE_NAME"; then
    print_warning "Service is running, stopping it..."
    systemctl stop "$SERVICE_NAME"
fi

# Step 8: Kill any process on the port
print_status "Checking port $PORT..."
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    print_warning "Port $PORT is in use, killing existing process..."
    lsof -ti:$PORT | xargs kill -9 2>/dev/null || true
    sleep 2
fi

# Step 9: Enable and start the service
print_status "Enabling service to start on boot..."
systemctl enable "$SERVICE_NAME"
if [ $? -ne 0 ]; then
    print_error "Failed to enable service"
    exit 1
fi
print_status "Service enabled"

print_status "Starting service..."
systemctl start "$SERVICE_NAME"
if [ $? -ne 0 ]; then
    print_error "Failed to start service"
    systemctl status "$SERVICE_NAME" --no-pager
    exit 1
fi
print_status "Service started"

# Step 10: Wait for service to initialize
print_status "Waiting for service to initialize..."
sleep 5

# Step 11: Verify service is running
print_status "Verifying service status..."
if ! systemctl is-active --quiet "$SERVICE_NAME"; then
    print_error "Service is not running!"
    systemctl status "$SERVICE_NAME" --no-pager
    exit 1
fi
print_status "Service is active"

# Step 12: Test HTTP connection
print_status "Testing HTTP connection..."
MAX_RETRIES=10
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s -o /dev/null -w "%{http_code}" "http://localhost:${PORT}" | grep -q "200\|304"; then
        print_status "HTTP server is responding"
        break
    else
        RETRY_COUNT=$((RETRY_COUNT + 1))
        if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
            print_error "HTTP server is not responding after $MAX_RETRIES attempts"
            print_warning "Checking service logs..."
            journalctl -u "$SERVICE_NAME" --no-pager -n 20
            exit 1
        fi
        print_warning "Waiting for HTTP server... (attempt $RETRY_COUNT/$MAX_RETRIES)"
        sleep 2
    fi
done

# Step 13: Display service information
echo ""
echo "========================================="
echo -e "${GREEN}Installation Complete!${NC}"
echo "========================================="
echo ""
echo "Service Information:"
echo "  Name: $SERVICE_NAME"
echo "  Status: $(systemctl is-active $SERVICE_NAME)"
echo "  Port: $PORT"
echo "  URL: http://localhost:${PORT}"
echo ""
echo "Useful Commands:"
echo "  View status:  sudo systemctl status $SERVICE_NAME"
echo "  View logs:    sudo journalctl -u $SERVICE_NAME -f"
echo "  Restart:      sudo systemctl restart $SERVICE_NAME"
echo "  Stop:         sudo systemctl stop $SERVICE_NAME"
echo "  Disable:      sudo systemctl disable $SERVICE_NAME"
echo ""
echo "Log Files:"
echo "  ${LOG_DIR}/${SERVICE_NAME}.log"
echo "  ${LOG_DIR}/${SERVICE_NAME}.error.log"
echo ""

# Step 14: Final verification
print_status "Running final verification..."

# Check if process is running
if pgrep -f "serve.*${PORT}" > /dev/null; then
    print_status "Process is running"
else
    print_error "Process not found!"
    exit 1
fi

# Check if port is listening
if netstat -tuln | grep -q ":${PORT}"; then
    print_status "Port $PORT is listening"
else
    print_error "Port $PORT is not listening!"
    exit 1
fi

# Final success message
echo ""
echo -e "${GREEN}✓ Service successfully installed and verified!${NC}"
echo -e "${GREEN}✓ Application is now accessible at http://localhost:${PORT}${NC}"
echo ""

exit 0