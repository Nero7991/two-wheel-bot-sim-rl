#!/bin/bash

# Two-Wheel Bot Service Uninstallation Script
# Run with: sudo ./uninstall-service.sh

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SERVICE_NAME="twowheelbot"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
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
echo "Two-Wheel Bot Service Uninstallation"
echo "========================================="
echo ""

# Step 1: Stop the service
if systemctl is-active --quiet "$SERVICE_NAME"; then
    print_status "Stopping service..."
    systemctl stop "$SERVICE_NAME"
else
    print_warning "Service is not running"
fi

# Step 2: Disable the service
if systemctl is-enabled --quiet "$SERVICE_NAME"; then
    print_status "Disabling service..."
    systemctl disable "$SERVICE_NAME"
else
    print_warning "Service is not enabled"
fi

# Step 3: Remove service file
if [ -f "$SERVICE_FILE" ]; then
    print_status "Removing service file..."
    rm "$SERVICE_FILE"
else
    print_warning "Service file not found"
fi

# Step 4: Reload systemd daemon
print_status "Reloading systemd daemon..."
systemctl daemon-reload

# Step 5: Kill any remaining processes on the port
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    print_warning "Killing processes on port $PORT..."
    lsof -ti:$PORT | xargs kill -9 2>/dev/null || true
fi

# Step 6: Remove log files (optional)
read -p "Do you want to remove log files? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -f "${LOG_DIR}/${SERVICE_NAME}.log" ]; then
        rm "${LOG_DIR}/${SERVICE_NAME}.log"
        print_status "Removed log file"
    fi
    if [ -f "${LOG_DIR}/${SERVICE_NAME}.error.log" ]; then
        rm "${LOG_DIR}/${SERVICE_NAME}.error.log"
        print_status "Removed error log file"
    fi
fi

echo ""
echo -e "${GREEN}✓ Service successfully uninstalled!${NC}"
echo ""