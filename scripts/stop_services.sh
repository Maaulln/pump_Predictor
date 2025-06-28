#!/bin/bash

# Service Stop Script
echo "🛑 Stopping Pump Maintenance Predictor Services..."

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Stop Docker services if running
if [ -f "deployment/docker/docker-compose.yml" ]; then
    if docker-compose -f deployment/docker/docker-compose.yml ps | grep -q "Up"; then
        print_status "Stopping Docker services..."
        docker-compose -f deployment/docker/docker-compose.yml down
        print_status "Docker services stopped ✓"
        exit 0
    fi
fi

# Stop services using saved PIDs
if [ -f ".service_pids" ]; then
    print_status "Stopping services using saved PIDs..."
    
    while read -r pid; do
        if ps -p $pid > /dev/null 2>&1; then
            kill $pid
            print_status "Stopped process: $pid ✓"
        else
            print_warning "Process $pid not found"
        fi
    done < .service_pids
    
    rm -f .service_pids
fi

# Force stop by process name
print_status "Force stopping any remaining processes..."

# Stop API processes
api_pids=$(pgrep -f "pump_predictor.api.main")
if [ ! -z "$api_pids" ]; then
    echo $api_pids | xargs kill
    print_status "API processes stopped ✓"
fi

# Stop Dashboard processes
dashboard_pids=$(pgrep -f "streamlit run")
if [ ! -z "$dashboard_pids" ]; then
    echo $dashboard_pids | xargs kill
    print_status "Dashboard processes stopped ✓"
fi

# Wait a moment for graceful shutdown
sleep 2

# Force kill if still running
api_pids=$(pgrep -f "pump_predictor.api.main")
if [ ! -z "$api_pids" ]; then
    echo $api_pids | xargs kill -9
    print_warning "Force killed API processes"
fi

dashboard_pids=$(pgrep -f "streamlit run")
if [ ! -z "$dashboard_pids" ]; then
    echo $dashboard_pids | xargs kill -9
    print_warning "Force killed Dashboard processes"
fi

print_status "All services stopped ✓"