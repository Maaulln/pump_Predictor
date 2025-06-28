#!/bin/bash

# Service Startup Script
echo "🚀 Starting Pump Maintenance Predictor Services..."

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_header() {
    echo -e "${BLUE}$1${NC}"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse arguments
START_API=true
START_DASHBOARD=true
START_DOCKER=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --api-only)
            START_DASHBOARD=false
            shift
            ;;
        --dashboard-only)
            START_API=false
            shift
            ;;
        --docker)
            START_DOCKER=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --api-only       Start only the API service"
            echo "  --dashboard-only Start only the dashboard"
            echo "  --docker         Use Docker Compose"
            echo "  --help           Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Function to check if port is available
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null; then
        return 1
    else
        return 0
    fi
}

# Function to wait for service
wait_for_service() {
    local url=$1
    local service_name=$2
    local max_attempts=30
    local attempt=1
    
    print_status "Waiting for $service_name to start..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$url" > /dev/null 2>&1; then
            print_status "$service_name is ready! ✓"
            return 0
        fi
        
        echo -n "."
        sleep 2
        ((attempt++))
    done
    
    print_error "$service_name failed to start within expected time"
    return 1
}

# Docker mode
if [ "$START_DOCKER" = true ]; then
    print_header "🐳 Starting with Docker Compose..."
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "docker-compose not found. Please install Docker Compose."
        exit 1
    fi
    
    cd deployment/docker
    docker-compose up -d
    
    print_status "Services started with Docker Compose"
    print_status "API: http://localhost:8000"
    print_status "Dashboard: http://localhost:8501"
    print_status "API Docs: http://localhost:8000/docs"
    
    exit 0
fi

# Check virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    print_warning "Virtual environment not detected. Activating..."
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    else
        print_error "Virtual environment not found. Run setup.sh first."
        exit 1
    fi
fi

# Check if models exist
if [ ! -f "models/best_model.joblib" ]; then
    print_warning "Trained models not found. Please train models first:"
    echo "  ./scripts/train_models.sh"
    exit 1
fi

# Start API
if [ "$START_API" = true ]; then
    print_header "🔧 Starting API Service..."
    
    if ! check_port 8000; then
        print_warning "Port 8000 is already in use. Trying to stop existing service..."
        pkill -f "pump_predictor.api.main"
        sleep 2
    fi
    
    # Start API in background
    nohup python -m pump_predictor.api.main > logs/api.log 2>&1 &
    API_PID=$!
    
    print_status "API service started (PID: $API_PID)"
    
    # Wait for API to be ready
    if wait_for_service "http://localhost:8000/health" "API"; then
        print_status "API available at: http://localhost:8000"
        print_status "API Documentation: http://localhost:8000/docs"
    else
        print_error "Failed to start API service"
        exit 1
    fi
fi

# Start Dashboard
if [ "$START_DASHBOARD" = true ]; then
    print_header "📊 Starting Dashboard..."
    
    if ! check_port 8501; then
        print_warning "Port 8501 is already in use. Trying to stop existing service..."
        pkill -f "streamlit run"
        sleep 2
    fi
    
    # Start dashboard in background
    nohup streamlit run dashboard/streamlit_app.py \
        --server.port 8501 \
        --server.address 0.0.0.0 \
        --server.headless true \
        > logs/dashboard.log 2>&1 &
    DASHBOARD_PID=$!
    
    print_status "Dashboard service started (PID: $DASHBOARD_PID)"
    
    # Wait for dashboard to be ready
    if wait_for_service "http://localhost:8501" "Dashboard"; then
        print_status "Dashboard available at: http://localhost:8501"
    else
        print_error "Failed to start Dashboard service"
        exit 1
    fi
fi

# Save PIDs for easy stopping
if [ "$START_API" = true ] && [ "$START_DASHBOARD" = true ]; then
    echo "$API_PID $DASHBOARD_PID" > .service_pids
elif [ "$START_API" = true ]; then
    echo "$API_PID" > .service_pids
elif [ "$START_DASHBOARD" = true ]; then
    echo "$DASHBOARD_PID" > .service_pids
fi

# Final status
echo
print_header "🎉 Services Started Successfully!"
echo
if [ "$START_API" = true ]; then
    print_status "🔧 API Service: http://localhost:8000"
    print_status "📚 API Documentation: http://localhost:8000/docs"
fi

if [ "$START_DASHBOARD" = true ]; then
    print_status "📊 Dashboard: http://localhost:8501"
fi

echo
print_status "To stop services: ./scripts/stop_services.sh"
print_status "View logs: tail -f logs/api.log logs/dashboard.log"