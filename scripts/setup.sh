#!/bin/bash

# Pump Maintenance Predictor Setup Script
echo "🔧 Setting up Pump Maintenance Predictor..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}$1${NC}"
}

# Check if Python 3.8+ is installed
print_header "🐍 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.8"

# Compare versions correctly using sort -V
if [[ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" == "$required_version" ]]; then
    print_status "Python $python_version detected ✓"
else
    print_error "Python 3.8+ required. Current version: $python_version"
    exit 1
fi

# Create virtual environment
print_header "🌐 Setting up virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_status "Virtual environment created ✓"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate
print_status "Virtual environment activated ✓"

# Upgrade pip
print_header "📦 Upgrading pip..."
pip install --upgrade pip

# Install requirements
print_header "📚 Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    print_status "Main dependencies installed ✓"
else
    print_error "requirements.txt not found!"
    exit 1
fi

# Install dashboard dependencies
if [ -f "dashboard/requirements.txt" ]; then
    pip install -r dashboard/requirements.txt
    print_status "Dashboard dependencies installed ✓"
fi

# Create necessary directories
print_header "📁 Creating directories..."
directories=("data" "models" "logs" "reports" "reports/plots" "reports/interactive_plots")

for dir in "${directories[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        print_status "Created directory: $dir ✓"
    else
        print_warning "Directory already exists: $dir"
    fi
done

# Set up pre-commit hooks (optional)
print_header "🔗 Setting up pre-commit hooks..."
if command -v pre-commit &> /dev/null; then
    pre-commit install
    print_status "Pre-commit hooks installed ✓"
else
    print_warning "pre-commit not found. Skipping..."
fi

# Generate sample data if not exists
print_header "📊 Setting up sample data..."
if [ ! -f "data/pump_data.csv" ]; then
    python3 -c "
import sys
sys.path.append('.')
from pump_predictor.data.preprocessing import DataPreprocessor
dp = DataPreprocessor()
sample_data = dp.create_synthetic_data(1000)
sample_data.to_csv('data/pump_data.csv', index=False)
print('Sample data generated: data/pump_data.csv')
"
    print_status "Sample data generated ✓"
else
    print_warning "Sample data already exists"
fi

# Run initial tests
print_header "🧪 Running tests..."
if command -v pytest &> /dev/null; then
    pytest tests/ -v --tb=short
    if [ $? -eq 0 ]; then
        print_status "All tests passed ✓"
    else
        print_warning "Some tests failed. Check output above."
    fi
else
    print_warning "pytest not found. Skipping tests..."
fi

# Setup complete
print_header "🎉 Setup Complete!"
echo
print_status "Next steps:"
echo "  1. Train models: python -m pump_predictor.main"
echo "  2. Start API: python -m pump_predictor.api.main"
echo "  3. Start Dashboard: streamlit run dashboard/streamlit_app.py"
echo
print_status "For Docker deployment:"
echo "  docker-compose -f deployment/docker/docker-compose.yml up"
echo
print_status "Documentation: Check README.md for detailed instructions"