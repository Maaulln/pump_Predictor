#!/bin/bash

# Model Training Script
echo "🤖 Starting model training pipeline..."

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

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    print_warning "Virtual environment not detected. Activating..."
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        print_status "Virtual environment activated ✓"
    else
        print_error "Virtual environment not found. Run setup.sh first."
        exit 1
    fi
fi

# Parse command line arguments
TUNE_HYPERPARAMS=false
MODEL_TYPES=("random_forest" "xgboost" "lightgbm")
CREATE_ENSEMBLE=true
QUICK_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --tune)
            TUNE_HYPERPARAMS=true
            shift
            ;;
        --models)
            IFS=',' read -ra MODEL_TYPES <<< "$2"
            shift 2
            ;;
        --no-ensemble)
            CREATE_ENSEMBLE=false
            shift
            ;;
        --quick)
            QUICK_MODE=true
            TUNE_HYPERPARAMS=false
            MODEL_TYPES=("random_forest")
            CREATE_ENSEMBLE=false
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --tune              Enable hyperparameter tuning"
            echo "  --models MODEL_LIST Comma-separated list of models (random_forest,xgboost,lightgbm)"
            echo "  --no-ensemble       Skip ensemble creation"
            echo "  --quick             Quick training mode (RF only, no tuning)"
            echo "  --help              Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Display configuration
print_status "Training Configuration:"
echo "  Models: ${MODEL_TYPES[*]}"
echo "  Hyperparameter Tuning: $TUNE_HYPERPARAMS"
echo "  Create Ensemble: $CREATE_ENSEMBLE"
echo "  Quick Mode: $QUICK_MODE"
echo

# Check if data exists
if [ ! -f "data/pump_data.csv" ]; then
    print_warning "Training data not found. Generating synthetic data..."
    python3 -c "
import sys
sys.path.append('.')
from pump_predictor.data.preprocessing import DataPreprocessor
dp = DataPreprocessor()
sample_data = dp.create_synthetic_data(2000)
sample_data.to_csv('data/pump_data.csv', index=False)
print('✓ Synthetic data generated')
"
fi

# Build command
CMD="python -m pump_predictor.main"

if [ "$TUNE_HYPERPARAMS" = true ]; then
    CMD="$CMD --tune"
fi

if [ "$CREATE_ENSEMBLE" = false ]; then
    CMD="$CMD --no-ensemble"
fi

if [ "$QUICK_MODE" = true ]; then
    CMD="$CMD --quick"
else
    # Add models if not quick mode
    models_str=$(printf ",%s" "${MODEL_TYPES[@]}")
    models_str=${models_str:1}  # Remove leading comma
    CMD="$CMD --models $models_str"
fi

# Run training
print_status "Starting training with command: $CMD"
echo

start_time=$(date +%s)

eval $CMD

exit_code=$?
end_time=$(date +%s)
duration=$((end_time - start_time))

echo
if [ $exit_code -eq 0 ]; then
    print_status "✅ Training completed successfully in ${duration} seconds!"
    print_status "Models saved in: models/"
    print_status "Reports generated in: reports/"
    echo
    print_status "Next steps:"
    echo "  1. Start API: python -m pump_predictor.api.main"
    echo "  2. Start Dashboard: streamlit run dashboard/streamlit_app.py"
else
    print_error "❌ Training failed with exit code: $exit_code"
    exit $exit_code
fi