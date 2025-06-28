#!/usr/bin/env python3

# Test if we can import basic dependencies
try:
    import pandas as pd
    print("✓ pandas imported successfully")
except ImportError as e:
    print(f"✗ pandas import failed: {e}")

try:
    import numpy as np
    print("✓ numpy imported successfully")
except ImportError as e:
    print(f"✗ numpy import failed: {e}")

try:
    from sklearn.preprocessing import StandardScaler
    print("✓ sklearn imported successfully")
except ImportError as e:
    print(f"✗ sklearn import failed: {e}")

try:
    from pump_predictor.config import FEATURE_COLUMNS, TARGET_COLUMN
    print("✓ config imported successfully")
except ImportError as e:
    print(f"✗ config import failed: {e}")

try:
    from pump_predictor.utils.logger import get_logger
    print("✓ logger imported successfully")
except ImportError as e:
    print(f"✗ logger import failed: {e}")

# Now try to import our class
try:
    import sys
    sys.path.insert(0, '/Users/maaulln/Downloads/pump_Predictor-main')
    
    # Try to execute the preprocessing file as a module
    with open('/Users/maaulln/Downloads/pump_Predictor-main/pump_predictor/data/preprocessing.py', 'r') as f:
        content = f.read()
        print(f"File content length: {len(content)} characters")
        print("First 100 characters:")
        print(repr(content[:100]))
        
    exec(open('/Users/maaulln/Downloads/pump_Predictor-main/pump_predictor/data/preprocessing.py').read())
    print("✓ preprocessing.py executed successfully")
    
    # Try to access the class
    if 'DataPreprocessor' in locals():
        print("✓ DataPreprocessor class found")
    else:
        print("✗ DataPreprocessor class not found in locals")
        print("Available names:", list(locals().keys()))
        
except Exception as e:
    print(f"✗ Error executing preprocessing.py: {e}")
    import traceback
    traceback.print_exc()
