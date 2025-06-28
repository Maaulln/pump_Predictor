"""
Comprehensive data validator for pump maintenance prediction
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import warnings
from scipy import stats

from pump_predictor.config import FEATURE_COLUMNS, TARGET_COLUMN
from pump_predictor.utils.logger import get_logger

logger = get_logger(__name__)

class ValidationLevel(Enum):
    """Validation strictness levels"""
    STRICT = "strict"      # Fail on any validation error
    MODERATE = "moderate"  # Warn on minor issues, fail on major
    LENIENT = "lenient"    # Only warn, don't fail

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    cleaned_data: Optional[pd.DataFrame] = None
    stats: Dict[str, Any] = None

class DataValidator:
    """Comprehensive data validator for pump sensor data"""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.MODERATE):
        self.validation_level = validation_level
        self.logger = logger
        
        # Define expected ranges for features
        self.feature_ranges = {
            'temperature': (-50, 200),        # Celsius
            'pressure': (0, 1000),           # PSI
            'vibration': (0, 100),           # Hz
            'flow_rate': (0, 1000),          # L/min
            'motor_current': (0, 50),        # Amperes
            'bearing_temperature': (-20, 150), # Celsius
            'oil_level': (0, 100),           # Percentage
            'power_consumption': (0, 100),    # kW
            'efficiency': (0, 100),          # Percentage
            'operating_hours': (0, 100000),  # Hours
            'load_factor': (0, 1),           # Ratio
            'ambient_temperature': (-40, 60), # Celsius
            'humidity': (0, 100)             # Percentage
        }
        
        # Define correlations that should exist
        self.expected_correlations = {
            ('temperature', 'bearing_temperature'): (0.3, 0.9),
            ('pressure', 'flow_rate'): (0.2, 0.8),
            ('power_consumption', 'load_factor'): (0.4, 0.9)
        }
    
    def validate_schema(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Validate data schema and structure"""
        errors = []
        warnings = []
        
        # Check required columns
        missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        # Check target column for training data
        if TARGET_COLUMN not in df.columns:
            warnings.append(f"Target column '{TARGET_COLUMN}' not found. Assuming inference mode.")
        
        # Check for unexpected columns
        expected_cols = set(FEATURE_COLUMNS + [TARGET_COLUMN])
        unexpected_cols = set(df.columns) - expected_cols
        if unexpected_cols:
            warnings.append(f"Unexpected columns found: {list(unexpected_cols)}")
        
        # Check data types
        for col in FEATURE_COLUMNS:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    warnings.append(f"Converted column '{col}' to numeric type")
                except:
                    errors.append(f"Column '{col}' cannot be converted to numeric")
        
        return errors, warnings
    
    def validate_ranges(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Validate feature value ranges"""
        errors = []
        warnings = []
        
        for col, (min_val, max_val) in self.feature_ranges.items():
            if col not in df.columns:
                continue
                
            # Check for values outside expected range
            out_of_range = df[(df[col] < min_val) | (df[col] > max_val)]
            if not out_of_range.empty:
                pct_invalid = len(out_of_range) / len(df) * 100
                msg = f"Column '{col}': {len(out_of_range)} values ({pct_invalid:.1f}%) outside expected range [{min_val}, {max_val}]"
                
                if pct_invalid > 10:  # More than 10% invalid
                    errors.append(msg)
                else:
                    warnings.append(msg)
        
        return errors, warnings
    
    def validate_missing_data(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Validate missing data patterns"""
        errors = []
        warnings = []
        
        # Check missing data percentage
        missing_pct = df.isnull().mean() * 100
        
        for col in FEATURE_COLUMNS:
            if col not in df.columns:
                continue
                
            pct = missing_pct[col]
            if pct > 50:
                errors.append(f"Column '{col}' has {pct:.1f}% missing values (>50%)")
            elif pct > 20:
                warnings.append(f"Column '{col}' has {pct:.1f}% missing values")
        
        # Check for completely missing rows
        completely_missing = df[FEATURE_COLUMNS].isnull().all(axis=1).sum()
        if completely_missing > 0:
            errors.append(f"{completely_missing} rows have all features missing")
        
        return errors, warnings
    
    def validate_distributions(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Validate statistical distributions"""
        errors = []
        warnings = []
        
        for col in FEATURE_COLUMNS:
            if col not in df.columns or df[col].isnull().all():
                continue
            
            values = df[col].dropna()
            if len(values) < 10:
                warnings.append(f"Column '{col}' has too few valid values for distribution analysis")
                continue
            
            # Check for constant values
            if values.nunique() == 1:
                warnings.append(f"Column '{col}' has constant values")
                continue
            
            # Check for extreme skewness
            try:
                skewness = stats.skew(values)
                if abs(skewness) > 3:
                    warnings.append(f"Column '{col}' is highly skewed (skewness: {skewness:.2f})")
            except:
                pass
            
            # Check for outliers using IQR method
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = values[(values < lower_bound) | (values > upper_bound)]
            if len(outliers) > len(values) * 0.1:  # More than 10% outliers
                warnings.append(f"Column '{col}' has {len(outliers)} potential outliers ({len(outliers)/len(values)*100:.1f}%)")
        
        return errors, warnings
    
    def validate_correlations(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Validate expected correlations between features"""
        errors = []
        warnings = []
        
        available_features = [col for col in FEATURE_COLUMNS if col in df.columns]
        if len(available_features) < 2:
            return errors, warnings
        
        try:
            corr_matrix = df[available_features].corr()
            
            for (col1, col2), (min_corr, max_corr) in self.expected_correlations.items():
                if col1 in corr_matrix.columns and col2 in corr_matrix.columns:
                    actual_corr = abs(corr_matrix.loc[col1, col2])
                    if actual_corr < min_corr:
                        warnings.append(f"Low correlation between '{col1}' and '{col2}': {actual_corr:.3f} (expected: {min_corr}-{max_corr})")
        except Exception as e:
            warnings.append(f"Could not compute correlations: {str(e)}")
        
        return errors, warnings
    
    def validate_target(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Validate target variable"""
        errors = []
        warnings = []
        
        if TARGET_COLUMN not in df.columns:
            return errors, warnings
        
        target = df[TARGET_COLUMN]
        
        # Check target distribution
        if target.isnull().any():
            missing_pct = target.isnull().mean() * 100
            if missing_pct > 5:
                errors.append(f"Target variable has {missing_pct:.1f}% missing values")
        
        # Check class balance for binary classification
        if target.nunique() == 2:
            class_counts = target.value_counts()
            minority_pct = class_counts.min() / class_counts.sum() * 100
            if minority_pct < 5:
                warnings.append(f"Severe class imbalance: minority class represents only {minority_pct:.1f}% of data")
            elif minority_pct < 20:
                warnings.append(f"Class imbalance detected: minority class represents {minority_pct:.1f}% of data")
        
        return errors, warnings
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data based on validation results"""
        cleaned_df = df.copy()
        
        # Remove rows with all features missing
        available_features = [col for col in FEATURE_COLUMNS if col in cleaned_df.columns]
        cleaned_df = cleaned_df.dropna(subset=available_features, how='all')
        
        # Cap outliers to reasonable ranges
        for col, (min_val, max_val) in self.feature_ranges.items():
            if col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].clip(lower=min_val, upper=max_val)
        
        # Fill missing values with median for numerical columns
        for col in available_features:
            if cleaned_df[col].isnull().any():
                median_val = cleaned_df[col].median()
                cleaned_df[col].fillna(median_val, inplace=True)
                logger.info(f"Filled missing values in '{col}' with median: {median_val}")
        
        return cleaned_df
    
    def validate(self, df: pd.DataFrame, clean_data: bool = True) -> ValidationResult:
        """Perform comprehensive data validation"""
        if df.empty:
            return ValidationResult(
                is_valid=False,
                errors=["Data is empty"],
                warnings=[],
                cleaned_data=None
            )
        
        all_errors = []
        all_warnings = []
        
        # Run all validation checks
        validation_methods = [
            self.validate_schema,
            self.validate_ranges,
            self.validate_missing_data,
            self.validate_distributions,
            self.validate_correlations,
            self.validate_target
        ]
        
        for method in validation_methods:
            try:
                errors, warnings = method(df)
                all_errors.extend(errors)
                all_warnings.extend(warnings)
            except Exception as e:
                all_errors.append(f"Validation error in {method.__name__}: {str(e)}")
        
        # Determine if data is valid based on validation level
        is_valid = True
        if self.validation_level == ValidationLevel.STRICT:
            is_valid = len(all_errors) == 0 and len(all_warnings) == 0
        elif self.validation_level == ValidationLevel.MODERATE:
            is_valid = len(all_errors) == 0
        # LENIENT: always valid, just warnings
        
        # Clean data if requested and possible
        cleaned_data = None
        if clean_data and not df.empty:
            try:
                cleaned_data = self.clean_data(df)
            except Exception as e:
                all_errors.append(f"Data cleaning failed: {str(e)}")
        
        # Generate statistics
        stats = self._generate_stats(cleaned_data if cleaned_data is not None else df)
        
        # Log results
        if all_errors:
            self.logger.error(f"Data validation failed with {len(all_errors)} errors")
            for error in all_errors:
                self.logger.error(f"  - {error}")
        
        if all_warnings:
            self.logger.warning(f"Data validation completed with {len(all_warnings)} warnings")
            for warning in all_warnings:
                self.logger.warning(f"  - {warning}")
        
        if is_valid and not all_errors and not all_warnings:
            self.logger.info("Data validation passed successfully")
        
        return ValidationResult(
            is_valid=is_valid,
            errors=all_errors,
            warnings=all_warnings,
            cleaned_data=cleaned_data,
            stats=stats
        )
    
    def _generate_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics"""
        if df.empty:
            return {}
        
        available_features = [col for col in FEATURE_COLUMNS if col in df.columns]
        
        stats = {
            'n_rows': len(df),
            'n_features': len(available_features),
            'missing_data_pct': df[available_features].isnull().mean().mean() * 100,
            'duplicate_rows': df.duplicated().sum(),
        }
        
        if TARGET_COLUMN in df.columns:
            target = df[TARGET_COLUMN]
            stats['target_stats'] = {
                'missing_pct': target.isnull().mean() * 100,
                'unique_values': target.nunique(),
                'distribution': target.value_counts().to_dict()
            }
        
        return stats
