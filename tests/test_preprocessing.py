"""
Tests for data preprocessing functionality
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from pump_predictor.data.preprocessing import DataPreprocessor

class TestDataPreprocessor:
    """Test cases for DataPreprocessor"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample pump data"""
        np.random.seed(42)
        n_samples = 100
        
        data = {
            'temperature': np.random.normal(75, 10, n_samples),
            'pressure': np.random.normal(175, 25, n_samples),
            'vibration': np.random.gamma(2, 1.5, n_samples),
            'flow_rate': np.random.normal(250, 30, n_samples),
            'maintenance_needed': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        }
        
        # Add some missing values
        missing_indices = np.random.choice(n_samples, 5, replace=False)
        data['temperature'][missing_indices[:3]] = np.nan
        data['pressure'][missing_indices[3:]] = np.nan
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def preprocessor(self):
        """Create DataPreprocessor instance"""
        return DataPreprocessor()
    
    def test_initialization(self, preprocessor):
        """Test preprocessor initialization"""
        assert preprocessor.scaler is None
        assert preprocessor.feature_names is None
        assert preprocessor.target_column is None
    
    def test_load_data_from_dataframe(self, preprocessor, sample_data):
        """Test loading data from DataFrame"""
        loaded_data = preprocessor.load_data(sample_data)
        
        assert isinstance(loaded_data, pd.DataFrame)
        assert len(loaded_data) == len(sample_data)
        assert list(loaded_data.columns) == list(sample_data.columns)
    
    def test_handle_missing_values(self, preprocessor, sample_data):
        """Test missing value handling"""
        # Check that we have missing values
        assert sample_data.isnull().sum().sum() > 0
        
        cleaned_data = preprocessor.handle_missing_values(sample_data)
        
        # After cleaning, no missing values should remain
        assert cleaned_data.isnull().sum().sum() == 0
        assert len(cleaned_data) <= len(sample_data)  # Some rows might be dropped
    
    def test_handle_missing_values_imputation(self, preprocessor, sample_data):
        """Test missing value imputation"""
        cleaned_data = preprocessor.handle_missing_values(sample_data, strategy='mean')
        
        # No missing values after imputation
        assert cleaned_data.isnull().sum().sum() == 0
        # All rows should be preserved with imputation
        assert len(cleaned_data) == len(sample_data)
    
    def test_detect_outliers(self, preprocessor, sample_data):
        """Test outlier detection"""
        outliers = preprocessor.detect_outliers(sample_data)
        
        assert isinstance(outliers, pd.Series)
        assert len(outliers) == len(sample_data)
        assert outliers.dtype == bool
    
    def test_remove_outliers(self, preprocessor, sample_data):
        """Test outlier removal"""
        clean_data = preprocessor.remove_outliers(sample_data)
        
        assert isinstance(clean_data, pd.DataFrame)
        assert len(clean_data) <= len(sample_data)
        assert list(clean_data.columns) == list(sample_data.columns)
    
    def test_split_features_target(self, preprocessor, sample_data):
        """Test feature-target splitting"""
        X, y = preprocessor.split_features_target(sample_data, 'maintenance_needed')
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y) == len(sample_data)
        assert 'maintenance_needed' not in X.columns
        assert y.name == 'maintenance_needed'
    
    def test_scale_features(self, preprocessor, sample_data):
        """Test feature scaling"""
        # Remove target column for scaling
        features = sample_data.drop('maintenance_needed', axis=1)
        
        scaled_features = preprocessor.scale_features(features)
        
        assert isinstance(scaled_features, pd.DataFrame)
        assert scaled_features.shape == features.shape
        assert list(scaled_features.columns) == list(features.columns)
        
        # Check that scaler is fitted
        assert preprocessor.scaler is not None
        
        # Check scaling properties (approximately mean=0, std=1)
        assert np.allclose(scaled_features.mean(), 0, atol=1e-10)
        assert np.allclose(scaled_features.std(), 1, atol=1e-10)
    
    def test_prepare_data_complete_workflow(self, preprocessor, sample_data):
        """Test complete data preparation workflow"""
        X_train, X_test, y_train, y_test = preprocessor.prepare_data(sample_data)
        
        # Check shapes
        assert len(X_train) > len(X_test)  # Default 80-20 split
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        
        # Check that features are scaled
        if preprocessor.scaler is not None:
            # Training features should be approximately normalized
            train_means = X_train.mean()
            train_stds = X_train.std()
            assert np.allclose(train_means, 0, atol=0.1)
            assert np.allclose(train_stds, 1, atol=0.1)
        
        # Check target values
        assert set(y_train.unique()).issubset({0, 1})
        assert set(y_test.unique()).issubset({0, 1})
    
    def test_prepare_data_with_custom_params(self, preprocessor, sample_data):
        """Test data preparation with custom parameters"""
        X_train, X_test, y_train, y_test = preprocessor.prepare_data(
            sample_data,
            test_size=0.3,
            random_state=123,
            target_column='maintenance_needed'
        )
        
        total_samples = len(X_train) + len(X_test)
        test_ratio = len(X_test) / total_samples
        
        # Check test size (approximately 30%)
        assert 0.25 <= test_ratio <= 0.35
    
    def test_transform_new_data(self, preprocessor, sample_data):
        """Test transforming new data with fitted preprocessor"""
        # First, prepare training data to fit the preprocessor
        X_train, X_test, y_train, y_test = preprocessor.prepare_data(sample_data)
        
        # Create new sample data
        new_data = pd.DataFrame({
            'temperature': [80.0, 70.0],
            'pressure': [180.0, 160.0],
            'vibration': [3.0, 1.5],
            'flow_rate': [260.0, 240.0]
        })
        
        # Transform new data
        transformed_data = preprocessor.transform_new_data(new_data)
        
        assert isinstance(transformed_data, np.ndarray)
        assert transformed_data.shape == (2, 4)
        
        # Check that transformation is consistent with training scaling
        assert not np.allclose(transformed_data, new_data.values)  # Should be scaled
    
    def test_transform_new_data_without_fitted_scaler(self, preprocessor):
        """Test transforming new data without fitted scaler"""
        new_data = pd.DataFrame({
            'temperature': [80.0],
            'pressure': [180.0],
            'vibration': [3.0],
            'flow_rate': [260.0]
        })
        
        with pytest.raises(ValueError, match="Scaler not fitted"):
            preprocessor.transform_new_data(new_data)
    
    def test_get_feature_names(self, preprocessor, sample_data):
        """Test getting feature names"""
        # Before fitting
        assert preprocessor.get_feature_names() is None
        
        # After fitting
        X_train, X_test, y_train, y_test = preprocessor.prepare_data(sample_data)
        feature_names = preprocessor.get_feature_names()
        
        assert isinstance(feature_names, list)
        assert 'maintenance_needed' not in feature_names
        assert len(feature_names) == X_train.shape[1]
    
    def test_create_synthetic_data(self, preprocessor):
        """Test synthetic data generation"""
        synthetic_data = preprocessor.create_synthetic_data(n_samples=50)
        
        assert isinstance(synthetic_data, pd.DataFrame)
        assert len(synthetic_data) == 50
        assert 'maintenance_needed' in synthetic_data.columns
        
        # Check data types and ranges
        assert synthetic_data['temperature'].between(40, 120).all()
        assert synthetic_data['pressure'].between(100, 300).all()
        assert synthetic_data['vibration'].between(0, 10).all()
        assert synthetic_data['flow_rate'].between(150, 400).all()
        assert set(synthetic_data['maintenance_needed'].unique()).issubset({0, 1})
    
    def test_get_data_summary(self, preprocessor, sample_data):
        """Test data summary generation"""
        summary = preprocessor.get_data_summary(sample_data)
        
        assert isinstance(summary, dict)
        assert 'shape' in summary
        assert 'missing_values' in summary
        assert 'data_types' in summary
        assert 'numerical_summary' in summary
        
        # Check summary contents
        assert summary['shape'] == sample_data.shape
        assert summary['missing_values']['total'] > 0  # We added missing values
        assert len(summary['data_types']) == len(sample_data.columns)
    
    def test_save_and_load_preprocessor(self, preprocessor, sample_data, tmp_path):
        """Test saving and loading preprocessor"""
        # Fit the preprocessor
        X_train, X_test, y_train, y_test = preprocessor.prepare_data(sample_data)
        
        # Save preprocessor
        save_path = tmp_path / "preprocessor.joblib"
        preprocessor.save_preprocessor(save_path)
        
        assert save_path.exists()
        
        # Load preprocessor
        new_preprocessor = DataPreprocessor()
        new_preprocessor.load_preprocessor(save_path)
        
        # Check that loaded preprocessor has the same properties
        assert new_preprocessor.scaler is not None
        assert new_preprocessor.feature_names == preprocessor.feature_names
        assert new_preprocessor.target_column == preprocessor.target_column
        
        # Test that transformation works the same
        new_data = pd.DataFrame({
            'temperature': [80.0],
            'pressure': [180.0],
            'vibration': [3.0],
            'flow_rate': [260.0]
        })
        
        original_transform = preprocessor.transform_new_data(new_data)
        loaded_transform = new_preprocessor.transform_new_data(new_data)
        
        assert np.allclose(original_transform, loaded_transform)
    
    def test_error_handling_invalid_target(self, preprocessor, sample_data):
        """Test error handling for invalid target column"""
        with pytest.raises(KeyError):
            preprocessor.prepare_data(sample_data, target_column='invalid_column')
    
    def test_error_handling_empty_data(self, preprocessor):
        """Test error handling for empty data"""
        empty_data = pd.DataFrame()
        
        with pytest.raises(ValueError, match="Data is empty"):
            preprocessor.prepare_data(empty_data)
    
    def test_feature_engineering(self, preprocessor, sample_data):
        """Test feature engineering capabilities"""
        # Remove target for feature engineering
        features = sample_data.drop('maintenance_needed', axis=1)
        
        engineered_features = preprocessor.engineer_features(features)
        
        assert isinstance(engineered_features, pd.DataFrame)
        assert engineered_features.shape[1] >= features.shape[1]  # Should have more features
        
        # Check that new features are created
        expected_new_features = ['temp_pressure_ratio', 'vibration_flow_ratio']
        for feature in expected_new_features:
            assert feature in engineered_features.columns
    
    def test_data_validation(self, preprocessor):
        """Test data validation"""
        # Valid data
        valid_data = pd.DataFrame({
            'temperature': [75.0, 80.0],
            'pressure': [175.0, 180.0],
            'vibration': [2.0, 3.0],
            'flow_rate': [250.0, 260.0],
            'maintenance_needed': [0, 1]
        })
        
        assert preprocessor.validate_data(valid_data) == True
        
        # Invalid data (negative values where they shouldn't be)
        invalid_data = pd.DataFrame({
            'temperature': [75.0, -50.0],  # Extremely low temperature
            'pressure': [-10.0, 180.0],    # Negative pressure
            'vibration': [2.0, 3.0],
            'flow_rate': [250.0, 260.0],
            'maintenance_needed': [0, 1]
        })
        
        assert preprocessor.validate_data(invalid_data) == False
    
    def test_correlation_analysis(self, preprocessor, sample_data):
        """Test correlation analysis"""
        correlation_report = preprocessor.analyze_correlations(sample_data)
        
        assert isinstance(correlation_report, dict)
        assert 'correlation_matrix' in correlation_report
        assert 'high_correlations' in correlation_report
        assert 'target_correlations' in correlation_report
        
        # Check correlation matrix shape
        expected_features = ['temperature', 'pressure', 'vibration', 'flow_rate']
        assert correlation_report['correlation_matrix'].shape == (len(expected_features), len(expected_features))