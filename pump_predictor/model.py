"""
Simple machine learning model for pump maintenance prediction
"""
import math
from typing import List, Tuple

class SimplePredictor:
    def __init__(self):
        self.weights = None
        self.bias = 0
        
    def sigmoid(self, x: float) -> float:
        """Sigmoid activation function"""
        return 1 / (1 + math.exp(-x))
    
    def train(self, features: List[List[float]], labels: List[int], 
              learning_rate: float = 0.1, epochs: int = 100) -> None:
        """Train the model using simple logistic regression"""
        # Initialize weights
        num_features = len(features[0])
        self.weights = [0.0] * num_features
        
        # Training loop
        for _ in range(epochs):
            for feature_set, label in zip(features, labels):
                # Forward pass
                prediction = self.predict(feature_set)
                error = label - prediction
                
                # Update weights
                for i in range(num_features):
                    self.weights[i] += learning_rate * error * feature_set[i]
                self.bias += learning_rate * error
    
    def predict(self, features: List[float]) -> float:
        """Make a prediction for a single sample"""
        if self.weights is None:
            raise ValueError("Model not trained yet")
            
        # Calculate weighted sum
        z = sum(w * x for w, x in zip(self.weights, features)) + self.bias
        return self.sigmoid(z)
    
    def evaluate(self, features: List[List[float]], labels: List[int]) -> dict:
        """Evaluate model performance"""
        predictions = [self.predict(feature_set) for feature_set in features]
        threshold_predictions = [1 if p >= 0.5 else 0 for p in predictions]
        
        # Calculate accuracy
        correct = sum(1 for p, l in zip(threshold_predictions, labels) if p == l)
        accuracy = correct / len(labels)
        
        return {
            "accuracy": accuracy,
            "total_samples": len(labels),
            "correct_predictions": correct
        }