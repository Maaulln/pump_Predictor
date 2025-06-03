"""
Data handling module for pump maintenance prediction
"""
import csv
from typing import List, Dict, Tuple
import json

class DataHandler:
    def __init__(self):
        self.data = []
        
    def load_sample_data(self) -> None:
        """Initialize with sample pump operational data"""
        self.data = [
            {"pressure": 100, "temperature": 85, "speed": 1750, "vibration": 2.5, "needs_maintenance": 0},
            {"pressure": 95, "temperature": 90, "speed": 1745, "vibration": 3.2, "needs_maintenance": 0},
            {"pressure": 85, "temperature": 95, "speed": 1730, "vibration": 4.8, "needs_maintenance": 1},
            {"pressure": 105, "temperature": 82, "speed": 1755, "vibration": 2.1, "needs_maintenance": 0},
            {"pressure": 80, "temperature": 98, "speed": 1720, "vibration": 5.2, "needs_maintenance": 1}
        ]
    
    def normalize_feature(self, values: List[float]) -> List[float]:
        """Normalize features to range [0,1]"""
        min_val = min(values)
        max_val = max(values)
        return [(x - min_val) / (max_val - min_val) if max_val > min_val else 0 for x in values]
    
    def prepare_data(self) -> Tuple[List[List[float]], List[int]]:
        """Prepare and normalize the data for training"""
        if not self.data:
            self.load_sample_data()
            
        # Extract features and labels
        features = []
        labels = []
        
        for record in self.data:
            features.append([
                record["pressure"],
                record["temperature"],
                record["speed"],
                record["vibration"]
            ])
            labels.append(record["needs_maintenance"])
            
        # Normalize each feature
        normalized_features = list(zip(*[
            self.normalize_feature([row[i] for row in features])
            for i in range(len(features[0]))
        ]))
        
        return list(normalized_features), labels