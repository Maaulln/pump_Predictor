import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_pump_data(num_samples=2000):
    """
    Generate realistic pump sensor data with more complex patterns
    """
    
    # Start date
    start_date = datetime(2025, 1, 1)
    
    # Generate timestamps (every 30 minutes for multiple pumps)
    timestamps = []
    pump_ids = []
    
    current_date = start_date
    pump_count = 20  # 20 different pumps
    
    samples_per_pump = num_samples // pump_count
    
    for pump_num in range(1, pump_count + 1):
        pump_id = f"PUMP_{pump_num:03d}"
        pump_start = start_date + timedelta(days=pump_num * 5)  # Stagger pump start dates
        
        for i in range(samples_per_pump):
            timestamp = pump_start + timedelta(hours=i * 0.5)  # Every 30 minutes
            timestamps.append(timestamp)
            pump_ids.append(pump_id)
    
    # Add remaining samples to fill exact count
    remaining = num_samples - len(timestamps)
    for i in range(remaining):
        pump_id = f"PUMP_{(i % pump_count) + 1:03d}"
        timestamp = timestamps[-1] + timedelta(hours=(i + 1) * 0.5)
        timestamps.append(timestamp)
        pump_ids.append(pump_id)
    
    data = []
    
    for i, (timestamp, pump_id) in enumerate(zip(timestamps, pump_ids)):
        
        # Simulate pump aging and degradation
        days_operating = (timestamp - start_date).days
        aging_factor = 1 + (days_operating / 365) * 0.1  # 10% degradation per year
        
        # Base pump characteristics (each pump has slightly different baseline)
        pump_num = int(pump_id.split('_')[1])
        base_efficiency = 0.85 + (pump_num % 10) * 0.01  # 0.85 - 0.94
        
        # Seasonal effects (summer = higher temperatures)
        day_of_year = timestamp.timetuple().tm_yday
        seasonal_temp_factor = 1 + 0.3 * np.sin(2 * np.pi * day_of_year / 365)
        
        # Daily cycle (higher load during day hours)
        hour = timestamp.hour
        daily_load_factor = 0.7 + 0.3 * np.sin(2 * np.pi * (hour - 6) / 24)
        
        # Generate correlated features
        # Load factor affects many other parameters
        base_load = 0.5 + 0.4 * daily_load_factor + np.random.normal(0, 0.1)
        load_factor = np.clip(base_load, 0.1, 1.0)
        
        # Operating hours (cumulative)
        operating_hours = days_operating * 12 + hour * 0.5 + np.random.normal(0, 50)
        operating_hours = max(100, operating_hours)
        
        # Temperature increases with load and ambient conditions
        ambient_temperature = 15 + 15 * seasonal_temp_factor + np.random.normal(0, 2)
        temperature = ambient_temperature + 20 + 30 * load_factor * aging_factor + np.random.normal(0, 3)
        
        # Pressure decreases with degradation and increases with load
        pressure = 150 - 30 * load_factor + np.random.normal(0, 8) - aging_factor * 5
        
        # Vibration increases with degradation and load
        vibration = 1.5 + 2 * load_factor * aging_factor + np.random.normal(0, 0.3)
        
        # Flow rate decreases with degradation
        flow_rate = 250 - 40 * load_factor + np.random.normal(0, 10) - aging_factor * 10
        
        # Motor current increases with load and degradation
        motor_current = 8 + 15 * load_factor * aging_factor + np.random.normal(0, 2)
        
        # Bearing temperature correlated with temperature and load
        bearing_temperature = temperature - 5 + 10 * load_factor + np.random.normal(0, 2)
        
        # Oil level decreases over time (maintenance refills it)
        base_oil_level = 95 - (operating_hours % 2000) / 2000 * 20  # Decreases over 2000 hours
        oil_level = base_oil_level + np.random.normal(0, 2)
        oil_level = np.clip(oil_level, 60, 100)
        
        # Power consumption increases with load and decreases with efficiency
        efficiency = base_efficiency * (1 - aging_factor * 0.1) + np.random.normal(0, 0.02)
        efficiency = np.clip(efficiency, 0.6, 0.95)
        
        power_consumption = (motor_current * 240 * load_factor) / 1000 / efficiency + np.random.normal(0, 1)
        
        # Humidity (environmental factor)
        humidity = 50 + 25 * seasonal_temp_factor + np.random.normal(0, 5)
        humidity = np.clip(humidity, 30, 90)
        
        # Determine maintenance needs based on multiple conditions
        needs_maintenance = False
        
        # High temperature condition
        if temperature > 85:
            needs_maintenance = True
        
        # High vibration condition
        if vibration > 4.0:
            needs_maintenance = True
        
        # Low efficiency condition
        if efficiency < 0.72:
            needs_maintenance = True
        
        # Low oil level condition
        if oil_level < 75:
            needs_maintenance = True
        
        # High operating hours without maintenance
        if operating_hours % 2000 > 1800:  # Near maintenance interval
            needs_maintenance = True
        
        # Low pressure condition
        if pressure < 110:
            needs_maintenance = True
        
        # High bearing temperature
        if bearing_temperature > 80:
            needs_maintenance = True
        
        # Add some randomness to make it more realistic
        if np.random.random() < 0.05:  # 5% random maintenance needs
            needs_maintenance = True
        
        # Sometimes reset maintenance status (after maintenance is done)
        if needs_maintenance and np.random.random() < 0.1:  # 10% chance maintenance was just done
            needs_maintenance = False
            # Slightly improve parameters after maintenance
            temperature *= 0.95
            vibration *= 0.9
            efficiency *= 1.05
            oil_level = min(oil_level + 10, 100)
        
        # Round values to realistic precision
        data.append({
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'pump_id': pump_id,
            'temperature': round(temperature, 1),
            'pressure': round(pressure, 1),
            'vibration': round(vibration, 1),
            'flow_rate': round(flow_rate, 1),
            'motor_current': round(motor_current, 1),
            'bearing_temperature': round(bearing_temperature, 1),
            'oil_level': round(oil_level, 1),
            'power_consumption': round(power_consumption, 1),
            'efficiency': round(efficiency, 3),
            'operating_hours': round(operating_hours, 1),
            'load_factor': round(load_factor, 2),
            'ambient_temperature': round(ambient_temperature, 1),
            'humidity': round(humidity, 1),
            'needs_maintenance': needs_maintenance
        })
    
    return pd.DataFrame(data)

# Generate the dataset
print("Generating new pump dataset...")
df = generate_pump_data(2000)

# Display info about the new dataset
print(f"\nDataset shape: {df.shape}")
print(f"\nTarget distribution:")
print(df['needs_maintenance'].value_counts())
print(f"\nTarget percentage:")
print(df['needs_maintenance'].value_counts(normalize=True) * 100)

print(f"\nFeature statistics:")
print(df.describe())

# Save to CSV
output_path = 'data/pump_data.csv'
df.to_csv(output_path, index=False)
print(f"\nDataset saved to {output_path}")

# Display sample of the data
print(f"\nSample data:")
print(df.head(10))
