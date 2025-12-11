"""
Script to generate sample product_sales.csv with intentional data quality issues
for testing the ML application.
"""

import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate sample data
n_samples = 200

data = {
    'product_id': [f'PROD_{i:03d}' for i in range(1, n_samples + 1)],
    'price': np.random.normal(25, 10, n_samples),
    'cost': np.random.normal(15, 8, n_samples),
    'units_sold': np.random.poisson(100, n_samples),
    'promotion_frequency': np.random.randint(0, 10, n_samples),
}

# Ensure price > cost
data['price'] = np.maximum(data['price'], data['cost'] + 2)

# Calculate profit
data['profit'] = (data['price'] - data['cost']) * data['units_sold']

# Add some intentional data quality issues

# 1. Missing values (about 5% missing in price and units_sold)
missing_indices_price = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
missing_indices_units = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
data['price'][missing_indices_price] = np.nan
data['units_sold'][missing_indices_units] = np.nan

# 2. Outliers (introduce some extreme values)
outlier_indices_price = np.random.choice(n_samples, size=int(n_samples * 0.03), replace=False)
outlier_indices_units = np.random.choice(n_samples, size=int(n_samples * 0.03), replace=False)
data['price'][outlier_indices_price] = np.random.uniform(100, 200, len(outlier_indices_price))
data['units_sold'][outlier_indices_units] = np.random.poisson(500, len(outlier_indices_units))

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV in data directory
import os
os.makedirs('data', exist_ok=True)
output_path = os.path.join('data', 'product_sales.csv')
df.to_csv(output_path, index=False)
print(f"Generated {output_path} with {n_samples} samples")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Data shape: {df.shape}")

