# Supermarket Product Sales Analysis - ML Application

A comprehensive application for analyzing supermarket product sales data using Machine Learning techniques.

## Features

### 1. Data Preprocessing
- Missing value detection and handling (mean/median/mode imputation or drop)
- Outlier detection using IQR and Z-score methods
- Data normalization (Min-Max) and standardization (Z-score)
- Complete preprocessing pipeline tracking

### 2. K-Means Clustering (From Scratch)
- Pure NumPy implementation (no sklearn.cluster.KMeans)
- K-means++ and random initialization methods
- Elbow method for optimal k selection (k=2 through k=8)
- WCSS (Within-Cluster Sum of Squares) calculation
- 2D cluster visualization with centroids

### 3. Regression Analysis
- Linear Regression implementation
- Polynomial Regression (configurable degree 1-5)
- Train/test split (80-20)
- Model evaluation with MSE and MAE metrics
- Model comparison and performance analysis

### 4. Interactive GUI
- **Data Overview Tab**: Dataset statistics, missing values, outliers, normalization info
- **Clustering Analysis Tab**: Elbow curve, cluster visualization, statistics table, business insights
- **Regression Analysis Tab**: Model comparison, scatter plots, best model discussion
- **Summary Tab**: Key findings, recommendations, limitations, improvements

## Requirements

Install dependencies using:
```bash
pip install -r requirements.txt
```

Required packages:
- numpy >= 1.21.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0
- scikit-learn >= 1.0.0

## Usage

1. Place your `product_sales.csv` file in the `data/` directory
2. Run the application:
```bash
python src/main.py
```

   Or from the project root:
```bash
cd src
python main.py
```

3. The application will attempt to load `data/product_sales.csv` on startup
4. If the file is not found, use File > Load Data to select your CSV file

### Generate Sample Data

To generate sample data for testing:
```bash
python scripts/generate_sample_data.py
```

This will create a sample `product_sales.csv` file in the `data/` directory with intentional data quality issues.

## Data Format

The CSV file should contain the following columns (at minimum):
- `price`: Product price
- `cost`: Product cost
- `units_sold`: Number of units sold
- `profit`: Profit (target variable for regression)
- `promotion_frequency`: Frequency of promotions

Additional numeric columns are supported and will be included in the analysis.

## Project Structure

```
.
├── src/                    # Source code modules
│   ├── __init__.py
│   ├── main.py            # Main Tkinter GUI application
│   ├── preprocessing.py  # Data preprocessing module
│   ├── kmeans.py          # K-means clustering implementation
│   ├── regression.py      # Linear and Polynomial regression
│   └── visualization.py  # Matplotlib visualization functions
├── data/                   # Data files
│   ├── .gitkeep
│   └── product_sales.csv  # Input data file (place your CSV here)
├── scripts/                # Utility scripts
│   └── generate_sample_data.py  # Script to generate sample data
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore file
└── README.md              # This file
```

## Module Descriptions

### preprocessing.py
Handles all data cleaning and transformation:
- `load_data()`: Load CSV with error handling
- `detect_missing_values()`: Identify missing values
- `handle_missing_values()`: Impute or drop missing values
- `detect_outliers_iqr()` / `detect_outliers_zscore()`: Outlier detection
- `normalize_minmax()` / `standardize_zscore()`: Feature scaling
- `PreprocessingPipeline`: Tracks all preprocessing decisions

### kmeans.py
K-means clustering from scratch:
- `initialize_centroids_kmeans_plus_plus()`: K-means++ initialization
- `initialize_centroids_random()`: Random initialization
- `kmeans()`: Main clustering algorithm
- `elbow_method()`: Calculate WCSS for different k values
- `calculate_wcss()`: Within-cluster sum of squares

### regression.py
Regression models and evaluation:
- `linear_regression_fit()`: Fit linear model using normal equation
- `polynomial_regression_fit()`: Fit polynomial model
- `train_test_split()`: Split data for training/testing
- `calculate_mse()` / `calculate_mae()`: Evaluation metrics

### visualization.py
Matplotlib charts for Tkinter:
- `create_elbow_curve()`: Elbow method visualization
- `create_cluster_scatter()`: 2D cluster plot with centroids
- `create_regression_scatter()`: Actual vs predicted plot

## Notes

- K-means is implemented from scratch using only NumPy (no sklearn.cluster.KMeans)
- All preprocessing decisions are tracked and displayed in the GUI
- The application handles missing data files gracefully
- Error messages are user-friendly and informative
- All visualizations are embedded in the Tkinter interface

## License

This project is for educational purposes.

