"""
Data preprocessing module for supermarket product sales analysis.
Handles missing values, outliers, and normalization/standardization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


class PreprocessingPipeline:
    """Tracks all preprocessing steps and decisions."""
    
    def __init__(self):
        self.missing_values_info = {}
        self.missing_values_strategy = None
        self.outliers_info = {}
        self.outliers_method = None
        self.normalization_method = None
        self.normalized_columns = []
        self.original_data_shape = None
        self.processed_data_shape = None
    
    def get_summary(self) -> Dict:
        """Return a summary of all preprocessing steps."""
        return {
            'missing_values': self.missing_values_info,
            'missing_strategy': self.missing_values_strategy,
            'outliers': self.outliers_info,
            'outliers_method': self.outliers_method,
            'normalization': self.normalization_method,
            'normalized_columns': self.normalized_columns,
            'original_shape': self.original_data_shape,
            'processed_shape': self.processed_data_shape
        }


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load CSV data with error handling.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame containing the loaded data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        pd.errors.EmptyDataError: If file is empty
    """
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            raise ValueError("The dataset is empty.")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}. Please ensure the file exists.")
    except pd.errors.EmptyDataError:
        raise ValueError("The CSV file is empty.")


def detect_missing_values(df: pd.DataFrame) -> Dict[str, int]:
    """
    Identify missing values per column.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with column names as keys and missing value counts as values
    """
    missing = df.isnull().sum()
    return {col: int(count) for col, count in missing.items() if count > 0}


def handle_missing_values(df: pd.DataFrame, strategy: str = 'mean') -> Tuple[pd.DataFrame, str]:
    """
    Handle missing values using specified strategy.
    
    Args:
        df: Input DataFrame
        strategy: Strategy to use ('mean', 'median', 'mode', 'drop')
        
    Returns:
        Tuple of (processed DataFrame, justification string)
    """
    df_processed = df.copy()
    justification = ""
    
    missing_before = df.isnull().sum().sum()
    
    if missing_before == 0:
        return df_processed, "No missing values detected."
    
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    
    if strategy == 'drop':
        df_processed = df_processed.dropna()
        justification = f"Dropped rows with missing values. Removed {missing_before} missing values."
    elif strategy == 'mean':
        for col in numeric_cols:
            if df_processed[col].isnull().sum() > 0:
                mean_val = df_processed[col].mean()
                df_processed[col].fillna(mean_val, inplace=True)
        justification = f"Imputed missing values using mean for numeric columns. Handled {missing_before} missing values."
    elif strategy == 'median':
        for col in numeric_cols:
            if df_processed[col].isnull().sum() > 0:
                median_val = df_processed[col].median()
                df_processed[col].fillna(median_val, inplace=True)
        justification = f"Imputed missing values using median for numeric columns. Handled {missing_before} missing values."
    elif strategy == 'mode':
        for col in df_processed.columns:
            if df_processed[col].isnull().sum() > 0:
                mode_val = df_processed[col].mode()[0] if not df_processed[col].mode().empty else None
                if mode_val is not None:
                    df_processed[col].fillna(mode_val, inplace=True)
        justification = f"Imputed missing values using mode for all columns. Handled {missing_before} missing values."
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'mean', 'median', 'mode', or 'drop'.")
    
    return df_processed, justification


def detect_outliers_iqr(df: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, List[int]]:
    """
    Detect outliers using Interquartile Range (IQR) method.
    
    Args:
        df: Input DataFrame
        columns: List of column names to check. If None, checks all numeric columns.
        
    Returns:
        Dictionary with column names as keys and list of outlier indices as values
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    outliers = {}
    
    for col in columns:
        if col not in df.columns:
            continue
            
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_indices = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
        outliers[col] = outlier_indices
    
    return outliers


def detect_outliers_zscore(df: pd.DataFrame, columns: Optional[List[str]] = None, threshold: float = 3.0) -> Dict[str, List[int]]:
    """
    Detect outliers using Z-score method.
    
    Args:
        df: Input DataFrame
        columns: List of column names to check. If None, checks all numeric columns.
        threshold: Z-score threshold (default 3.0)
        
    Returns:
        Dictionary with column names as keys and list of outlier indices as values
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    outliers = {}
    
    for col in columns:
        if col not in df.columns:
            continue
            
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        outlier_indices = df[z_scores > threshold].index.tolist()
        outliers[col] = outlier_indices
    
    return outliers


def normalize_minmax(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Normalize features using Min-Max scaling (0 to 1).
    
    Args:
        df: Input DataFrame
        columns: List of column names to normalize. If None, normalizes all numeric columns.
        
    Returns:
        DataFrame with normalized columns
    """
    df_normalized = df.copy()
    
    if columns is None:
        columns = df_normalized.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if col not in df_normalized.columns:
            continue
            
        col_min = df_normalized[col].min()
        col_max = df_normalized[col].max()
        
        if col_max != col_min:
            df_normalized[col] = (df_normalized[col] - col_min) / (col_max - col_min)
        else:
            # Handle constant columns
            df_normalized[col] = 0.0
    
    return df_normalized


def standardize_zscore(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Standardize features using Z-score (mean=0, std=1).
    
    Args:
        df: Input DataFrame
        columns: List of column names to standardize. If None, standardizes all numeric columns.
        
    Returns:
        DataFrame with standardized columns
    """
    df_standardized = df.copy()
    
    if columns is None:
        columns = df_standardized.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if col not in df_standardized.columns:
            continue
            
        col_mean = df_standardized[col].mean()
        col_std = df_standardized[col].std()
        
        if col_std != 0:
            df_standardized[col] = (df_standardized[col] - col_mean) / col_std
        else:
            # Handle constant columns
            df_standardized[col] = 0.0
    
    return df_standardized

