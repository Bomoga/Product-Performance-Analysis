"""
Linear and Polynomial Regression implementation.
Includes model training, prediction, and evaluation metrics.
"""

import numpy as np
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split as sklearn_train_test_split


def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, 
                     random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and testing sets.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        test_size: Proportion of data for testing (default 0.2 for 80-20 split)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    return sklearn_train_test_split(X, y, test_size=test_size, random_state=random_state)


def linear_regression_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Fit linear regression model using normal equation.
    
    Normal equation: coefficients = (X^T * X)^(-1) * X^T * y
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        
    Returns:
        Coefficient vector (n_features + 1,) including bias term
    """
    # Add bias term (column of ones)
    X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
    
    # Normal equation: (X^T * X)^(-1) * X^T * y
    try:
        # Calculate (X^T * X)^(-1) * X^T
        XTX = np.dot(X_with_bias.T, X_with_bias)
        XTX_inv = np.linalg.inv(XTX)
        XTy = np.dot(X_with_bias.T, y)
        coefficients = np.dot(XTX_inv, XTy)
    except np.linalg.LinAlgError:
        # If matrix is singular, use pseudo-inverse
        XTX_inv = np.linalg.pinv(np.dot(X_with_bias.T, X_with_bias))
        coefficients = np.dot(XTX_inv, np.dot(X_with_bias.T, y))
    
    return coefficients


def linear_regression_predict(X: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """
    Make predictions using linear regression model.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        coefficients: Model coefficients (n_features + 1,) including bias
        
    Returns:
        Predicted values (n_samples,)
    """
    # Add bias term
    X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
    
    # Predictions: X * coefficients
    predictions = np.dot(X_with_bias, coefficients)
    
    return predictions


def create_polynomial_features(X: np.ndarray, degree: int) -> np.ndarray:
    """
    Create polynomial features up to specified degree.
    
    For degree=2: [x1, x2] -> [x1, x2, x1^2, x1*x2, x2^2]
    
    Args:
        X: Feature matrix (n_samples, n_features)
        degree: Polynomial degree
        
    Returns:
        Polynomial feature matrix (bias term will be added in linear_regression_predict)
    """
    n_samples, n_features = X.shape
    
    # Start with original features (degree 1)
    poly_features = [X]
    
    # Add higher degree features
    for d in range(2, degree + 1):
        # Add individual feature powers: x1^d, x2^d, etc.
        for i in range(n_features):
            poly_features.append((X[:, i] ** d).reshape(-1, 1))
        
        # Add interaction terms for degree 2
        if d == 2 and n_features > 1:
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    poly_features.append((X[:, i] * X[:, j]).reshape(-1, 1))
    
    # Concatenate all features
    if len(poly_features) > 1:
        X_poly = np.hstack(poly_features)
    else:
        X_poly = poly_features[0]
    
    return X_poly


def polynomial_regression_fit(X: np.ndarray, y: np.ndarray, degree: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit polynomial regression model.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        degree: Polynomial degree
        
    Returns:
        Tuple of (coefficients, X_poly) where X_poly is the polynomial feature matrix
    """
    # Create polynomial features
    X_poly = create_polynomial_features(X, degree)
    
    # Fit linear regression on polynomial features
    coefficients = linear_regression_fit(X_poly, y)
    
    return coefficients, X_poly


def polynomial_regression_predict(X: np.ndarray, coefficients: np.ndarray, degree: int) -> np.ndarray:
    """
    Make predictions using polynomial regression model.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        coefficients: Model coefficients
        degree: Polynomial degree used during training
        
    Returns:
        Predicted values (n_samples,)
    """
    # Create polynomial features (must match training)
    X_poly = create_polynomial_features(X, degree)
    
    # Make predictions using linear regression predict (which adds bias term)
    predictions = linear_regression_predict(X_poly, coefficients)
    
    return predictions


def calculate_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Squared Error (MSE).
    
    MSE = (1/n) * sum((y_true - y_pred)^2)
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        
    Returns:
        MSE value
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    
    mse = np.mean((y_true - y_pred) ** 2)
    return float(mse)


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error (MAE).
    
    MAE = (1/n) * sum(|y_true - y_pred|)
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        
    Returns:
        MAE value
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    
    mae = np.mean(np.abs(y_true - y_pred))
    return float(mae)


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """
    Evaluate model performance using MSE and MAE.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        
    Returns:
        Tuple of (MSE, MAE)
    """
    mse = calculate_mse(y_true, y_pred)
    mae = calculate_mae(y_true, y_pred)
    
    return mse, mae

