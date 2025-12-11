"""
Visualization functions for matplotlib charts embedded in Tkinter.
All functions return matplotlib.Figure objects for use with FigureCanvasTkAgg.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Tuple
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend


def create_elbow_curve(k_values: List[int], wcss_values: List[float], 
                       figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
    """
    Create elbow curve plot (k vs WCSS).
    
    Args:
        k_values: List of k values tested
        wcss_values: Corresponding WCSS values
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(k_values, wcss_values, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Within-Cluster Sum of Squares (WCSS)', fontsize=12)
    ax.set_title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(k_values)
    
    plt.tight_layout()
    return fig


def create_cluster_scatter(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray,
                          feature1: str, feature2: str, feature1_idx: int, feature2_idx: int,
                          figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Create 2D scatter plot of clusters with color-coded points and centroids.
    
    Args:
        X: Data points (n_samples, n_features)
        labels: Cluster assignments (n_samples,)
        centroids: Cluster centroids (k, n_features)
        feature1: Name of first feature (for axis label)
        feature2: Name of second feature (for axis label)
        feature1_idx: Index of first feature in X
        feature2_idx: Index of second feature in X
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get unique cluster labels
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    # Define colors for clusters
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    # Plot data points colored by cluster
    for i, cluster_id in enumerate(unique_labels):
        cluster_mask = labels == cluster_id
        cluster_points = X[cluster_mask]
        
        ax.scatter(cluster_points[:, feature1_idx], cluster_points[:, feature2_idx],
                  c=[colors[i]], label=f'Cluster {cluster_id}', alpha=0.6, s=50)
    
    # Plot centroids with distinct markers
    ax.scatter(centroids[:, feature1_idx], centroids[:, feature2_idx],
              c='red', marker='X', s=300, linewidths=2, 
              edgecolors='black', label='Centroids', zorder=10)
    
    # Add centroid labels
    for i, centroid in enumerate(centroids):
        ax.annotate(f'C{i}', (centroid[feature1_idx], centroid[feature2_idx]),
                   fontsize=12, fontweight='bold', ha='center', va='center',
                   color='white', zorder=11)
    
    ax.set_xlabel(feature1, fontsize=12)
    ax.set_ylabel(feature2, fontsize=12)
    ax.set_title(f'K-Means Clustering: {feature1} vs {feature2}', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_regression_scatter(y_true: np.ndarray, y_pred: np.ndarray, 
                             model_name: str, figsize: Tuple[int, int] = (8, 8)) -> plt.Figure:
    """
    Create scatter plot of actual vs predicted values with y=x reference line.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        model_name: Name of the model (for title)
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot of actual vs predicted
    ax.scatter(y_true, y_pred, alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
    
    # Perfect prediction line (y=x)
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
           label='Perfect Prediction (y=x)')
    
    ax.set_xlabel('Actual Values', fontsize=12)
    ax.set_ylabel('Predicted Values', fontsize=12)
    ax.set_title(f'{model_name}: Actual vs Predicted', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Make axes equal for better visualization
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    return fig

