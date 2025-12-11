"""
K-means clustering implementation from scratch.
No sklearn.cluster.KMeans used - pure numpy implementation.
"""

import numpy as np
from typing import Tuple, List, Optional
import random


def calculate_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        point1: First point as numpy array
        point2: Second point as numpy array
        
    Returns:
        Euclidean distance
    """
    return np.sqrt(np.sum((point1 - point2) ** 2))


def initialize_centroids_random(X: np.ndarray, k: int, random_state: Optional[int] = None) -> np.ndarray:
    """
    Initialize centroids randomly by selecting k random points from dataset.
    
    Args:
        X: Input data (n_samples, n_features)
        k: Number of clusters
        random_state: Random seed for reproducibility
        
    Returns:
        Array of initial centroids (k, n_features)
    """
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)
    
    n_samples, n_features = X.shape
    
    if k > n_samples:
        raise ValueError(f"Cannot create {k} clusters from {n_samples} samples.")
    
    # Select k random indices
    random_indices = random.sample(range(n_samples), k)
    centroids = X[random_indices].copy()
    
    return centroids


def initialize_centroids_kmeans_plus_plus(X: np.ndarray, k: int, random_state: Optional[int] = None) -> np.ndarray:
    """
    Initialize centroids using K-means++ algorithm.
    
    K-means++ selects the first centroid randomly, then selects subsequent
    centroids with probability proportional to the squared distance from the
    nearest existing centroid.
    
    Args:
        X: Input data (n_samples, n_features)
        k: Number of clusters
        random_state: Random seed for reproducibility
        
    Returns:
        Array of initial centroids (k, n_features)
    """
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)
    
    n_samples, n_features = X.shape
    
    if k > n_samples:
        raise ValueError(f"Cannot create {k} clusters from {n_samples} samples.")
    
    # Initialize centroids array
    centroids = np.zeros((k, n_features))
    
    # Step 1: Select first centroid randomly
    first_idx = random.randint(0, n_samples - 1)
    centroids[0] = X[first_idx].copy()
    
    # Step 2: Select remaining centroids
    for i in range(1, k):
        # Calculate distances from each point to nearest centroid
        distances = np.zeros(n_samples)
        for j in range(n_samples):
            min_dist = float('inf')
            for c in range(i):
                dist = calculate_distance(X[j], centroids[c])
                if dist < min_dist:
                    min_dist = dist
            distances[j] = min_dist ** 2  # Square the distance
        
        # Select next centroid with probability proportional to distance^2
        # Normalize distances to probabilities
        probabilities = distances / distances.sum()
        
        # Select index based on probabilities
        cumulative_probs = np.cumsum(probabilities)
        r = random.random()
        next_idx = np.searchsorted(cumulative_probs, r)
        
        # Ensure index is within bounds
        next_idx = min(next_idx, n_samples - 1)
        centroids[i] = X[next_idx].copy()
    
    return centroids


def assign_clusters(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Assign each data point to the nearest centroid.
    
    Args:
        X: Input data (n_samples, n_features)
        centroids: Current centroids (k, n_features)
        
    Returns:
        Array of cluster labels (n_samples,)
    """
    n_samples = X.shape[0]
    k = centroids.shape[0]
    labels = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        min_dist = float('inf')
        closest_centroid = 0
        
        for j in range(k):
            dist = calculate_distance(X[i], centroids[j])
            if dist < min_dist:
                min_dist = dist
                closest_centroid = j
        
        labels[i] = closest_centroid
    
    return labels


def update_centroids(X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """
    Recalculate centroids based on current cluster assignments.
    
    Args:
        X: Input data (n_samples, n_features)
        labels: Cluster assignments (n_samples,)
        k: Number of clusters
        
    Returns:
        Updated centroids (k, n_features)
    """
    n_features = X.shape[1]
    centroids = np.zeros((k, n_features))
    
    for i in range(k):
        # Find all points assigned to cluster i
        cluster_points = X[labels == i]
        
        if len(cluster_points) > 0:
            # Calculate mean of all points in cluster
            centroids[i] = cluster_points.mean(axis=0)
        else:
            # Handle empty cluster - keep previous centroid or use random point
            # In practice, we'll keep the previous centroid
            pass
    
    return centroids


def calculate_wcss(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    """
    Calculate Within-Cluster Sum of Squares (WCSS).
    
    WCSS = sum of squared distances from each point to its cluster centroid.
    
    Args:
        X: Input data (n_samples, n_features)
        labels: Cluster assignments (n_samples,)
        centroids: Cluster centroids (k, n_features)
        
    Returns:
        WCSS value
    """
    wcss = 0.0
    n_samples = X.shape[0]
    
    for i in range(n_samples):
        cluster_id = labels[i]
        centroid = centroids[cluster_id]
        dist = calculate_distance(X[i], centroid)
        wcss += dist ** 2
    
    return wcss


def kmeans(X: np.ndarray, k: int, init_method: str = 'kmeans++', 
           max_iters: int = 300, tol: float = 1e-4, 
           random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Main K-means clustering algorithm.
    
    Args:
        X: Input data (n_samples, n_features)
        k: Number of clusters
        init_method: Initialization method ('kmeans++' or 'random')
        max_iters: Maximum number of iterations
        tol: Tolerance for convergence (centroid movement threshold)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (labels, centroids, n_iterations)
    """
    n_samples, n_features = X.shape
    
    if k <= 0 or k > n_samples:
        raise ValueError(f"k must be between 1 and {n_samples}.")
    
    # Initialize centroids
    if init_method == 'kmeans++':
        centroids = initialize_centroids_kmeans_plus_plus(X, k, random_state)
    elif init_method == 'random':
        centroids = initialize_centroids_random(X, k, random_state)
    else:
        raise ValueError(f"Unknown init_method: {init_method}. Use 'kmeans++' or 'random'.")
    
    # Main K-means loop
    for iteration in range(max_iters):
        # Assign points to nearest centroids
        labels = assign_clusters(X, centroids)
        
        # Update centroids
        new_centroids = update_centroids(X, labels, k)
        
        # Check for convergence
        centroid_shift = 0.0
        for i in range(k):
            shift = calculate_distance(centroids[i], new_centroids[i])
            centroid_shift = max(centroid_shift, shift)
        
        centroids = new_centroids
        
        # Check if converged
        if centroid_shift < tol:
            break
    
    return labels, centroids, iteration + 1


def elbow_method(X: np.ndarray, k_range: List[int], init_method: str = 'kmeans++',
                 max_iters: int = 300, tol: float = 1e-4, 
                 random_state: Optional[int] = None) -> Tuple[List[int], List[float]]:
    """
    Calculate WCSS for different k values to find optimal number of clusters.
    
    Args:
        X: Input data (n_samples, n_features)
        k_range: List of k values to test (e.g., [2, 3, 4, 5, 6, 7, 8])
        init_method: Initialization method ('kmeans++' or 'random')
        max_iters: Maximum number of iterations per k
        tol: Tolerance for convergence
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (k_values, wcss_values)
    """
    k_values = []
    wcss_values = []
    
    for k in k_range:
        if k > X.shape[0]:
            continue  # Skip if k is larger than number of samples
        
        try:
            labels, centroids, n_iters = kmeans(X, k, init_method, max_iters, tol, random_state)
            wcss = calculate_wcss(X, labels, centroids)
            
            k_values.append(k)
            wcss_values.append(wcss)
        except Exception as e:
            print(f"Error computing k={k}: {e}")
            continue
    
    return k_values, wcss_values

