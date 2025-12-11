"""
Main Tkinter GUI application for Supermarket Product Sales Analysis.
Implements multi-tab interface with data overview, clustering, regression, and summary.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading

# Import custom modules
import preprocessing
import kmeans
import regression
import visualization


class Application:
    """Main application class for the ML Supermarket Analysis App."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Supermarket Product Sales Analysis")
        self.root.geometry("1200x800")
        
        # Data storage
        self.raw_data = None
        self.processed_data = None
        self.preprocessing_pipeline = preprocessing.PreprocessingPipeline()
        self.clustering_results = {}
        self.regression_results = {}
        
        # Feature selection for clustering
        self.clustering_features = []
        self.feature_names = []
        
        # Create GUI
        self.create_widgets()
        
        # Try to load data on startup
        self.load_data_on_startup()
    
    def create_widgets(self):
        """Create the main GUI structure with tabs."""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_data_overview_tab()
        self.create_clustering_tab()
        self.create_regression_tab()
        self.create_summary_tab()
        
        # Create menu bar
        self.create_menu()
        
        # Create status bar
        self.status_label = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
    
    def create_menu(self):
        """Create menu bar with file operations."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Data", command=self.load_data_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
    
    def load_data_on_startup(self):
        """Try to load product_sales.csv on startup."""
        import os
        # Try data directory first, then root directory
        data_paths = [
            os.path.join("data", "product_sales.csv"),
            "data/product_sales.csv",
            "product_sales.csv"
        ]
        for path in data_paths:
            try:
                if os.path.exists(path):
                    self.load_data(path)
                    return
            except:
                continue
        # File not found - user can load it manually
    
    def load_data_dialog(self):
        """Open file dialog to load data."""
        filepath = filedialog.askopenfilename(
            title="Select Data File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filepath:
            self.load_data(filepath)
    
    def load_data(self, filepath: str):
        """Load and preprocess data."""
        try:
            # Update status
            self.update_status("Loading data...")
            
            # Load raw data
            self.raw_data = preprocessing.load_data(filepath)
            self.preprocessing_pipeline.original_data_shape = self.raw_data.shape
            
            # Preprocess data
            self.preprocess_data()
            
            # Update GUI
            self.update_data_overview()
            self.update_feature_combos()
            self.update_status(f"Data loaded successfully: {self.raw_data.shape[0]} rows, {self.raw_data.shape[1]} columns")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data:\n{str(e)}")
            self.update_status("Error loading data")
    
    def preprocess_data(self):
        """Apply preprocessing pipeline to data."""
        if self.raw_data is None:
            return
        
        df = self.raw_data.copy()
        
        # Detect missing values
        missing = preprocessing.detect_missing_values(df)
        self.preprocessing_pipeline.missing_values_info = missing
        
        # Handle missing values (use mean imputation by default)
        if missing:
            df, justification = preprocessing.handle_missing_values(df, strategy='mean')
            self.preprocessing_pipeline.missing_values_strategy = 'mean'
        else:
            self.preprocessing_pipeline.missing_values_strategy = 'none'
        
        # Detect outliers using IQR
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        outliers = preprocessing.detect_outliers_iqr(df, numeric_cols)
        self.preprocessing_pipeline.outliers_info = outliers
        self.preprocessing_pipeline.outliers_method = 'IQR'
        
        # Normalize using Min-Max
        df_normalized = preprocessing.normalize_minmax(df, numeric_cols)
        self.preprocessing_pipeline.normalization_method = 'Min-Max'
        self.preprocessing_pipeline.normalized_columns = numeric_cols
        
        self.processed_data = df_normalized
        self.preprocessing_pipeline.processed_data_shape = df_normalized.shape
        
        # Store feature names for clustering
        self.feature_names = numeric_cols
    
    def update_status(self, message: str):
        """Update status bar if it exists."""
        if hasattr(self, 'status_label'):
            self.status_label.config(text=message)
        self.root.update_idletasks()
    
    # ==================== DATA OVERVIEW TAB ====================
    
    def create_data_overview_tab(self):
        """Create Data Overview tab."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Data Overview")
        
        # Create scrollable frame
        canvas = tk.Canvas(tab)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Statistics section
        stats_frame = ttk.LabelFrame(scrollable_frame, text="Dataset Statistics", padding=10)
        stats_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.stats_tree = ttk.Treeview(stats_frame, columns=("Mean", "Median", "Std Dev"), show="headings", height=10)
        self.stats_tree.heading("#0", text="Feature")
        self.stats_tree.heading("Mean", text="Mean")
        self.stats_tree.heading("Median", text="Median")
        self.stats_tree.heading("Std Dev", text="Std Dev")
        self.stats_tree.column("#0", width=150)
        self.stats_tree.column("Mean", width=120)
        self.stats_tree.column("Median", width=120)
        self.stats_tree.column("Std Dev", width=120)
        self.stats_tree.pack(fill=tk.BOTH, expand=True)
        
        # Missing values section
        missing_frame = ttk.LabelFrame(scrollable_frame, text="Missing Values", padding=10)
        missing_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.missing_text = tk.Text(missing_frame, height=5, wrap=tk.WORD)
        self.missing_text.pack(fill=tk.BOTH, expand=True)
        
        # Outliers section
        outliers_frame = ttk.LabelFrame(scrollable_frame, text="Outliers Detection", padding=10)
        outliers_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.outliers_text = tk.Text(outliers_frame, height=5, wrap=tk.WORD)
        self.outliers_text.pack(fill=tk.BOTH, expand=True)
        
        # Normalization section
        norm_frame = ttk.LabelFrame(scrollable_frame, text="Normalization", padding=10)
        norm_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.norm_text = tk.Text(norm_frame, height=3, wrap=tk.WORD)
        self.norm_text.pack(fill=tk.BOTH, expand=True)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def update_data_overview(self):
        """Update Data Overview tab with current data."""
        if self.processed_data is None:
            return
        
        # Update statistics
        for item in self.stats_tree.get_children():
            self.stats_tree.delete(item)
        
        numeric_cols = self.processed_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            mean_val = self.processed_data[col].mean()
            median_val = self.processed_data[col].median()
            std_val = self.processed_data[col].std()
            
            self.stats_tree.insert("", tk.END, text=col, values=(
                f"{mean_val:.2f}", f"{median_val:.2f}", f"{std_val:.2f}"
            ))
        
        # Update missing values
        self.missing_text.delete(1.0, tk.END)
        if self.preprocessing_pipeline.missing_values_info:
            self.missing_text.insert(tk.END, "Missing Values Detected:\n")
            for col, count in self.preprocessing_pipeline.missing_values_info.items():
                pct = (count / self.preprocessing_pipeline.original_data_shape[0]) * 100
                self.missing_text.insert(tk.END, f"  • {col}: {count} ({pct:.2f}%)\n")
            self.missing_text.insert(tk.END, f"\nStrategy Applied: {self.preprocessing_pipeline.missing_values_strategy}\n")
        else:
            self.missing_text.insert(tk.END, "No missing values detected.")
        
        # Update outliers
        self.outliers_text.delete(1.0, tk.END)
        if self.preprocessing_pipeline.outliers_info:
            self.outliers_text.insert(tk.END, f"Method: {self.preprocessing_pipeline.outliers_method}\n\n")
            total_outliers = 0
            for col, indices in self.preprocessing_pipeline.outliers_info.items():
                count = len(indices)
                total_outliers += count
                pct = (count / self.preprocessing_pipeline.processed_data_shape[0]) * 100
                self.outliers_text.insert(tk.END, f"  • {col}: {count} outliers ({pct:.2f}%)\n")
            self.outliers_text.insert(tk.END, f"\nTotal outliers detected: {total_outliers}")
        else:
            self.outliers_text.insert(tk.END, "No outliers detected.")
        
        # Update normalization
        self.norm_text.delete(1.0, tk.END)
        self.norm_text.insert(tk.END, f"Method: {self.preprocessing_pipeline.normalization_method}\n")
        self.norm_text.insert(tk.END, f"Normalized columns: {len(self.preprocessing_pipeline.normalized_columns)}\n")
        self.norm_text.insert(tk.END, f"All numeric features have been normalized to [0, 1] range.")
    
    # ==================== CLUSTERING TAB ====================
    
    def create_clustering_tab(self):
        """Create Clustering Analysis tab."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Clustering Analysis")
        
        # Control frame
        control_frame = ttk.Frame(tab)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Feature selection
        ttk.Label(control_frame, text="Features for Clustering:").pack(side=tk.LEFT, padx=5)
        self.feature1_var = tk.StringVar()
        self.feature2_var = tk.StringVar()
        self.feature1_combo = ttk.Combobox(control_frame, textvariable=self.feature1_var, width=15, state="readonly")
        self.feature1_combo.pack(side=tk.LEFT, padx=5)
        self.feature2_combo = ttk.Combobox(control_frame, textvariable=self.feature2_var, width=15, state="readonly")
        self.feature2_combo.pack(side=tk.LEFT, padx=5)
        
        # K value selector
        ttk.Label(control_frame, text="K:").pack(side=tk.LEFT, padx=5)
        self.k_var = tk.IntVar(value=3)
        k_spinbox = ttk.Spinbox(control_frame, from_=2, to=8, textvariable=self.k_var, width=5)
        k_spinbox.pack(side=tk.LEFT, padx=5)
        
        # Buttons
        ttk.Button(control_frame, text="Run Elbow Method", command=self.run_elbow_method).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Run Clustering", command=self.run_clustering).pack(side=tk.LEFT, padx=5)
        
        # Main content frame
        content_frame = ttk.Frame(tab)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Left side: Plots
        plots_frame = ttk.Frame(content_frame)
        plots_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Elbow curve frame
        self.elbow_frame = ttk.LabelFrame(plots_frame, text="Elbow Curve", padding=5)
        self.elbow_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.elbow_canvas = None
        
        # Cluster scatter frame
        self.scatter_frame = ttk.LabelFrame(plots_frame, text="Cluster Visualization", padding=5)
        self.scatter_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.scatter_canvas = None
        
        # Right side: Statistics and insights
        right_frame = ttk.Frame(content_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=5)
        
        # Cluster statistics
        stats_frame = ttk.LabelFrame(right_frame, text="Cluster Statistics", padding=5)
        stats_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create treeview for cluster stats
        self.cluster_stats_tree = ttk.Treeview(stats_frame, columns=("Count", "Avg Price", "Avg Cost", "Avg Units", "Avg Profit", "Avg Promo"), 
                                               show="headings", height=8)
        self.cluster_stats_tree.heading("#0", text="Cluster")
        self.cluster_stats_tree.heading("Count", text="Count")
        self.cluster_stats_tree.heading("Avg Price", text="Avg Price")
        self.cluster_stats_tree.heading("Avg Cost", text="Avg Cost")
        self.cluster_stats_tree.heading("Avg Units", text="Avg Units")
        self.cluster_stats_tree.heading("Avg Profit", text="Avg Profit")
        self.cluster_stats_tree.heading("Avg Promo", text="Avg Promo")
        
        for col in ["Count", "Avg Price", "Avg Cost", "Avg Units", "Avg Profit", "Avg Promo"]:
            self.cluster_stats_tree.column(col, width=80)
        
        self.cluster_stats_tree.pack(fill=tk.BOTH, expand=True)
        
        # Insights frame
        insights_frame = ttk.LabelFrame(right_frame, text="Business Insights", padding=5)
        insights_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.insights_text = tk.Text(insights_frame, height=15, width=50, wrap=tk.WORD)
        self.insights_text.pack(fill=tk.BOTH, expand=True)
        
        scrollbar_insights = ttk.Scrollbar(insights_frame, orient="vertical", command=self.insights_text.yview)
        self.insights_text.configure(yscrollcommand=scrollbar_insights.set)
        scrollbar_insights.pack(side=tk.RIGHT, fill=tk.Y)
    
    def update_feature_combos(self):
        """Update feature combo boxes with available features."""
        if self.processed_data is None:
            return
        
        numeric_cols = self.processed_data.select_dtypes(include=[np.number]).columns.tolist()
        self.feature1_combo['values'] = numeric_cols
        self.feature2_combo['values'] = numeric_cols
        
        # Set defaults
        if 'price' in numeric_cols:
            self.feature1_var.set('price')
        elif numeric_cols:
            self.feature1_var.set(numeric_cols[0])
        
        if 'units_sold' in numeric_cols:
            self.feature2_var.set('units_sold')
        elif len(numeric_cols) > 1:
            self.feature2_var.set(numeric_cols[1])
    
    def run_elbow_method(self):
        """Run elbow method to find optimal k."""
        if self.processed_data is None:
            messagebox.showwarning("Warning", "Please load data first.")
            return
        
        try:
            self.update_status("Running elbow method...")
            
            # Get numeric features for clustering
            numeric_cols = self.processed_data.select_dtypes(include=[np.number]).columns.tolist()
            X = self.processed_data[numeric_cols].values
            
            # Run elbow method
            k_range = list(range(2, 9))
            k_values, wcss_values = kmeans.elbow_method(X, k_range, init_method='kmeans++', random_state=42)
            
            # Store results
            self.clustering_results['elbow'] = {'k_values': k_values, 'wcss_values': wcss_values}
            
            # Display elbow curve
            self.display_elbow_curve(k_values, wcss_values)
            
            self.update_status("Elbow method completed.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run elbow method:\n{str(e)}")
            self.update_status("Error in elbow method")
    
    def display_elbow_curve(self, k_values, wcss_values):
        """Display elbow curve plot."""
        # Clear previous plot
        if self.elbow_canvas:
            self.elbow_canvas.get_tk_widget().destroy()
        
        # Create new plot
        fig = visualization.create_elbow_curve(k_values, wcss_values)
        
        self.elbow_canvas = FigureCanvasTkAgg(fig, self.elbow_frame)
        self.elbow_canvas.draw()
        self.elbow_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def run_clustering(self):
        """Run K-means clustering with selected k."""
        if self.processed_data is None:
            messagebox.showwarning("Warning", "Please load data first.")
            return
        
        try:
            k = self.k_var.get()
            self.update_status(f"Running K-means clustering with k={k}...")
            
            # Get selected features or all numeric
            feature1 = self.feature1_var.get()
            feature2 = self.feature2_var.get()
            
            if not feature1 or not feature2:
                # Use all numeric features
                numeric_cols = self.processed_data.select_dtypes(include=[np.number]).columns.tolist()
                X = self.processed_data[numeric_cols].values
                feature1_idx = 0
                feature2_idx = 1 if len(numeric_cols) > 1 else 0
                feature1_name = numeric_cols[0]
                feature2_name = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]
            else:
                # Use selected features for visualization, but cluster on all numeric
                numeric_cols = self.processed_data.select_dtypes(include=[np.number]).columns.tolist()
                X = self.processed_data[numeric_cols].values
                feature1_idx = numeric_cols.index(feature1)
                feature2_idx = numeric_cols.index(feature2)
                feature1_name = feature1
                feature2_name = feature2
            
            # Run K-means
            labels, centroids, n_iters = kmeans.kmeans(X, k, init_method='kmeans++', random_state=42)
            
            # Store results
            self.clustering_results['labels'] = labels
            self.clustering_results['centroids'] = centroids
            self.clustering_results['k'] = k
            self.clustering_results['feature1'] = feature1_name
            self.clustering_results['feature2'] = feature2_name
            self.clustering_results['feature1_idx'] = feature1_idx
            self.clustering_results['feature2_idx'] = feature2_idx
            
            # Display cluster visualization
            self.display_cluster_scatter(X, labels, centroids, feature1_name, feature2_name, feature1_idx, feature2_idx)
            
            # Update cluster statistics
            self.update_cluster_statistics(labels)
            
            # Generate insights
            self.generate_cluster_insights(labels, centroids)
            
            self.update_status(f"Clustering completed with k={k} in {n_iters} iterations.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run clustering:\n{str(e)}")
            self.update_status("Error in clustering")
    
    def display_cluster_scatter(self, X, labels, centroids, feature1, feature2, feature1_idx, feature2_idx):
        """Display cluster scatter plot."""
        # Clear previous plot
        if self.scatter_canvas:
            self.scatter_canvas.get_tk_widget().destroy()
        
        # Create new plot
        fig = visualization.create_cluster_scatter(X, labels, centroids, feature1, feature2, feature1_idx, feature2_idx)
        
        self.scatter_canvas = FigureCanvasTkAgg(fig, self.scatter_frame)
        self.scatter_canvas.draw()
        self.scatter_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def update_cluster_statistics(self, labels):
        """Update cluster statistics table."""
        # Clear existing items
        for item in self.cluster_stats_tree.get_children():
            self.cluster_stats_tree.delete(item)
        
        if self.raw_data is None:
            return
        
        # Get unique clusters
        unique_labels = np.unique(labels)
        
        # Calculate statistics for each cluster
        for cluster_id in unique_labels:
            cluster_mask = labels == cluster_id
            cluster_data = self.raw_data[cluster_mask]
            
            count = len(cluster_data)
            
            # Calculate averages
            stats = {}
            numeric_cols = ['price', 'cost', 'units_sold', 'profit', 'promotion_frequency']
            for col in numeric_cols:
                if col in cluster_data.columns:
                    stats[col] = cluster_data[col].mean()
                else:
                    stats[col] = 0.0
            
            self.cluster_stats_tree.insert("", tk.END, text=f"Cluster {cluster_id}", values=(
                count,
                f"{stats.get('price', 0):.2f}",
                f"{stats.get('cost', 0):.2f}",
                f"{stats.get('units_sold', 0):.2f}",
                f"{stats.get('profit', 0):.2f}",
                f"{stats.get('promotion_frequency', 0):.2f}"
            ))
    
    def generate_cluster_insights(self, labels, centroids):
        """Generate business insights from clustering results."""
        self.insights_text.delete(1.0, tk.END)
        
        if self.raw_data is None:
            return
        
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        insights = f"Clustering Analysis Results (k={n_clusters})\n"
        insights += "=" * 50 + "\n\n"
        
        # Analyze each cluster
        for cluster_id in unique_labels:
            cluster_mask = labels == cluster_id
            cluster_data = self.raw_data[cluster_mask]
            
            insights += f"Cluster {cluster_id}:\n"
            insights += f"  • Size: {len(cluster_data)} products ({len(cluster_data)/len(self.raw_data)*100:.1f}%)\n"
            
            if 'price' in cluster_data.columns:
                avg_price = cluster_data['price'].mean()
                insights += f"  • Average Price: ${avg_price:.2f}\n"
            
            if 'units_sold' in cluster_data.columns:
                avg_units = cluster_data['units_sold'].mean()
                insights += f"  • Average Units Sold: {avg_units:.1f}\n"
            
            if 'profit' in cluster_data.columns:
                avg_profit = cluster_data['profit'].mean()
                insights += f"  • Average Profit: ${avg_profit:.2f}\n"
            
            insights += "\n"
        
        # Overall insights
        insights += "Key Insights:\n"
        insights += "• Clusters help identify product segments with similar characteristics\n"
        insights += "• Use clusters to develop targeted marketing strategies\n"
        insights += "• High-profit clusters may indicate premium product segments\n"
        insights += "• Low-price, high-volume clusters may represent value products\n"
        
        self.insights_text.insert(tk.END, insights)
    
    # ==================== REGRESSION TAB ====================
    
    def create_regression_tab(self):
        """Create Regression Analysis tab."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Regression Analysis")
        
        # Control frame
        control_frame = ttk.Frame(tab)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(control_frame, text="Polynomial Degree:").pack(side=tk.LEFT, padx=5)
        self.degree_var = tk.IntVar(value=2)
        degree_spinbox = ttk.Spinbox(control_frame, from_=1, to=5, textvariable=self.degree_var, width=5)
        degree_spinbox.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Train Models", command=self.train_regression_models).pack(side=tk.LEFT, padx=5)
        
        # Main content
        content_frame = ttk.Frame(tab)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Left: Model comparison and plot
        left_frame = ttk.Frame(content_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Model comparison table
        comparison_frame = ttk.LabelFrame(left_frame, text="Model Comparison", padding=5)
        comparison_frame.pack(fill=tk.BOTH, expand=False, padx=5, pady=5)
        
        self.model_comparison_tree = ttk.Treeview(comparison_frame, columns=("MSE", "MAE"), show="headings", height=3)
        self.model_comparison_tree.heading("#0", text="Model")
        self.model_comparison_tree.heading("MSE", text="MSE")
        self.model_comparison_tree.heading("MAE", text="MAE")
        self.model_comparison_tree.column("#0", width=150)
        self.model_comparison_tree.column("MSE", width=150)
        self.model_comparison_tree.column("MAE", width=150)
        self.model_comparison_tree.pack(fill=tk.BOTH, expand=True)
        
        # Regression plot frame
        self.regression_plot_frame = ttk.LabelFrame(left_frame, text="Best Model: Actual vs Predicted", padding=5)
        self.regression_plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.regression_canvas = None
        
        # Right: Discussion
        right_frame = ttk.Frame(content_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=5)
        
        discussion_frame = ttk.LabelFrame(right_frame, text="Model Analysis", padding=5)
        discussion_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.discussion_text = tk.Text(discussion_frame, height=25, width=50, wrap=tk.WORD)
        self.discussion_text.pack(fill=tk.BOTH, expand=True)
        
        scrollbar_discussion = ttk.Scrollbar(discussion_frame, orient="vertical", command=self.discussion_text.yview)
        self.discussion_text.configure(yscrollcommand=scrollbar_discussion.set)
        scrollbar_discussion.pack(side=tk.RIGHT, fill=tk.Y)
    
    def train_regression_models(self):
        """Train and evaluate regression models."""
        if self.processed_data is None:
            messagebox.showwarning("Warning", "Please load data first.")
            return
        
        try:
            degree = self.degree_var.get()
            self.update_status("Training regression models...")
            
            # Prepare data
            numeric_cols = self.processed_data.select_dtypes(include=[np.number]).columns.tolist()
            
            # Target variable: profit
            if 'profit' not in self.raw_data.columns:
                messagebox.showerror("Error", "Target variable 'profit' not found in dataset.")
                return
            
            # Features: all numeric except profit
            feature_cols = [col for col in numeric_cols if col != 'profit']
            if not feature_cols:
                messagebox.showerror("Error", "No feature columns available.")
                return
            
            # Use original data (not normalized) for regression
            X = self.raw_data[feature_cols].values
            y = self.raw_data['profit'].values
            
            # Handle any remaining NaN values
            mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X = X[mask]
            y = y[mask]
            
            if len(X) == 0:
                messagebox.showerror("Error", "No valid data after cleaning.")
                return
            
            # Train/test split (80-20)
            X_train, X_test, y_train, y_test = regression.train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train Linear Regression
            linear_coef = regression.linear_regression_fit(X_train, y_train)
            y_pred_linear = regression.linear_regression_predict(X_test, linear_coef)
            mse_linear, mae_linear = regression.evaluate_model(y_test, y_pred_linear)
            
            # Train Polynomial Regression
            poly_coef, X_train_poly = regression.polynomial_regression_fit(X_train, y_train, degree)
            y_pred_poly = regression.polynomial_regression_predict(X_test, poly_coef, degree)
            mse_poly, mae_poly = regression.evaluate_model(y_test, y_pred_poly)
            
            # Store results
            self.regression_results = {
                'linear': {'mse': mse_linear, 'mae': mae_linear, 'y_test': y_test, 'y_pred': y_pred_linear},
                'polynomial': {'mse': mse_poly, 'mae': mae_poly, 'y_test': y_test, 'y_pred': y_pred_poly, 'degree': degree}
            }
            
            # Update comparison table
            self.update_model_comparison()
            
            # Display best model plot
            self.display_regression_plot()
            
            # Generate discussion
            self.generate_regression_discussion()
            
            self.update_status("Regression models trained successfully.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to train models:\n{str(e)}")
            self.update_status("Error in regression training")
    
    def update_model_comparison(self):
        """Update model comparison table."""
        # Clear existing items
        for item in self.model_comparison_tree.get_children():
            self.model_comparison_tree.delete(item)
        
        if not self.regression_results:
            return
        
        linear = self.regression_results['linear']
        poly = self.regression_results['polynomial']
        
        self.model_comparison_tree.insert("", tk.END, text="Linear Regression", values=(
            f"{linear['mse']:.2f}", f"{linear['mae']:.2f}"
        ))
        
        self.model_comparison_tree.insert("", tk.END, text=f"Polynomial Regression (degree={poly['degree']})", values=(
            f"{poly['mse']:.2f}", f"{poly['mae']:.2f}"
        ))
    
    def display_regression_plot(self):
        """Display regression scatter plot for best model."""
        if not self.regression_results:
            return
        
        # Determine best model (lower MSE is better)
        linear = self.regression_results['linear']
        poly = self.regression_results['polynomial']
        
        if linear['mse'] < poly['mse']:
            best_model = 'linear'
            model_name = "Linear Regression"
        else:
            best_model = 'polynomial'
            model_name = f"Polynomial Regression (degree={poly['degree']})"
        
        # Clear previous plot
        if self.regression_canvas:
            self.regression_canvas.get_tk_widget().destroy()
        
        # Create plot
        results = self.regression_results[best_model]
        fig = visualization.create_regression_scatter(results['y_test'], results['y_pred'], model_name)
        
        self.regression_canvas = FigureCanvasTkAgg(fig, self.regression_plot_frame)
        self.regression_canvas.draw()
        self.regression_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def generate_regression_discussion(self):
        """Generate discussion about model performance."""
        self.discussion_text.delete(1.0, tk.END)
        
        if not self.regression_results:
            return
        
        linear = self.regression_results['linear']
        poly = self.regression_results['polynomial']
        
        discussion = "Model Performance Analysis\n"
        discussion += "=" * 50 + "\n\n"
        
        # Compare models
        if linear['mse'] < poly['mse']:
            best_model = "Linear Regression"
            discussion += f"Best Model: {best_model}\n"
            discussion += f"  • MSE: {linear['mse']:.2f}\n"
            discussion += f"  • MAE: {linear['mae']:.2f}\n\n"
            discussion += "Why Linear Regression performed better:\n"
            discussion += "  • Simpler model with fewer parameters\n"
            discussion += "  • Less prone to overfitting\n"
            discussion += "  • Relationship may be approximately linear\n\n"
        else:
            best_model = f"Polynomial Regression (degree={poly['degree']})"
            discussion += f"Best Model: {best_model}\n"
            discussion += f"  • MSE: {poly['mse']:.2f}\n"
            discussion += f"  • MAE: {poly['mae']:.2f}\n\n"
            discussion += "Why Polynomial Regression performed better:\n"
            discussion += "  • Captures non-linear relationships\n"
            discussion += "  • Better fit to complex data patterns\n\n"
        
        # Overfitting/Underfitting discussion
        discussion += "Overfitting/Underfitting Analysis:\n"
        discussion += "-" * 30 + "\n"
        
        if poly['mse'] < linear['mse']:
            if poly['degree'] >= 3:
                discussion += "• Polynomial model may be overfitting if degree is high\n"
                discussion += "• Consider cross-validation to verify generalization\n"
            else:
                discussion += "• Polynomial model shows good fit without overfitting\n"
        else:
            discussion += "• Linear model is simpler and more generalizable\n"
            discussion += "• Lower risk of overfitting\n"
        
        discussion += "\nRecommendations:\n"
        discussion += "• Use the best model for profit prediction\n"
        discussion += "• Monitor model performance on new data\n"
        discussion += "• Consider feature engineering to improve predictions\n"
        
        self.discussion_text.insert(tk.END, discussion)
    
    # ==================== SUMMARY TAB ====================
    
    def create_summary_tab(self):
        """Create Summary tab."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Summary")
        
        # Create scrollable frame
        canvas = tk.Canvas(tab)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Key Findings
        findings_frame = ttk.LabelFrame(scrollable_frame, text="Key Findings", padding=10)
        findings_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.findings_text = tk.Text(findings_frame, height=8, wrap=tk.WORD)
        self.findings_text.pack(fill=tk.BOTH, expand=True)
        
        # Business Recommendations
        recommendations_frame = ttk.LabelFrame(scrollable_frame, text="Business Recommendations", padding=10)
        recommendations_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.recommendations_text = tk.Text(recommendations_frame, height=8, wrap=tk.WORD)
        self.recommendations_text.pack(fill=tk.BOTH, expand=True)
        
        # Limitations
        limitations_frame = ttk.LabelFrame(scrollable_frame, text="Limitations", padding=10)
        limitations_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.limitations_text = tk.Text(limitations_frame, height=6, wrap=tk.WORD)
        self.limitations_text.pack(fill=tk.BOTH, expand=True)
        
        # Potential Improvements
        improvements_frame = ttk.LabelFrame(scrollable_frame, text="Potential Improvements", padding=10)
        improvements_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.improvements_text = tk.Text(improvements_frame, height=6, wrap=tk.WORD)
        self.improvements_text.pack(fill=tk.BOTH, expand=True)
        
        # Button to generate summary
        ttk.Button(scrollable_frame, text="Generate Summary", command=self.generate_summary).pack(pady=10)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def generate_summary(self):
        """Generate comprehensive summary."""
        # Key Findings
        findings = "Key Findings from Analysis\n"
        findings += "=" * 50 + "\n\n"
        
        if self.clustering_results:
            k = self.clustering_results.get('k', 'N/A')
            findings += f"1. Clustering Analysis:\n"
            findings += f"   • Optimal number of clusters: {k}\n"
            findings += f"   • Products grouped into distinct segments\n"
            findings += f"   • Each cluster shows unique characteristics\n\n"
        
        if self.regression_results:
            linear = self.regression_results['linear']
            poly = self.regression_results['polynomial']
            best = 'Linear' if linear['mse'] < poly['mse'] else f"Polynomial (degree={poly['degree']})"
            findings += f"2. Regression Analysis:\n"
            findings += f"   • Best model: {best}\n"
            findings += f"   • Linear MSE: {linear['mse']:.2f}, MAE: {linear['mae']:.2f}\n"
            findings += f"   • Polynomial MSE: {poly['mse']:.2f}, MAE: {poly['mae']:.2f}\n\n"
        
        findings += "3. Data Quality:\n"
        findings += f"   • Processed {self.preprocessing_pipeline.processed_data_shape[0]} products\n"
        findings += f"   • Applied {self.preprocessing_pipeline.normalization_method} normalization\n"
        findings += f"   • Detected and handled outliers using {self.preprocessing_pipeline.outliers_method}\n"
        
        self.findings_text.delete(1.0, tk.END)
        self.findings_text.insert(tk.END, findings)
        
        # Business Recommendations
        recommendations = "Business Recommendations\n"
        recommendations += "=" * 50 + "\n\n"
        recommendations += "1. Product Segmentation:\n"
        recommendations += "   • Use clustering results to identify product categories\n"
        recommendations += "   • Develop targeted marketing strategies for each cluster\n"
        recommendations += "   • Optimize pricing based on cluster characteristics\n\n"
        recommendations += "2. Profit Optimization:\n"
        recommendations += "   • Use regression models to predict profit for new products\n"
        recommendations += "   • Focus on features that drive profitability\n"
        recommendations += "   • Monitor actual vs predicted profit to refine models\n\n"
        recommendations += "3. Inventory Management:\n"
        recommendations += "   • Analyze units_sold patterns across clusters\n"
        recommendations += "   • Adjust stock levels based on cluster performance\n"
        recommendations += "   • Consider promotion strategies for low-performing clusters\n"
        
        self.recommendations_text.delete(1.0, tk.END)
        self.recommendations_text.insert(tk.END, recommendations)
        
        # Limitations
        limitations = "Limitations of Analysis\n"
        limitations += "=" * 50 + "\n\n"
        limitations += "1. Data Limitations:\n"
        limitations += "   • Analysis based on available historical data only\n"
        limitations += "   • May not capture external factors (seasonality, competition)\n"
        limitations += "   • Missing values handled with imputation (may introduce bias)\n\n"
        limitations += "2. Model Limitations:\n"
        limitations += "   • K-means assumes spherical clusters (may not fit all data)\n"
        limitations += "   • Regression models assume linear/polynomial relationships\n"
        limitations += "   • No time-series analysis (temporal patterns not considered)\n\n"
        limitations += "3. Scope Limitations:\n"
        limitations += "   • Analysis limited to available features\n"
        limitations += "   • No external validation data\n"
        limitations += "   • Business context not fully captured in data\n"
        
        self.limitations_text.delete(1.0, tk.END)
        self.limitations_text.insert(tk.END, limitations)
        
        # Potential Improvements
        improvements = "Potential Improvements\n"
        improvements += "=" * 50 + "\n\n"
        improvements += "1. Data Enhancements:\n"
        improvements += "   • Collect more features (customer demographics, seasonality)\n"
        improvements += "   • Include time-series data for trend analysis\n"
        improvements += "   • Gather external data (market conditions, competitor prices)\n\n"
        improvements += "2. Model Improvements:\n"
        improvements += "   • Try other clustering algorithms (DBSCAN, hierarchical)\n"
        improvements += "   • Implement cross-validation for regression models\n"
        improvements += "   • Explore ensemble methods for better predictions\n"
        improvements += "   • Add regularization to prevent overfitting\n\n"
        improvements += "3. Analysis Enhancements:\n"
        improvements += "   • Perform feature importance analysis\n"
        improvements += "   • Add interactive visualizations\n"
        improvements += "   • Implement real-time prediction capabilities\n"
        improvements += "   • Create automated reporting system\n"
        
        self.improvements_text.delete(1.0, tk.END)
        self.improvements_text.insert(tk.END, improvements)


def main():
    """Main entry point."""
    root = tk.Tk()
    app = Application(root)
    root.mainloop()


if __name__ == "__main__":
    main()

