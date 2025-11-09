# your code here
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from time import time

# Exercise 1: kmeans function
def kmeans(X, k):
    """
    Performs k-means clustering on a numerical NumPy array X.
    
    Parameters:
    - X: numpy array of shape (n_samples, n_features)
    - k: number of clusters
    
    Returns:
    - tuple (centroids, labels) where:
        - centroids: array of shape (k, n_features)
        - labels: array of shape (n_samples,)
    """
    # Create and fit KMeans model
    kmeans_model = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_model.fit(X)
    
    # Get centroids and labels
    centroids = kmeans_model.cluster_centers_
    labels = kmeans_model.labels_
    
    return (centroids, labels)


# Exercise 2: Load diamonds dataset and prepare data
# Load the diamonds dataset
diamonds = sns.load_dataset('diamonds')

# Identify and extract only numerical columns
# The numerical columns in diamonds are: carat, depth, table, price, x, y, z
diamonds_numeric = diamonds.select_dtypes(include=[np.number])


def kmeans_diamonds(n, k):
    """
    Runs kmeans on the first n rows of the numeric diamonds dataset.
    
    Parameters:
    - n: number of rows to use from diamonds dataset
    - k: number of clusters
    
    Returns:
    - tuple (centroids, labels) from kmeans function
    """
    # Get first n rows of numeric diamonds data
    X = diamonds_numeric.iloc[:n].values
    
    # Run kmeans
    centroids, labels = kmeans(X, k)
    
    return (centroids, labels)


# Exercise 3: Timer function
def kmeans_timer(n, k, n_iter=5):
    """
    Runs kmeans_diamonds(n, k) n_iter times and returns average runtime.
    
    Parameters:
    - n: number of rows for kmeans_diamonds
    - k: number of clusters for kmeans_diamonds
    - n_iter: number of iterations to run (default=5)
    
    Returns:
    - average time in seconds across all runs
    """
    times = []
    
    for _ in range(n_iter):
        start = time()
        _ = kmeans_diamonds(n, k)
        runtime = time() - start
        times.append(runtime)
    
    # Return average time
    return np.mean(times)