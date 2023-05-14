import numpy as np
from tslearn.clustering import TimeSeriesKMeans

def ts_k_means_clustering(data: np.ndarray, n_clusters:int=3, max_iter: int=50, random_state:int=42, verbose: int=0, metric:str='euclidean'):
    # Check if data is 2D, if so, reshape it to 3D
    if data.ndim == 2:
        data = data.reshape((data.shape[0], data.shape[1], 1))

    # Create a TimeSeriesKMeans instance
    model = TimeSeriesKMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=random_state, verbose=verbose, metric=metric)
    
    # Fit the model to the data and predict cluster assignments
    cluster_assignments = model.fit_predict(data)
    
    return cluster_assignments
