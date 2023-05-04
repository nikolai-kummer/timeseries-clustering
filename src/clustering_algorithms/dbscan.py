from sklearn.cluster import DBSCAN

def dbscan_clustering(data, eps=0.5, min_samples=5):
    """
    Applies DBSCAN clustering algorithm to the input data.

    Args:
        data (numpy.ndarray): The input data for clustering.
        eps (float): The maximum distance between two samples for them to be considered as in the same neighborhood.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.

    Returns:
        numpy.ndarray: The cluster labels for each input data point.
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(data)
    return dbscan.labels_
