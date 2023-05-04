from sklearn.cluster import KMeans

def k_means_clustering(dataset, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(dataset)
    labels = kmeans.labels_
    return labels
