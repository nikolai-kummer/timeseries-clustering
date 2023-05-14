import numpy as np
from sklearn.decomposition import PCA

def apply_pca(dataset: np.ndarray, n_components: int =10) -> np.ndarray:
    pca = PCA(n_components=n_components)
    transformed_dataset = pca.fit_transform(dataset)
    return transformed_dataset
