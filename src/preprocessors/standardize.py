import numpy as np


def standardize(dataset: np.ndarray) -> np.ndarray:
    mean_value = np.mean(dataset)
    std_dev = np.std(dataset)
    return (dataset - mean_value) / std_dev