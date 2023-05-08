import numpy as np

def normalize(dataset):
    min_value = np.min(dataset)
    max_value = np.max(dataset)

    preprocessed_data = (dataset - min_value) / (max_value - min_value)
    return preprocessed_data