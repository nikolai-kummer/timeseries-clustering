import numpy as np
from sklearn.preprocessing import MinMaxScaler

def normalize(dataset):
    scaler = MinMaxScaler()
    preprocessed_data = scaler.fit_transform(dataset)
    return preprocessed_data