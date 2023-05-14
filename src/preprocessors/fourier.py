import numpy as np

def fourier_transform(dataset: np.ndarray) -> np.ndarray:
    transformed_dataset = np.fft.fft(dataset)
    return np.abs(transformed_dataset)
