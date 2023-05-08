import json
import numpy  as np
import pandas as pd


class BaseGenerator:
    def __init__(self, **kwargs):
        self.params = kwargs

    def generate(self):
        raise NotImplementedError()

class SineWaveGenerator(BaseGenerator):
    def generate(self) -> np.ndarray:
        length = self.params.get('length', 1000)
        amplitude = self.params.get('amplitude', 1)
        frequency = self.params.get('frequency', 0.01)
        phase = self.params.get('phase', 0)
        time = np.arange(length)
        sine_wave = amplitude * np.sin(2 * np.pi * frequency * time + phase)
        return sine_wave

class ConstantGenerator(BaseGenerator):
    def generate(self) -> np.ndarray:
        length = self.params.get('length', 1000)
        value = self.params.get('value', 0)
        constant_series = np.full(length, value)
        return constant_series

class CSVGenerator(BaseGenerator):
    def generate(self) -> np.ndarray:
        file_path = self.params['file_path']
        column = self.params['column_name']
        num_instances = self.params.get('num_instances', 1e6)
        
        data = pd.read_csv(file_path)
        
        # Extract the specified column and parse JSON strings into lists
        column_data = data[column].apply(json.loads)
        
        # Convert the lists into a 2D NumPy array
        time_series_array = np.array(column_data.to_list())
        
        # If the CSV file has more time series than needed, randomly select the specified number of instances
        if time_series_array.shape[0] > num_instances:
            indices = np.random.choice(time_series_array.shape[0], num_instances, replace=False)
            time_series_array = time_series_array[indices]

        return time_series_array