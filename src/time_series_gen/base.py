import numpy as np

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
