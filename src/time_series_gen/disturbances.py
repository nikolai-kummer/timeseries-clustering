import numpy as np

class Disturbance:
    def __init__(self, **kwargs):
        self.params = kwargs

    def apply(self, series):
        raise NotImplementedError()

class NoiseDisturbance(Disturbance):
    def apply(self, series):
        noise_std = self.params.get('noise_std', 0.1)
        noise = np.random.normal(0, noise_std, len(series))
        return series + noise

class SignalLossDisturbance(Disturbance):
    def apply(self, series):
        start = self.params.get('start', 0)
        end = self.params.get('end', len(series))
        series[start:end] = 0
        return series
