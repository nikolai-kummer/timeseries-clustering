from typing import Tuple

from time_series_gen.base import SineWaveGenerator, ConstantGenerator, CSVGenerator
from time_series_gen.transformations import ConstantOffsetTransformation, NonLinearTransformation
from time_series_gen.disturbances import NoiseDisturbance, SignalLossDisturbance


class Registry:
    def __init__(self):
        self._registry = {}

    def register(self, name, cls):
        self._registry[name] = cls

    def get(self, name):
        return self._registry.get(name)

base_generator_registry = Registry()
transformation_registry = Registry()
disturbance_registry = Registry()

base_generator_registry.register('sine_wave', SineWaveGenerator)
base_generator_registry.register('constant', ConstantGenerator)
base_generator_registry.register('from_csv', CSVGenerator)

transformation_registry.register('constant_offset', ConstantOffsetTransformation)
transformation_registry.register('nonlinear', NonLinearTransformation)

disturbance_registry.register('noise', NoiseDisturbance)
disturbance_registry.register('signal_loss', SignalLossDisturbance)


def generate_time_series(global_params, config):
    # Get the base generator class from the registry and create an instance
    base_generator_name = config['base']
    base_generator_cls = base_generator_registry.get(base_generator_name)
    merged_params = {**global_params, **config.get('params', {})}
    base_generator = base_generator_cls(**merged_params)

    # Generate the base time series
    base_series = base_generator.generate()

    # Apply transformations
    if 'transformations' in config:
        for transformation_config in config['transformations']:
            transformation_name = transformation_config['name']
            transformation_cls = transformation_registry.get(transformation_name)
            transformation = transformation_cls(**transformation_config.get('params', {}))
            base_series = transformation.apply(base_series)

    # Apply disturbances
    if 'disturbances' in config:
        for disturbance_config in config['disturbances']:
            disturbance_name = disturbance_config['name']
            disturbance_cls = disturbance_registry.get(disturbance_name)
            disturbance = disturbance_cls(**disturbance_config.get('params', {}))
            base_series = disturbance.apply(base_series)

    return base_series
