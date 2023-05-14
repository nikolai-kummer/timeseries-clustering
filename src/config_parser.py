import importlib
import yaml


def import_function(module_name, function_name):
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    return function

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def apply_preprocessor(preprocessor_config, dataset):
    module = preprocessor_config['module']
    function = preprocessor_config['function']
    params = preprocessor_config.get('params', {})
    preprocessor_function = import_function(module, function)
    return preprocessor_function(dataset, **params)

def apply_clustering_algorithm(clustering_algorithm_config, dataset):
    module = clustering_algorithm_config['module']
    function = clustering_algorithm_config['function']
    params = clustering_algorithm_config.get('params', {})
    clustering_function = import_function(module, function)
    return clustering_function(dataset, **params)
