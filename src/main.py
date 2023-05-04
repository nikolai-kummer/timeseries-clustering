import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple

from report_helper import render_html_report, generate_cluster_sample_plots_base64, generate_pca_plot_image, generate_initial_sample_plots_base64
from config_parser import load_config, apply_preprocessor, apply_clustering_algorithm
from metrics.metrics import metrics_registry
from time_series_gen.time_series_gen import generate_time_series


def apply_preprocessors(time_series: np.ndarray, config) -> np.ndarray:
    # Load the preprocessors 
    preprocessed_time_series = time_series.copy()

    for preprocessor_config in config['preprocessors']:
        print(f"Applying {preprocessor_config['name']} preprocessor...")        
        preprocessed_time_series = apply_preprocessor(preprocessor_config, preprocessed_time_series)

    return preprocessed_time_series


def generate_time_series_dict(global_params, config)->Dict[int, np.ndarray]:
    # Generate the time series from the configuration
    generated_time_series = {}
    time_series_index = 0

    length = global_params.get('length', 100)
    for time_series_index, time_series_config in enumerate(config['time_series']):
        num_instances = time_series_config.get('num_instances', 1)
        generated_time_series_array = np.empty((num_instances, length), dtype=float)

        for series_idx in range(num_instances):
            time_series = generate_time_series(global_params, time_series_config)
            generated_time_series_array[series_idx, :] = time_series.ravel()
            
        generated_time_series[time_series_index] = generated_time_series_array
    return generated_time_series

def generate_pca_plot(X_data, cluster_labels = None):
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_data)
    return generate_pca_plot_image(X_train_pca, cluster_labels)
    


def split_train_test(X_data, y_data, global_params):
    # Split the data into train and test sets
    test_size = global_params.get('test_size', 0.2)
    seed = global_params.get('seed', 42)
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=test_size, random_state=seed)

    return X_train, X_test, y_train, y_test


def stack_generated_time_series(generated_time_series) -> Tuple[np.ndarray, np.ndarray]:
    data_arrays = [time_series for time_series in generated_time_series.values()]
    label_arrays = [np.full(generated_time_series[label].shape[0], label) for label in generated_time_series.keys()]

    stacked_data = np.vstack(data_arrays)
    label_vector = np.hstack(label_arrays)

    return stacked_data, label_vector


if __name__ == '__main__':
    config_file = 'config/config_example.yaml'
    config = load_config(config_file)
    global_params = config.get('global_params', {})

    generated_time_series = generate_time_series_dict(global_params, config)
    x_data, y_data = stack_generated_time_series(generated_time_series)
    x_data_preprocessed = apply_preprocessors(x_data, config)
    X_train, X_test, y_train, y_test = split_train_test(x_data_preprocessed, y_data, global_params)

    # Load the clustering algorithms from the configuration
    clustering_algorithms = config['clustering_algorithms']

    # Define initial plotting data
    initial_samples_data = {
        'total_samples': len(X_train),
        'initial_sample_plots': generate_initial_sample_plots_base64(X_train)
    }

    # Perform clustering on the train data
    clustering_algorithms_data = []
    for clustering_algorithm_config in clustering_algorithms:
        algorithm_name = clustering_algorithm_config['name']
        print(f"Applying {algorithm_name} clustering algorithm...")

        cluster_assignments_train = apply_clustering_algorithm(clustering_algorithm_config, X_train)
        print(f"Cluster assignments (train data): {cluster_assignments_train}")

        # generate_pca_plot(X_train, cluster_assignments_train)

        # Calculate and display the specified metrics
        for metric_config in config['metrics']:
            metric_name = metric_config['name']
            metric_func = metrics_registry.get(metric_name)
            if metric_func:
                score = metric_func(X_train, y_train, cluster_assignments_train)
                print(f"{algorithm_name} {metric_name}: {score}")
            else:
                print(f"Metric '{metric_name}' not found.")

        # Collect data for the current clustering algorithm
        cluster_sizes = np.bincount(cluster_assignments_train+1)
        cluster_base64_images = generate_cluster_sample_plots_base64(cluster_assignments_train, X_train)

        clustering_algorithms_data.append({
            'name': algorithm_name,
            'cluster_assignments': cluster_assignments_train,
            'cluster_sizes': cluster_sizes,
            'cluster_base64_images': cluster_base64_images,
            'pca_base64_image': generate_pca_plot(X_train, cluster_assignments_train)
        })

        # Render the HTML report
        render_html_report(initial_samples_data, clustering_algorithms_data, 'report.html')