import base64
import matplotlib.pyplot as plt
import numpy as np
import os

from io import BytesIO
from jinja2 import Environment, FileSystemLoader


def fig_to_base64(fig: plt.figure, close_fig: bool=True):
    buffer = BytesIO()
    fig.savefig(buffer, format='png')
    if close_fig:
        plt.close(fig)
    base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return base64_image

def generate_initial_sample_plots_base64(X_train: np.ndarray, num_samples:int=15, num_panels:int=5):
    base64_images = []
    
    for sample_idx in range(min(num_samples, len(X_train))):
        fig, ax = plt.subplots(figsize=(5, 1), dpi=100)
        ax.plot(X_train[sample_idx])
        ax.axis('off')
        base64_images.append(fig_to_base64(fig))

    return base64_images


def generate_cluster_sample_plots_base64(cluster_assignments, X_train, n_panels: int=5):
    base64_images = {}

    unique_cluster_assignments = np.unique(cluster_assignments)
    for cluster_label in unique_cluster_assignments:
        cluster_samples = X_train[cluster_assignments == cluster_label][:n_panels]

        # Create a 1x5 grid of subplots with an aspect ratio of 1:5
        fig, axes = plt.subplots(1, n_panels, figsize=(4*n_panels, n_panels), sharey=True)
        fig.subplots_adjust(hspace=0.3, wspace=0.0)

        # Plot each sample in a separate subplot
        for i, sample in enumerate(cluster_samples):
            axes[i].plot(sample)
            axes[i].set_title(f"Sample {i+1}")

        # If there are fewer than 5 samples, leave the remaining subplots empty
        for i in range(len(cluster_samples), n_panels):
            axes[i].set_axis_off()

        # Set common title, xlabel, and ylabel for all subplots
        fig.suptitle(f"Cluster {cluster_label} Sample Data", fontsize=16)
        axes[0].set_ylabel("Value", fontsize=14)

        base64_images[cluster_label] = fig_to_base64(fig)

    return base64_images

def generate_pca_plot_image(X_data, cluster_cols) -> plt.figure:
    # Plot the 2D representation of the clustered data
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(X_data[:, 0], X_data[:, 1], c=cluster_cols)
    ax.set_title("PCA Clustering")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    return fig_to_base64(fig)
    

def render_html_report(initial_sample_data, clustering_algorithms, output_file):
    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template('src/report_templates/report_template.html')

    data = {
        'initial_sample_data': initial_sample_data,
        'clustering_algorithms': clustering_algorithms
    }

    with open(output_file, 'w') as f:
        f.write(template.render(data))