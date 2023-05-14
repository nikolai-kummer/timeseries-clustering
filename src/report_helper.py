import base64
import matplotlib.pyplot as plt
import numpy as np
import os

from io import BytesIO
from jinja2 import Environment, FileSystemLoader
from sklearn.manifold import TSNE

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
        n_cluster_samples = sum(cluster_assignments == cluster_label)
        fig.suptitle(f"Cluster {cluster_label} Sample Data, n = {n_cluster_samples}", fontsize=16)
        axes[0].set_ylabel("Value", fontsize=14)

        base64_images[cluster_label] = fig_to_base64(fig)

    return base64_images

def generate_pca_plot_image(X_data, cluster_cols, pca_model) -> plt.figure:
    # Plot the 2D representation of the clustered data and the explained variance ratio
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Subplot for PCA clustering
    scatter = ax1.scatter(X_data[:, 0], X_data[:, 1], c=cluster_cols)
    ax1.set_title("PCA Clustering")
    ax1.set_xlabel("PCA Component 1")
    ax1.set_ylabel("PCA Component 2")
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label("Cluster")

    # Subplot for explained variance ratio
    explained_variance_ratio = pca_model.explained_variance_ratio_
    n_components = np.arange(1, len(explained_variance_ratio) + 1)
    ax2.bar(n_components, explained_variance_ratio)
    ax2.set_title("Explained Variance Ratio")
    ax2.set_xlabel("PCA Components")
    ax2.set_ylabel("Explained Variance Ratio")
    ax2.set_xticks(n_components)

    # Adjust the layout
    plt.tight_layout()

    return fig_to_base64(fig)

def generate_tsne_plot_image(X_data, cluster_labels=None, perplexity=30) -> plt.figure:
    # Perform t-SNE on the normalized data
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    data_tsne = tsne.fit_transform(X_data)

    # Plot the t-SNE results
    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(data_tsne[:, 0], data_tsne[:, 1], c=cluster_labels)
    ax.set_title("t-SNE Clustering")
    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")

    # Add a colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Cluster")

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