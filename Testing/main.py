from Testing.Clustering.Algoritmi.KMEANS import KMeansAnalysis
from Testing.Clustering.Algoritmi.DBSCAN import DBSCANAnalysis
from Testing.Clustering.Algoritmi.GerarchicoDivisivo import HDClusterAnalysis
from Testing.Dataset import random_points_generator as rpg

def generate_datasets():
    """
    Genera datasets per testare gli algoritmi di clustering.
    """
    datasets = {
        "Gaussian Clusters": rpg.generate_gaussian_clusters(n_clusters=3, n_points_per_cluster=100, cluster_spread=5),
        "Uniform Data": rpg.generate_uniform_data(n_points=300, x_range=(0, 100), y_range=(0, 100)),
        "Overlapping Clusters": rpg.generate_overlapping_clusters(n_clusters=3, n_points_per_cluster=100, overlap=10),
        "Non-Spherical Clusters": rpg.generate_non_spherical_clusters(n_clusters=3, n_points_per_cluster=100, elongation=10),
        "Clusters with Outliers": rpg.generate_clusters_with_outliers(n_clusters=3, n_points_per_cluster=100, n_outliers=20),
        "Ring Clusters": rpg.generate_ring_clusters(n_clusters=3, n_points_per_cluster=100, radius_range=(10, 50)),
        "Spiral Clusters": rpg.generate_spiral_clusters(n_clusters=3, n_points_per_cluster=100),
        "Grid Clusters": rpg.generate_grid_clusters(n_clusters_per_side=3, n_points_per_cluster=100, jitter=3),
        "Hierarchical Clusters": rpg.generate_hierarchical_clusters(n_clusters=3, n_points_per_cluster=100, subcluster_ratio=0.2),
        "Cluster Chains": rpg.generate_cluster_chains(n_clusters=3, n_points_per_cluster=100, chain_length=50),
        "Density-Based Clusters": rpg.generate_density_clusters(n_clusters=3, n_points_per_cluster=100, density_factor=5)
    }
    return datasets


def kmeans_example(datasets):
    if datasets is None:
        datasets = generate_datasets()

    kmeans_analysis = KMeansAnalysis(n_clusters=3)


    # Perform clustering on each dataset
    for dataset_name, dataset in datasets.items():
        print(f"\nTesting {dataset_name} with KMeans...")
        data, kmeans = kmeans_analysis.perform_clustering(dataset)
        data, kmeans = kmeans_analysis.perform_clustering(dataset, False)

def dbscan_example(datasets):
    if datasets is None:
        datasets = generate_datasets()

    dbscan_analysis = DBSCANAnalysis(eps=5, min_samples=5)



    for dataset_name, dataset in datasets.items():
        print(f"\nTesting {dataset_name} with DBSCAN...")
        data, dbscan = dbscan_analysis.perform_clustering(dataset)

def divisive_clustering_example(datasets):
    if datasets is None:
        datasets = generate_datasets()

    divisive_analysis = HDClusterAnalysis()

    datasets = generate_datasets()

    for dataset_name, dataset in datasets.items():
        print(f"\nTesting {dataset_name} with Divisive Clustering...")
        divisive_analysis.perform_clustering(dataset)

def main():

    datasets = generate_datasets()

    print("\nRunning KMeans Example:")
    kmeans_example(datasets)

    print("\nRunning DBSCAN Example:")
    dbscan_example(datasets)

    print("\nRunning Divisive Clustering Example:")
    divisive_clustering_example(datasets)

if __name__ == "__main__":
    main()