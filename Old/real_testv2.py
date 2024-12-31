from PathetumEnviromentv2 import ClusterEnvironment

def main():
    # Create the environment for Cartesian plotting
    env = ClusterEnvironment()

    # Step 0: Initial Data - Unset cluster labels (label = -1)
    initial_data = [
        {"x": 5.0, "y": 5.0, "label_cluster": -1},  # Cluster 0 region
        {"x": 6.0, "y": 7.0, "label_cluster": -1},
        {"x": 5.5, "y": 6.0, "label_cluster": -1},
        {"x": 15.0, "y": 15.0, "label_cluster": -1},  # Cluster 1 region
        {"x": 16.0, "y": 14.0, "label_cluster": -1},
        {"x": 14.5, "y": 16.5, "label_cluster": -1},
        {"x": 25.0, "y": 5.0, "label_cluster": -1},  # Cluster 2 region
        {"x": 26.0, "y": 6.0, "label_cluster": -1},
        {"x": 25.5, "y": 4.5, "label_cluster": -1},
        {"x": 10.0, "y": 25.0, "label_cluster": -1},  # Cluster 3 region
        {"x": 11.0, "y": 26.0, "label_cluster": -1},
        {"x": 9.5, "y": 24.5, "label_cluster": -1},
    ]
    env.update_environment(initial_data, step_title="Step 0: Raw Data Points")

    # Step 1: Initial Clustering - Assign points to clusters
    step_1_data = [
        {"x": 5.0, "y": 5.0, "label_cluster": 0},  # Cluster 0
        {"x": 6.0, "y": 7.0, "label_cluster": 0},
        {"x": 5.5, "y": 6.0, "label_cluster": 0},
        {"x": 15.0, "y": 15.0, "label_cluster": 1},  # Cluster 1
        {"x": 16.0, "y": 14.0, "label_cluster": 1},
        {"x": 14.5, "y": 16.5, "label_cluster": 1},
        {"x": 25.0, "y": 5.0, "label_cluster": 2},  # Cluster 2
        {"x": 26.0, "y": 6.0, "label_cluster": 2},
        {"x": 25.5, "y": 4.5, "label_cluster": 2},
        {"x": 10.0, "y": 25.0, "label_cluster": 3},  # Cluster 3
        {"x": 11.0, "y": 26.0, "label_cluster": 3},
        {"x": 9.5, "y": 24.5, "label_cluster": 3},
    ]
    env.update_environment(step_1_data, step_title="Step 1: Initial Clustering")

    # Step 2: Updated Clustering - Reassign some points to new clusters
    step_2_data = [
        {"x": 5.0, "y": 5.0, "label_cluster": 0},  # Cluster 0
        {"x": 6.0, "y": 7.0, "label_cluster": 0},
        {"x": 5.5, "y": 6.0, "label_cluster": 0},
        {"x": 15.0, "y": 15.0, "label_cluster": 1},  # Cluster 1
        {"x": 16.0, "y": 14.0, "label_cluster": 1},
        {"x": 14.5, "y": 16.5, "label_cluster": 1},
        {"x": 25.0, "y": 5.0, "label_cluster": 2},  # Cluster 2
        {"x": 26.0, "y": 6.0, "label_cluster": 2},
        {"x": 25.5, "y": 4.5, "label_cluster": 2},
        {"x": 10.0, "y": 25.0, "label_cluster": 3},  # Cluster 3
        {"x": 11.0, "y": 26.0, "label_cluster": 3},
        {"x": 9.5, "y": 24.5, "label_cluster": 3},
        {"x": 18.0, "y": 18.0, "label_cluster": 4},  # New Cluster 4
        {"x": 19.0, "y": 19.0, "label_cluster": 4},
        {"x": 18.5, "y": 17.5, "label_cluster": 4},
    ]
    env.update_environment(step_2_data, step_title="Step 2: Updated Clustering")

if __name__ == "__main__":
    main()
