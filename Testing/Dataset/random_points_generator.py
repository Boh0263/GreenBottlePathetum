# random_points_generator.py
# Questo script in python è stato realizzato per il solo scopo di dimostrare il variare del comportamento degli algoritmi di clustering in base alla distribuzione dei punti.

import numpy as np
import pandas as pd
import requests
import json
import random


def load_api_key(config_file='config.json'):
    """
    Load the API key from the configuration file.
    """
    try:
        with open(config_file, 'r') as file:
            config = json.load(file)
            return config.get("api_key", None)
    except FileNotFoundError:
        print(f"Configuration file {config_file} not found.")
        return None
    except json.JSONDecodeError:
        print("Error decoding JSON from the configuration file.")
        return None


def get_random_seed(config_file='random_config.json'):
    """
    Get a true random integer using Random.org's JSON-RPC API (generateIntegers method).
    If the API call fails, fall back to Python's random module.
    """
    # Load API key from the configuration file
    api_key = load_api_key(config_file)
    if api_key is None:
        print("API key not found. Using fallback method.")
        return random.randint(0, 1000000)  # Fallback using random module

    url = "https://api.random.org/json-rpc/2/invoke"
    headers = {
        "Content-Type": "application/json"
    }

    # Request payload
    payload = {
        "jsonrpc": "2.0",
        "method": "generateIntegers",
        "params": {
            "apiKey": api_key,
            "n": 1,
            "min": 0,
            "max": 1000000,
            "replacement": True  # Allow repeated values (non rilevante)
        },
        "id": 42  # A request ID for tracking (arbitrary number)
    }

    try:
        # Send the request
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response_data = response.json()

        # Extract the random number from the response
        if "result" in response_data:
            return response_data["result"]["random"]["data"][0]
        else:
            print("Error in API response:", response_data)
            return random.randint(0, 1000000)  # Fallback using random module

    except requests.RequestException as e:
        print("HTTP Request failed:", e)
        return random.randint(0, 1000000)  # Fallback using random module
    except json.JSONDecodeError:
        print("Failed to decode JSON response.")
        return random.randint(0, 1000000)  # Fallback using random module




def generate_uniform_data(n_points=100, x_range=(0, 100), y_range=(0, 100)):
    np.random.seed(get_random_seed())
    x = np.random.uniform(x_range[0], x_range[1], n_points)
    y = np.random.uniform(y_range[0], y_range[1], n_points)
    return pd.DataFrame({'x': x, 'y': y})



def generate_overlapping_clusters(n_clusters=3, n_points_per_cluster=100, overlap=10, random_seed=get_random_seed()):
    np.random.seed(random_seed)
    data = []
    for i in range(n_clusters):
        center_x, center_y = np.random.uniform(40, 60), np.random.uniform(40, 60)
        points_x = np.random.normal(center_x, overlap, n_points_per_cluster)
        points_y = np.random.normal(center_y, overlap, n_points_per_cluster)
        data.append(pd.DataFrame({'x': points_x, 'y': points_y}))
    return pd.concat(data, ignore_index=True)



def generate_gaussian_clusters(n_clusters=3, n_points_per_cluster=100, cluster_spread=5, random_seed=get_random_seed()):
    np.random.seed(random_seed)
    data = []
    for i in range(n_clusters):
        center_x, center_y = np.random.uniform(20, 80), np.random.uniform(20, 80)
        points_x = np.random.normal(center_x, cluster_spread, n_points_per_cluster)
        points_y = np.random.normal(center_y, cluster_spread, n_points_per_cluster)
        data.append(pd.DataFrame({'x': points_x, 'y': points_y}))
    return pd.concat(data, ignore_index=True)



def generate_non_spherical_clusters(n_clusters=3, n_points_per_cluster=100, elongation=10, random_seed=get_random_seed()):
    np.random.seed(random_seed)
    data = []
    for i in range(n_clusters):
        center_x, center_y = np.random.uniform(20, 80), np.random.uniform(20, 80)
        points_x = np.random.normal(center_x, elongation, n_points_per_cluster)
        points_y = np.random.normal(center_y, 1, n_points_per_cluster)  # Narrow width
        data.append(pd.DataFrame({'x': points_x, 'y': points_y}))
    return pd.concat(data, ignore_index=True)



def generate_clusters_with_outliers(n_clusters=3, n_points_per_cluster=100, n_outliers=20, random_seed=get_random_seed()):
    np.random.seed(random_seed)
    data = generate_gaussian_clusters(n_clusters=n_clusters, n_points_per_cluster=n_points_per_cluster, cluster_spread=5, random_seed=random_seed)
    outliers_x = np.random.uniform(0, 100, n_outliers)
    outliers_y = np.random.uniform(0, 100, n_outliers)
    outliers = pd.DataFrame({'x': outliers_x, 'y': outliers_y})
    return pd.concat([data, outliers], ignore_index=True)


def generate_density_clusters(n_clusters=3, n_points_per_cluster=100, density_factor=5, random_seed=get_random_seed()):
    np.random.seed(random_seed)
    data = []
    for i in range(n_clusters):
        center_x, center_y = np.random.uniform(20, 80), np.random.uniform(20, 80)
        dense_points_x = np.random.normal(center_x, density_factor, int(n_points_per_cluster * 0.7))
        dense_points_y = np.random.normal(center_y, density_factor, int(n_points_per_cluster * 0.7))
        sparse_points_x = np.random.normal(center_x, density_factor * 3, int(n_points_per_cluster * 0.3))
        sparse_points_y = np.random.normal(center_y, density_factor * 3, int(n_points_per_cluster * 0.3))
        data.append(pd.DataFrame({'x': np.concatenate([dense_points_x, sparse_points_x]),
                                  'y': np.concatenate([dense_points_y, sparse_points_y])}))
    return pd.concat(data, ignore_index=True)


def generate_cluster_chains(n_clusters=3, n_points_per_cluster=100, chain_length=50, random_seed=get_random_seed()):
    np.random.seed(random_seed)
    data = []
    for i in range(n_clusters):
        start_x, start_y = np.random.uniform(20, 80), np.random.uniform(20, 80)
        end_x, end_y = start_x + np.random.uniform(-chain_length, chain_length), start_y + np.random.uniform(-chain_length, chain_length)
        t = np.linspace(0, 1, n_points_per_cluster)
        points_x = start_x * (1 - t) + end_x * t + np.random.normal(0, 1, n_points_per_cluster)
        points_y = start_y * (1 - t) + end_y * t + np.random.normal(0, 1, n_points_per_cluster)
        data.append(pd.DataFrame({'x': points_x, 'y': points_y}))
    return pd.concat(data, ignore_index=True)

def generate_hierarchical_clusters(n_clusters=3, n_points_per_cluster=100, subcluster_ratio=0.2, random_seed=get_random_seed()):
    np.random.seed(random_seed)
    data = []
    for i in range(n_clusters):
        center_x, center_y = np.random.uniform(20, 80), np.random.uniform(20, 80)
        # Main cluster
        main_points_x = np.random.normal(center_x, 5, int(n_points_per_cluster * (1 - subcluster_ratio)))
        main_points_y = np.random.normal(center_y, 5, int(n_points_per_cluster * (1 - subcluster_ratio)))
        data.append(pd.DataFrame({'x': main_points_x, 'y': main_points_y}))
        # Subcluster within the main cluster
        subcluster_x = np.random.normal(center_x + np.random.uniform(-5, 5), 2, int(n_points_per_cluster * subcluster_ratio))
        subcluster_y = np.random.normal(center_y + np.random.uniform(-5, 5), 2, int(n_points_per_cluster * subcluster_ratio))
        data.append(pd.DataFrame({'x': subcluster_x, 'y': subcluster_y}))
    return pd.concat(data, ignore_index=True)


def generate_spiral_clusters(n_clusters=3, n_points_per_cluster=100, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    data = []
    for i in range(n_clusters):
        # Randomize the center position for each spiral
        center_x = np.random.uniform(10, 90)
        center_y = np.random.uniform(10, 90)

        # Generate the spiral
        theta = np.linspace(0, 4 * np.pi, n_points_per_cluster) + (2 * np.pi * i / n_clusters)
        r = theta + np.random.normal(0, 0.5, n_points_per_cluster)
        points_x = r * np.cos(theta) + np.random.normal(0, 0.2, n_points_per_cluster)
        points_y = r * np.sin(theta) + np.random.normal(0, 0.2, n_points_per_cluster)

        # Shift the spiral center to random position
        points_x += center_x
        points_y += center_y

        # Normalize to fit in (0, 0) to (100, 100)
        points_x = (points_x - points_x.min()) / (points_x.max() - points_x.min()) * 100
        points_y = (points_y - points_y.min()) / (points_y.max() - points_y.min()) * 100

        data.append(pd.DataFrame({'x': points_x, 'y': points_y}))

    return pd.concat(data, ignore_index=True)

def generate_grid_clusters(n_clusters_per_side=3, n_points_per_cluster=100, jitter=2, random_seed=get_random_seed()):
    np.random.seed(random_seed)
    data = []
    grid_x, grid_y = np.linspace(20, 80, n_clusters_per_side), np.linspace(20, 80, n_clusters_per_side)
    for cx in grid_x:
        for cy in grid_y:
            points_x = np.random.normal(cx, jitter, n_points_per_cluster)
            points_y = np.random.normal(cy, jitter, n_points_per_cluster)
            data.append(pd.DataFrame({'x': points_x, 'y': points_y}))
    return pd.concat(data, ignore_index=True)


def generate_ring_clusters(n_clusters=3, n_points_per_cluster=100, radius_range=(10, 50), random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    data = []
    for i in range(n_clusters):
        # Randomize the center position for each ring
        center_x = np.random.uniform(10, 90)
        center_y = np.random.uniform(10, 90)

        # Generate the ring
        radius = np.random.uniform(*radius_range)
        angles = np.random.uniform(0, 2 * np.pi, n_points_per_cluster)
        points_x = radius * np.cos(angles) + np.random.normal(0, 1, n_points_per_cluster)
        points_y = radius * np.sin(angles) + np.random.normal(0, 1, n_points_per_cluster)

        # Shift the ring center to random position
        points_x += center_x
        points_y += center_y

        # Normalize to fit in (0, 0) to (100, 100)
        points_x = (points_x - points_x.min()) / (points_x.max() - points_x.min()) * 100
        points_y = (points_y - points_y.min()) / (points_y.max() - points_y.min()) * 100

        data.append(pd.DataFrame({'x': points_x, 'y': points_y}))

    return pd.concat(data, ignore_index=True)

