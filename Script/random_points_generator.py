# random_points_generator.py

import numpy as np


def generate_random_points(n, x_range=(0, 100), y_range=(0, 100)):
    """
    Genera n punti casuali nelle coordinate specificate.

    Parameters:
    - n: numero di punti da generare
    - x_range: intervallo per la coordinata x
    - y_range: intervallo per la coordinata y

    Returns:
    - points: array di punti casuali (n, 2)
    """
    # Generiamo n valori casuali per le coordinate x e y
    x_points = np.random.uniform(x_range[0], x_range[1], n)
    y_points = np.random.uniform(y_range[0], y_range[1], n)

    # Combiniamo x e y in un array di coordinate
    points = np.vstack((x_points, y_points)).T
    return points