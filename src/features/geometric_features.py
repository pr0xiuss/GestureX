import numpy as np


def calculate_distances(features: np.ndarray,) -> np.ndarray:
    landmarks = features.reshape(-1, 21, 3)

    pairs = [
        (0, 4),
        (0, 8),
        (0, 12),
        (0, 16),
        (0, 20),
        (4, 8),
        (8, 12),
        (12, 16),
        (16, 20),
    ]

    distances = []

    for first, second in pairs:
        distance = np.linalg.norm(
            landmarks[:, first, :] - landmarks[:, second, :],
            axis=1,
        )

        distances.append(distance)

    return np.column_stack(distances)