import numpy as np


def calculate_angle(first: np.ndarray, middle: np.ndarray, last: np.ndarray,) -> np.ndarray:
    vector_a = first - middle
    vector_b = last - middle

    denominator = (
        np.linalg.norm(vector_a, axis=1)
        * np.linalg.norm(vector_b, axis=1)
    )

    denominator = np.maximum(denominator, 1e-8)

    cosine = np.sum(
        vector_a * vector_b,
        axis=1,
    ) / denominator

    cosine = np.clip(cosine, -1.0, 1.0)

    return np.arccos(cosine)


def calculate_angles(features: np.ndarray) -> np.ndarray:
    landmarks = features.reshape(-1, 21, 3)

    angle_triplets = [
        (1, 2, 3),
        (2, 3, 4),
        (5, 6, 7),
        (6, 7, 8),
        (9, 10, 11),
        (10, 11, 12),
        (13, 14, 15),
        (14, 15, 16),
        (17, 18, 19),
        (18, 19, 20),
    ]

    angles = []

    for first, middle, last in angle_triplets:
        angle = calculate_angle(
            landmarks[:, first, :],
            landmarks[:, middle, :],
            landmarks[:, last, :],
        )

        angles.append(angle)

    return np.column_stack(angles)