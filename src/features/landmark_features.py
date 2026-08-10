import numpy as np
import pandas as pd


LANDMARK_COLUMNS = [
    f"landmark_{i}_{axis}"
    for i in range(21)
    for axis in ("x", "y", "z")
]


def extract_landmarks(df: pd.DataFrame) -> np.ndarray:
    missing_columns = [
        column
        for column in LANDMARK_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing landmark columns: {missing_columns}"
        )

    return df[LANDMARK_COLUMNS].to_numpy(dtype=np.float64)


def make_wrist_relative(features: np.ndarray) -> np.ndarray:
    if features.ndim != 2 or features.shape[1] != 63:
        raise ValueError(
            "Expected features with shape (n_samples, 63)"
        )

    landmarks = features.reshape(-1, 21, 3)

    wrist = landmarks[:, 0:1, :]
    relative_landmarks = landmarks - wrist

    return relative_landmarks.reshape(-1, 63)


def normalize_scale(features: np.ndarray, return_scales: bool = False,) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    if features.ndim != 2 or features.shape[1] != 63:
        raise ValueError(
            "Expected features with shape (n_samples, 63)"
        )

    landmarks = features.reshape(-1, 21, 3)

    wrist = landmarks[:, 0, :]
    middle_mcp = landmarks[:, 9, :]

    scales = np.linalg.norm(
        middle_mcp - wrist,
        axis=1,
    )

    if np.any(scales <= 1e-8):
        raise ValueError("Cannot normalize samples with zero hand scale")

    normalized = landmarks / scales[:, np.newaxis, np.newaxis]
    normalized = normalized.reshape(-1, 63)

    if return_scales:
        return normalized, scales

    return normalized