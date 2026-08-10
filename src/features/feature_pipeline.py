import numpy as np
import pandas as pd

from src.features.angle_features import calculate_angles
from src.features.geometric_features import calculate_distances

from src.features.landmark_features import (
    extract_landmarks,
    make_wrist_relative,
    normalize_scale,
)


def build_features(df: pd.DataFrame) -> np.ndarray:
    
    raw_features = extract_landmarks(df)

    relative_features = make_wrist_relative(raw_features)

    normalized_features = normalize_scale(relative_features)

    distances = calculate_distances(normalized_features)

    angles = calculate_angles(normalized_features)

    return np.hstack([normalized_features, distances, angles,])