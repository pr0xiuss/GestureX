from dataclasses import dataclass


@dataclass(frozen=True)
class CursorConfig:
    sensitivity: float = 1.35
    smoothing_factor: float = 0.35
    mirror_x: bool = True

    def __post_init__(self):
        if self.sensitivity <= 0:
            raise ValueError("Sensitivity must be greater than 0")

        if not 0 < self.smoothing_factor <= 1:
            raise ValueError(
                "Smoothing factor must be between 0 and 1"
            )
