import math

import pyautogui

from src.actions.cursor_config import CursorConfig


class CursorController:
    def __init__(self, config: CursorConfig | None = None):
        self.config = config or CursorConfig()

        self.screen_width, self.screen_height = pyautogui.size()

        self.tracking = False

        self.anchor_input_x = None
        self.anchor_input_y = None

        self.anchor_cursor_x = None
        self.anchor_cursor_y = None

        self.previous_x = None
        self.previous_y = None

    def update(self, x: float, y: float) -> tuple[int, int]:
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("Cursor coordinates must be finite")

        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))

        if self.config.mirror_x:
            x = 1.0 - x

        if not self.tracking:
            cursor_x, cursor_y = pyautogui.position()

            self.anchor_input_x = x
            self.anchor_input_y = y

            self.anchor_cursor_x = cursor_x
            self.anchor_cursor_y = cursor_y

            self.previous_x = float(cursor_x)
            self.previous_y = float(cursor_y)

            self.tracking = True

            return cursor_x, cursor_y

        delta_x = x - self.anchor_input_x
        delta_y = y - self.anchor_input_y

        target_x = (
            self.anchor_cursor_x
            + delta_x * self.screen_width * self.config.sensitivity
        )

        target_y = (
            self.anchor_cursor_y
            + delta_y * self.screen_height * self.config.sensitivity
        )

        alpha = self.config.smoothing_factor

        self.previous_x += alpha * (target_x - self.previous_x)
        self.previous_y += alpha * (target_y - self.previous_y)

        cursor_x = max(
            0,
            min(self.screen_width - 1, round(self.previous_x)),
        )

        cursor_y = max(
            0,
            min(self.screen_height - 1, round(self.previous_y)),
        )

        return cursor_x, cursor_y

    def release(self):
        """Pause tracking while preserving the desktop cursor position."""
        self.tracking = False

        self.anchor_input_x = None
        self.anchor_input_y = None

        self.anchor_cursor_x = None
        self.anchor_cursor_y = None

        self.previous_x = None
        self.previous_y = None

    def reset(self):
        self.release()

    def recalibrate(self):
        self.release()