
from src.actions.platforms.factory import create_platform
from src.actions.cursor_controller import CursorController


class DesktopController:
    def __init__(self):
        self.platform = create_platform()
        self.cursor = CursorController()

    def move_mouse(self, x: int, y: int) -> None:
        self.platform.move_mouse(x, y)

    def click(self) -> None:
        self.platform.click()

    def volume_up(self) -> None:
        self.platform.volume_up()

    def volume_down(self) -> None:
        self.platform.volume_down()

    def play(self) -> bool:
        return self.platform.play()

    def play_pause(self) -> bool:
        return self.platform.play_pause()

    def pause(self) -> bool:
        return self.platform.pause()

    def execute_action(
        self,
        action: str | None,
        x: float | None = None,
        y: float | None = None,
    ) -> bool:

        if action != "move_mouse":
            self.cursor.release()
            
        if action is None:
            return False

        if action == "move_mouse":
            if x is None or y is None:
                raise ValueError("Normalized mouse coordinates are required")

            screen_x, screen_y = self.cursor.update(x, y)
            self.move_mouse(screen_x, screen_y)
            return True

        if action == "left_click":
            self.click()
            return True

        if action == "volume_up":
            self.volume_up()
            return True

        if action == "volume_down":
            self.volume_down()
            return True

        if action == "play":
            return self.play()

        if action == "pause":
            return self.pause()

        raise ValueError(f"Unknown desktop action: {action}")
