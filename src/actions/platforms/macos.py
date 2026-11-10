import subprocess

import pyautogui

from src.actions.platforms.base import DesktopPlatform


class MacOSDesktop(DesktopPlatform):
    MEDIA_APPS = {
        "Music": "Music",
        "Spotify": "Spotify",
        "VLC": "VLC",
    }

    def __init__(self):
        pyautogui.FAILSAFE = True

    def move_mouse(self, x: int, y: int) -> None:
        pyautogui.moveTo(x, y)

    def click(self) -> None:
        pyautogui.click()

    def volume_up(self) -> None:
        self._adjust_volume(5)

    def volume_down(self) -> None:
        self._adjust_volume(-5)

    def play(self) -> bool:
        return self._media_command("play")

    def play_pause(self) -> bool:
        return self._media_command("playpause")

    def pause(self) -> bool:
        return self._media_command("pause")

    def _media_command(self, action: str) -> bool:
        app = self._find_running_media_app()

        if app is None:
            return False

        result = subprocess.run(
            [
                "osascript",
                "-e",
                f'tell application "{app}" to {action}',
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Media command failed for {app}: "
                f"{result.stderr.strip()}"
            )

        return True

    @classmethod
    def _find_running_media_app(cls) -> str | None:
        result = subprocess.run(
            [
                "osascript",
                "-e",
                "tell application \"System Events\" "
                "to get name of every process",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        running_apps = result.stdout

        for app in cls.MEDIA_APPS:
            if app in running_apps:
                return cls.MEDIA_APPS[app]

        return None

    @staticmethod
    def _adjust_volume(change: int) -> None:
        script = (
            "set currentVolume to output volume of "
            "(get volume settings)\n"
            f"set newVolume to currentVolume + {change}\n"
            "if newVolume > 100 then set newVolume to 100\n"
            "if newVolume < 0 then set newVolume to 0\n"
            "set volume output volume newVolume"
        )

        subprocess.run(
            ["osascript", "-e", script],
            check=True,
            capture_output=True,
            text=True,
        )
