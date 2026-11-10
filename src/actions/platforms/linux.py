import subprocess

import pyautogui

from src.actions.platforms.base import DesktopPlatform


class LinuxDesktop(DesktopPlatform):
    def __init__(self):
        pyautogui.FAILSAFE = True

    def move_mouse(self, x: int, y: int) -> None:
        pyautogui.moveTo(x, y)

    def click(self) -> None:
        self._run_ydotool("click", "0xC0")

    def volume_up(self) -> None:
        self._adjust_volume("+5%")

    def volume_down(self) -> None:
        self._adjust_volume("-5%")

    def play(self) -> bool:
        return self._media_command("play")

    def play_pause(self) -> bool:
        return self._media_command("play-pause")

    def pause(self) -> bool:
        return self._media_command("pause")

    @staticmethod
    def _run_ydotool(*args: str) -> None:
        try:
            result = subprocess.run(
                ["ydotool", *args],
                capture_output=True,
                text=True,
                check=False,
            )

        except FileNotFoundError as exc:
            raise RuntimeError(
                "ydotool is not installed. "
                "Install it using: sudo dnf install ydotool"
            ) from exc

        if result.returncode != 0:
            error = result.stderr.strip()

            raise RuntimeError(
                f"ydotool command failed: {error}"
            )

    @staticmethod
    def _media_command(action: str) -> bool:
        try:
            result = subprocess.run(
                ["playerctl", action],
                capture_output=True,
                text=True,
                check=False,
            )

        except FileNotFoundError as exc:
            raise RuntimeError(
                "playerctl is not installed. "
                "Install it using: sudo dnf install playerctl"
            ) from exc

        if result.returncode == 0:
            return True

        error = result.stderr.strip()

        unavailable_messages = (
            "No players found",
            "No player could handle this command",
        )

        if any(message in error for message in unavailable_messages):
            return False

        raise RuntimeError(
            f"Media command '{action}' failed: {error}"
        )

    @staticmethod
    def _adjust_volume(amount: str) -> None:
        try:
            subprocess.run(
                [
                    "pactl",
                    "set-sink-volume",
                    "@DEFAULT_SINK@",
                    amount,
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        except FileNotFoundError as exc:
            raise RuntimeError(
                "pactl is not installed or unavailable."
            ) from exc

        except subprocess.CalledProcessError as exc:
            error = exc.stderr.strip() if exc.stderr else str(exc)

            raise RuntimeError(
                f"Volume adjustment failed: {error}"
            ) from exc
