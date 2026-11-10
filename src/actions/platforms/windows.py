import asyncio

import pyautogui
from pycaw.pycaw import AudioUtilities

from winrt.windows.media.control import (
    GlobalSystemMediaTransportControlsSessionManager,
    GlobalSystemMediaTransportControlsSessionPlaybackStatus,
)

from src.actions.platforms.base import DesktopPlatform


class WindowsDesktop(DesktopPlatform):
    def __init__(self):
        pyautogui.FAILSAFE = True
        self.volume = AudioUtilities.GetSpeakers().EndpointVolume

    def move_mouse(self, x: int, y: int) -> None:
        pyautogui.moveTo(x, y)

    def click(self) -> None:
        pyautogui.click()

    def volume_up(self) -> None:
        self._adjust_volume(0.05)

    def volume_down(self) -> None:
        self._adjust_volume(-0.05)

    def play(self) -> bool:
        return self._run_media_command("play")

    def pause(self) -> bool:
        return self._run_media_command("pause")

    def play_pause(self) -> bool:
        return self._run_media_command("play_pause")

    def _adjust_volume(self, change: float) -> None:
        current = self.volume.GetMasterVolumeLevelScalar()

        updated = max(
            0.0,
            min(1.0, current + change),
        )

        self.volume.SetMasterVolumeLevelScalar(updated, None)

    @staticmethod
    def _run_media_command(action: str) -> bool:
        return asyncio.run(
            WindowsDesktop._media_command(action)
        )

    @staticmethod
    async def _media_command(action: str) -> bool:
        manager = (
            await GlobalSystemMediaTransportControlsSessionManager
            .request_async()
        )

        session = manager.get_current_session()

        if session is None:
            return False

        playback_info = session.get_playback_info()
        status = playback_info.playback_status

        playing = (
            status
            == GlobalSystemMediaTransportControlsSessionPlaybackStatus.PLAYING
        )

        if action == "play":
            if playing:
                return True

            return await session.try_play_async()

        if action == "pause":
            if not playing:
                return True

            return await session.try_pause_async()

        if action == "play_pause":
            if playing:
                return await session.try_pause_async()

            return await session.try_play_async()

        raise ValueError(
            f"Unsupported media action: {action}"
        )
