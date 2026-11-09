import platform
import subprocess


class VolumeController:
    VOLUME_STEP = 5

    def __init__(self):
        self.os_name = platform.system()

        if self.os_name not in {"Linux", "Windows", "Darwin"}:
            raise RuntimeError(
                f"Unsupported operating system: {self.os_name}"
            )

    def increase(self):
        self._adjust_volume(self.VOLUME_STEP)

    def decrease(self):
        self._adjust_volume(-self.VOLUME_STEP)

    def _adjust_volume(self, change):
        if self.os_name == "Linux":
            self._linux_volume(change)

        elif self.os_name == "Windows":
            self._windows_volume(change)

        elif self.os_name == "Darwin":
            self._macos_volume(change)

    @staticmethod
    def _linux_volume(change):
        amount = f"{'+' if change > 0 else ''}{change}%"

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

    @staticmethod
    def _windows_volume(change):
        from pycaw.pycaw import AudioUtilities

        device = AudioUtilities.GetSpeakers()
        volume = device.EndpointVolume

        current = volume.GetMasterVolumeLevelScalar()
        updated = max(0.0, min(1.0, current + change / 100))

        volume.SetMasterVolumeLevelScalar(updated, None)

    @staticmethod
    def _macos_volume(change):
        result = subprocess.run(
            [
                "osascript",
                "-e",
                "output volume of (get volume settings)",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        current = int(result.stdout.strip())
        updated = max(0, min(100, current + change))

        subprocess.run(
            [
                "osascript",
                "-e",
                f"set volume output volume {updated}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )