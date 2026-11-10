import platform

from src.actions.platforms.base import DesktopPlatform


def create_platform() -> DesktopPlatform:
    os_name = platform.system()

    if os_name == "Linux":
        from src.actions.platforms.linux import LinuxDesktop

        return LinuxDesktop()

    if os_name == "Windows":
        from src.actions.platforms.windows import WindowsDesktop

        return WindowsDesktop()

    if os_name == "Darwin":
        from src.actions.platforms.macos import MacOSDesktop

        return MacOSDesktop()

    raise RuntimeError(f"Unsupported operating system: {os_name}")
