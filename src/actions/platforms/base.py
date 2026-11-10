from abc import ABC, abstractmethod


class DesktopPlatform(ABC):

    @abstractmethod
    def move_mouse(self, x: int, y: int) -> None:
        pass

    @abstractmethod
    def click(self) -> None:
        pass

    @abstractmethod
    def volume_up(self) -> None:
        pass

    @abstractmethod
    def volume_down(self) -> None:
        pass

    @abstractmethod
    def play_pause(self) -> None:
        pass

    @abstractmethod
    def pause(self) -> None:
        pass
    
    def play(self) -> None:
        raise NotImplementedError(
            "Explicit play is not implemented for this platform."
        )