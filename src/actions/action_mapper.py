class GestureActionMapper:
    GESTURE_ACTIONS = {
        "open": "play",
        "close": "pause",
        "point": "move_mouse",
        "peace": "left_click",
        "thumb": "volume_up",
        "rock": "volume_down",
    }

    CONTINUOUS_ACTIONS = {"move_mouse"}
    REPEAT_ACTIONS = {"volume_up", "volume_down"}

    def __init__(
        self,
        repeat_interval_frames: int = 12,
        release_frames: int = 3,
    ):
        if repeat_interval_frames <= 0:
            raise ValueError("repeat_interval_frames must be greater than 0")

        if release_frames <= 0:
            raise ValueError("release_frames must be greater than 0")

        self.repeat_interval_frames = repeat_interval_frames
        self.release_frames = release_frames

        self.active_gesture = None
        self.frames_held = 0
        self.frames_without_gesture = 0

    def update(self, gesture: str | None) -> str | None:
        if gesture is None:
            self.frames_without_gesture += 1

            if self.frames_without_gesture >= self.release_frames:
                self.reset()

            return None

        gesture = gesture.lower()
        action = self.GESTURE_ACTIONS.get(gesture)

        if action is None:
            self.frames_without_gesture += 1

            if self.frames_without_gesture >= self.release_frames:
                self.reset()

            return None

        self.frames_without_gesture = 0

        if gesture != self.active_gesture:
            self.active_gesture = gesture
            self.frames_held = 1
            return action

        self.frames_held += 1

        if action in self.CONTINUOUS_ACTIONS:
            return action

        if action in self.REPEAT_ACTIONS:
            if self.frames_held % self.repeat_interval_frames == 0:
                return action

        return None

    def reset(self) -> None:
        self.active_gesture = None
        self.frames_held = 0
        self.frames_without_gesture = 0
