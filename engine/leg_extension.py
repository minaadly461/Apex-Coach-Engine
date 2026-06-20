# engine/leg_extension.py
import time
from collections import deque
from engine.base import BaseExercise

START_BENT_ANGLE = 95
FULL_EXTENSION_ANGLE = 160
ANGLE_SMOOTHING_WINDOW = 7

class LegExtension(BaseExercise):
    def __init__(self, target_reps=10):
        super().__init__(target_reps)
        self.stage = "DOWN"
        self.angles_buffer = deque(maxlen=ANGLE_SMOOTHING_WINDOW)
        self.rep_start_time = time.time()
        self.last_rep_time = time.time()

    def smooth_angle(self):
        return sum(self.angles_buffer) / len(self.angles_buffer)

    def process_frame(self, frame_bytes):
        frame = self.decode_frame(frame_bytes)
        landmarks = self.extract_landmarks(frame)

        if landmarks is None:
            return None

        # Keypoints
        l_hip  = [landmarks[23].x, landmarks[23].y]
        l_knee = [landmarks[25].x, landmarks[25].y]
        l_ank  = [landmarks[27].x, landmarks[27].y]
        r_hip  = [landmarks[24].x, landmarks[24].y]
        r_knee = [landmarks[26].x, landmarks[26].y]
        r_ank  = [landmarks[28].x, landmarks[28].y]

        # Calculations
        l_angle = self.calculate_angle(l_hip, l_knee, l_ank)
        r_angle = self.calculate_angle(r_hip, r_knee, r_ank)
        avg_angle = (l_angle + r_angle) / 2.0

        self.angles_buffer.append(avg_angle)

        if len(self.angles_buffer) < 2:
            return None

        smoothed_angle = self.smooth_angle()
        self.angles_history.append(smoothed_angle)

        # State Machine
        if smoothed_angle >= FULL_EXTENSION_ANGLE:
            self.stage = "UP"

        elif smoothed_angle <= START_BENT_ANGLE and self.stage == "UP":
            self.stage = "DOWN"

            duration = time.time() - self.rep_start_time
            self.rep_durations.append(duration)

            self.good_reps += 1
            self.rep_start_time = time.time()
            self.last_rep_time = time.time()

            event = self.build_event(
                "rep_completed",
                say=str(self.good_reps)
            )

            if self.is_done():
                return self.build_analytics()

            return event

        return self.build_event("frame_update")