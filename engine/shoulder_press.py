# engine/shoulder_press.py
import time
from engine.base import BaseExercise

UPPER_THRESH = 150
LOWER_THRESH = 45
REVERSAL_TOLERANCE = 20

class ShoulderPress(BaseExercise):
    def __init__(self, target_reps=10):
        super().__init__(target_reps)
        self.stage = "Waiting..."
        self.current_rep_form = "Good"
        self.extreme_angle = 0
        self.rep_start_time = time.time()
        self.last_rep_time = time.time()

    def process_frame(self, frame_bytes):
        frame = self.decode_frame(frame_bytes)
        landmarks = self.extract_landmarks(frame)

        if landmarks is None:
            return None

        # Keypoints
        l_sh  = [landmarks[11].x, landmarks[11].y]
        l_el  = [landmarks[13].x, landmarks[13].y]
        l_wr  = [landmarks[15].x, landmarks[15].y]
        r_sh  = [landmarks[12].x, landmarks[12].y]
        r_el  = [landmarks[14].x, landmarks[14].y]
        r_wr  = [landmarks[16].x, landmarks[16].y]
        l_hip = [landmarks[23].x, landmarks[23].y]
        r_hip = [landmarks[24].x, landmarks[24].y]

        # Calculations
        avg_angle = (
            self.calculate_angle(l_sh, l_el, l_wr) +
            self.calculate_angle(r_sh, r_el, r_wr)
        ) / 2.0

        avg_sway = (
            self.get_torso_lean(l_sh, l_hip) +
            self.get_torso_lean(r_sh, r_hip)
        ) / 2.0

        self.angles_history.append(avg_angle)

        # Form Check
        if avg_sway > 20:
            if self.current_rep_form == "Good":
                self.current_rep_form = "Bad"
                self.bad_reps += 1
                self.bad_reps_list.append(
                    f"Rep #{self.good_reps + self.bad_reps}: Back arched"
                )
                return self.build_event(
                    "form_error",
                    say="Don't arch your back"
                )

        # State Machine
        if self.stage == "Waiting...":
            if avg_angle <= LOWER_THRESH + 15:
                self.stage = "pushing_up"
                self.extreme_angle = avg_angle
                self.current_rep_form = "Good"

        elif self.stage == "pushing_up":
            self.extreme_angle = max(self.extreme_angle, avg_angle)

            if avg_angle >= UPPER_THRESH:
                self.stage = "pulling_down"
                self.extreme_angle = avg_angle

            elif (self.extreme_angle > LOWER_THRESH + 20 and
                  avg_angle < self.extreme_angle - REVERSAL_TOLERANCE):
                if self.current_rep_form == "Good":
                    self.current_rep_form = "Bad"
                    self.bad_reps += 1
                    self.bad_reps_list.append(
                        f"Rep #{self.good_reps + self.bad_reps}: Half rep (not high enough)"
                    )
                    self.stage = "pulling_down"
                    self.extreme_angle = avg_angle
                    return self.build_event(
                        "form_error",
                        say="Wrong, push all the way up"
                    )

        elif self.stage == "pulling_down":
            self.extreme_angle = min(self.extreme_angle, avg_angle)

            if avg_angle <= LOWER_THRESH:
                duration = time.time() - self.rep_start_time
                self.rep_durations.append(duration)

                if self.current_rep_form == "Good":
                    self.good_reps += 1
                    event = self.build_event(
                        "rep_completed",
                        say=str(self.good_reps)
                    )
                else:
                    event = self.build_event("bad_rep", say="Wrong")

                self.stage = "pushing_up"
                self.current_rep_form = "Good"
                self.extreme_angle = avg_angle
                self.rep_start_time = time.time()
                self.last_rep_time = time.time()

                if self.is_done():
                    return self.build_analytics()

                return event

            elif (self.extreme_angle < UPPER_THRESH - 20 and
                  avg_angle > self.extreme_angle + REVERSAL_TOLERANCE):
                if self.current_rep_form == "Good":
                    self.current_rep_form = "Bad"
                    self.bad_reps += 1
                    self.bad_reps_list.append(
                        f"Rep #{self.good_reps + self.bad_reps}: Half rep (not low enough)"
                    )
                    self.stage = "pushing_up"
                    self.extreme_angle = avg_angle
                    return self.build_event(
                        "form_error",
                        say="Wrong, lower all the way down"
                    )

        return self.build_event("frame_update")