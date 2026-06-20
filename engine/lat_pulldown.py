# engine/lat_pulldown.py

import time
from collections import deque
from engine.base import BaseExercise
import mediapipe as mp

mp_pose = mp.solutions.pose

LEFT_SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER.value
LEFT_ELBOW = mp_pose.PoseLandmark.LEFT_ELBOW.value
LEFT_WRIST = mp_pose.PoseLandmark.LEFT_WRIST.value
LEFT_HIP = mp_pose.PoseLandmark.LEFT_HIP.value

RIGHT_SHOULDER = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
RIGHT_ELBOW = mp_pose.PoseLandmark.RIGHT_ELBOW.value
RIGHT_WRIST = mp_pose.PoseLandmark.RIGHT_WRIST.value
RIGHT_HIP = mp_pose.PoseLandmark.RIGHT_HIP.value


UPPER_THRESH = 150
LOWER_THRESH = 50

REVERSAL_TOLERANCE = 15
TORSO_SWAY_LIMIT = 25
ANGLE_WINDOW = 5
IDLE_TIMEOUT = 5


class LatPulldown(BaseExercise):

    def __init__(self, target_reps=10):

        super().__init__(target_reps)

        self.stage = "Waiting"
        self.current_rep_form = "Good"

        self.extreme_angle = None

        self.rep_start_time = None
        self.last_rep_time = time.time()

        self.angle_buffer = deque(
            maxlen=ANGLE_WINDOW
        )

        self.pending_voice = None

    ################################################

    def smooth_angle(self, angle):

        self.angle_buffer.append(
            angle
        )

        return sum(
            self.angle_buffer
        ) / len(self.angle_buffer)

    ################################################

    def point(self, landmarks, idx):

        return [
            landmarks[idx].x,
            landmarks[idx].y
        ]

    ################################################

    def reset_rep(self):

        self.stage = "Waiting"

        self.current_rep_form = "Good"

        self.extreme_angle = None

        self.rep_start_time = None

    ################################################

    def process_frame(self, frame_bytes):

        frame = self.decode_frame(
            frame_bytes
        )

        landmarks = self.extract_landmarks(
            frame
        )

        if landmarks is None:
            return None

        ################################################
        # joints
        ################################################

        l_sh = self.point(
            landmarks,
            LEFT_SHOULDER
        )

        l_el = self.point(
            landmarks,
            LEFT_ELBOW
        )

        l_wr = self.point(
            landmarks,
            LEFT_WRIST
        )

        l_hip = self.point(
            landmarks,
            LEFT_HIP
        )

        r_sh = self.point(
            landmarks,
            RIGHT_SHOULDER
        )

        r_el = self.point(
            landmarks,
            RIGHT_ELBOW
        )

        r_wr = self.point(
            landmarks,
            RIGHT_WRIST
        )

        r_hip = self.point(
            landmarks,
            RIGHT_HIP
        )

        ################################################
        # elbow angle
        ################################################

        left_angle = self.calculate_angle(
            l_sh,
            l_el,
            l_wr
        )

        right_angle = self.calculate_angle(
            r_sh,
            r_el,
            r_wr
        )

        avg_angle = (
            left_angle +
            right_angle
        ) / 2

        avg_angle = self.smooth_angle(
            avg_angle
        )

        ################################################
        # torso lean
        ################################################

        left_sway = self.get_torso_lean(
            l_sh,
            l_hip
        )

        right_sway = self.get_torso_lean(
            r_sh,
            r_hip
        )

        avg_sway = (
            left_sway +
            right_sway
        ) / 2

        ################################################
        # timeout reset
        ################################################

        if (

            self.stage != "Waiting"

            and

            time.time()
            -
            self.last_rep_time

            >

            IDLE_TIMEOUT

        ):

            self.reset_rep()

        ################################################
        # form check
        ################################################

        if (

            avg_sway >
            TORSO_SWAY_LIMIT

            and

            self.current_rep_form
            ==
            "Good"

        ):

            self.current_rep_form = "Bad"

            self.bad_reps_list.append(

                f"Rep #{self.good_reps+self.bad_reps+1}"
                f": Back swing detected"

            )

            self.pending_voice = (
                "Don't lean back"
            )

        ################################################
        # state machine
        ################################################

        if self.stage == "Waiting":

            if avg_angle >= UPPER_THRESH:

                self.stage = "Pulling"

                self.extreme_angle = avg_angle

                self.rep_start_time = (
                    time.time()
                )

        ################################################

        elif self.stage == "Pulling":

            self.extreme_angle = min(

                self.extreme_angle,
                avg_angle

            )

            if avg_angle <= LOWER_THRESH:

                self.stage = "Returning"

                self.extreme_angle = (
                    avg_angle
                )

            elif (

                avg_angle >

                self.extreme_angle
                +
                REVERSAL_TOLERANCE

            ):

                self.current_rep_form = "Bad"

                self.pending_voice = (
                    "Pull lower"
                )

                self.stage = (
                    "Returning"
                )

                self.extreme_angle = (
                    avg_angle
                )

        ################################################

        elif self.stage == "Returning":

            self.extreme_angle = max(

                self.extreme_angle,
                avg_angle

            )

            if avg_angle >= UPPER_THRESH:

                duration = (

                    time.time()
                    -
                    self.rep_start_time

                )

                self.rep_durations.append(
                    duration
                )

                ################################

                if (

                    self.current_rep_form
                    ==
                    "Good"

                ):

                    self.good_reps += 1

                    event = (

                        self.build_event(

                            "rep_completed",

                            say=str(
                                self.good_reps
                            )

                        )
                    )

                else:

                    self.bad_reps += 1

                    event = (

                        self.build_event(

                            "bad_rep",

                            say="Wrong"

                        )
                    )

                ################################

                self.last_rep_time = (
                    time.time()
                )

                self.reset_rep()

                if self.is_done():

                    return (
                        self.build_analytics()
                    )

                return event

        ################################################
        # delayed voice
        ################################################

        if self.pending_voice:

            msg = self.pending_voice

            self.pending_voice = None

            return self.build_event(
                "form_error",
                say=msg
            )

        return self.build_event(
            "frame_update"
        )