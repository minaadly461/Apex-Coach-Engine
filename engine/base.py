import numpy as np
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose

class BaseExercise:
    def __init__(self, target_reps=10):
        self.target_reps = target_reps
        self.good_reps = 0
        self.bad_reps = 0
        self.stage = "Waiting..."
        self.current_rep_form = "Good"
        self.angles_history = []
        self.rep_durations = []
        self.bad_reps_list = []
        self.rep_start_time = None
        self.last_rep_time = None
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def calculate_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def get_torso_lean(self, shoulder, hip):
        dx = shoulder[0] - hip[0]
        dy = hip[1] - shoulder[1]
        return np.abs(np.arctan2(dx, dy) * 180.0 / np.pi)

    def decode_frame(self, frame_bytes):
        np_arr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return frame

    def extract_landmarks(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        if results.pose_landmarks:
            return results.pose_landmarks.landmark
        return None

    def build_event(self, event_type, say=None, extra=None):
        event = {
            "event": event_type,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "target_reps": self.target_reps,
        }
        if say:
            event["say"] = say
        if extra:
            event.update(extra)
        return event

    def build_analytics(self):
        return {
            "event": "workout_complete",
            "say": "Workout complete",
            "analytics": {
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "rep_durations": self.rep_durations,
                "angle_history": self.angles_history,
                "bad_reps_list": self.bad_reps_list,
            }
        }

    def process_frame(self, frame_bytes):
        """
        Override this in each exercise.
        Returns a dict event or None.
        """
        raise NotImplementedError

    def is_done(self):
        return self.good_reps >= self.target_reps