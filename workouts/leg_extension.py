import cv2
import mediapipe as mp
import time
import pyttsx3
import threading
from collections import deque
from leg_utils import calculate_angle, plot_leg_analytics


# ===================== SETTINGS =====================
START_BENT_ANGLE = 95
FULL_EXTENSION_ANGLE = 160
ANGLE_SMOOTHING_WINDOW = 7
VOICE_RATE = 170


# ===================== VOICE ENGINE =====================
engine = pyttsx3.init()
engine.setProperty('rate', VOICE_RATE)

def speak_async(text):
    def run():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run, daemon=True).start()


# ===================== HELPERS =====================
def smooth_angle(history):
    return sum(history) / len(history)


# ===================== MAIN WORKOUT =====================
def start_workout(target_reps=10):

    cap = cv2.VideoCapture(0)

    # ✅ check camera
    if not cap.isOpened():
        print("Camera not working")
        return

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    good_reps = 0
    bad_reps = 0

    stage = "DOWN"

    angles_history = deque(maxlen=ANGLE_SMOOTHING_WINDOW)
    all_angles = []   # ✅ FIX: store all angles

    rep_durations = []
    rep_start_time = None

    last_rep_time = time.time()

    countdown = 3
    is_ready = False
    countdown_start = time.time()

    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:

        while cap.isOpened():

            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # ===================== COUNTDOWN =====================
            if not is_ready:
                elapsed = int(time.time() - countdown_start)
                remaining = countdown - elapsed

                if remaining > 0:
                    cv2.putText(image, f"STARTING IN {remaining}",
                                (120, 250),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.2, (0, 255, 255), 3)
                else:
                    is_ready = True
                    speak_async("Go")
                    rep_start_time = time.time()

                cv2.imshow("Leg Extension Coach", image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # ===================== POSE =====================
            if results.pose_landmarks:

                lm = results.pose_landmarks.landmark

                def knee_angle(a, b, c):
                    return calculate_angle(
                        [a.x, a.y],
                        [b.x, b.y],
                        [c.x, c.y]
                    )

                l_angle = knee_angle(lm[23], lm[25], lm[27])
                r_angle = knee_angle(lm[24], lm[26], lm[28])

                avg_angle = (l_angle + r_angle) / 2
                angles_history.append(avg_angle)

                smoothed_angle = smooth_angle(angles_history)
                all_angles.append(smoothed_angle)   # ✅ FIX

                # ===================== STATE MACHINE (SIMPLIFIED) =====================

                # UP position
                if smoothed_angle >= FULL_EXTENSION_ANGLE:
                    stage = "UP"

                # DOWN → count rep
                elif smoothed_angle <= START_BENT_ANGLE and stage == "UP":
                    stage = "DOWN"

                    duration = time.time() - rep_start_time
                    rep_durations.append(duration)

                    good_reps += 1
                    speak_async(str(good_reps))

                    rep_start_time = time.time()
                    last_rep_time = time.time()

                # ===================== UI =====================
                per = int(
                    max(0, min(100,
                        ((smoothed_angle - START_BENT_ANGLE) /
                         (FULL_EXTENSION_ANGLE - START_BENT_ANGLE)) * 100
                    ))
                )

                cv2.rectangle(image, (20, 100), (40, 350), (50, 50, 50), -1)
                cv2.rectangle(image, (20, 350 - int(2.5 * per)),
                              (40, 350), (0, 255, 204), -1)

                cv2.putText(image, f"REPS: {good_reps}/{target_reps}",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)

                cv2.putText(image, f"STAGE: {stage}",
                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 255), 2)

                cv2.putText(image, f"ANGLE: {int(smoothed_angle)}",
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 255), 2)

                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

            cv2.imshow("Leg Extension Coach", image)

            # ===================== EXIT =====================
            if (good_reps >= target_reps or
                cv2.waitKey(10) & 0xFF == ord('q')):
                break

    cap.release()
    cv2.destroyAllWindows()

    speak_async("Workout Complete")

    # ✅ FIX: send full data
    plot_leg_analytics(all_angles, rep_durations, [])


if __name__ == "__main__":
    start_workout(5)