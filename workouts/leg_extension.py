import cv2
import mediapipe as mp
import time
from utils import calculate_angle, plot_analytics

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def start_workout(target_reps=10):
    cap = cv2.VideoCapture(0)
    UPPER_THRESH, LOWER_THRESH = 145, 95
    good_reps = 0
    stage = "Waiting..."
    angles_history, rep_durations = [], []
    rep_start_time = time.time()
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.RGB2BGR)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                # Hips, Knees, Ankles
                l_hip, l_kn, l_an = [landmarks[23].x, landmarks[23].y], [landmarks[25].x, landmarks[25].y], [landmarks[27].x, landmarks[27].y]
                r_hip, r_kn, r_an = [landmarks[24].x, landmarks[24].y], [landmarks[26].x, landmarks[26].y], [landmarks[28].x, landmarks[28].y]
                
                avg_angle = (calculate_angle(l_hip, l_kn, l_an) + calculate_angle(r_hip, r_kn, r_an)) / 2.0
                angles_history.append(avg_angle)
                
                if stage == "Waiting...":
                    if avg_angle <= LOWER_THRESH + 10:
                        stage = "extending"
                elif stage == "extending":
                    if avg_angle >= UPPER_THRESH:
                        stage = "returning"
                elif stage == "returning":
                    if avg_angle <= LOWER_THRESH:
                        rep_durations.append(time.time() - rep_start_time)
                        good_reps += 1
                        stage, rep_start_time = "extending", time.time()

                cv2.rectangle(image, (0,0), (320, 160), (0,0,0), -1)
                cv2.putText(image, f"Reps: {good_reps}/{target_reps}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(image, f"Stage: {stage}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
            cv2.imshow('Apex Coach - Leg Extension', image)
            if good_reps >= target_reps or cv2.waitKey(10) & 0xFF == ord('q'): break
                
    cap.release()
    cv2.destroyAllWindows()
    # Note: Leg extension machine forces form, so no bad_reps tracking is used
    plot_analytics(angles_history, rep_durations, 'LEG EXTENSION: TIME PER REP', 'LEG EXTENSION: KNEE ANGLE')