import cv2
import mediapipe as mp
import time
from utils import calculate_angle, get_torso_lean, plot_analytics

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def start_workout(target_reps=10):
    cap = cv2.VideoCapture(0)
    good_reps, bad_reps = 0, 0
    stage, current_rep_form = "Waiting...", "Good"
    angles_history, rep_durations, bad_reps_list = [], [], []
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
                l_sh, l_el, l_wr = [landmarks[11].x, landmarks[11].y], [landmarks[13].x, landmarks[13].y], [landmarks[15].x, landmarks[15].y]
                r_sh, r_el, r_wr = [landmarks[12].x, landmarks[12].y], [landmarks[14].x, landmarks[14].y], [landmarks[16].x, landmarks[16].y]
                l_hip, r_hip = [landmarks[23].x, landmarks[23].y], [landmarks[24].x, landmarks[24].y]
                
                avg_angle = (calculate_angle(l_sh, l_el, l_wr) + calculate_angle(r_sh, r_el, r_wr)) / 2.0
                avg_drift = (calculate_angle(l_hip, l_sh, l_el) + calculate_angle(r_hip, r_sh, r_el)) / 2.0
                avg_sway = (get_torso_lean(l_sh, l_hip) + get_torso_lean(r_sh, r_hip)) / 2.0
                angles_history.append(avg_angle)
                
                warning_text, color = "FORM: GOOD", (0, 255, 255)
                
                if avg_sway > 15 or avg_drift > 30:
                    if current_rep_form == "Good":
                        current_rep_form = "Bad"
                        bad_reps_list.append(f"Rep #{good_reps+bad_reps+1}: Cheating/Swinging detected.")
                    warning_text, color = "WARNING: CHEATING!", (0, 0, 255)
                    
                if avg_angle > 145:
                    if stage == "up":
                        rep_durations.append(time.time() - rep_start_time)
                        if current_rep_form == "Good": good_reps += 1
                        else: bad_reps += 1
                        stage, current_rep_form = "down", "Good"
                        rep_start_time = time.time()
                    elif stage == "Waiting...":
                        stage, current_rep_form = "down", "Good"
                        
                if avg_angle < 55 and stage == "down":
                    stage = "up"
                    
                cv2.rectangle(image, (0,0), (320, 160), (0,0,0), -1)
                cv2.putText(image, f"Good Reps: {good_reps}/{target_reps}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(image, f"Bad Reps: {bad_reps}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(image, f"Stage: {stage}", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(image, warning_text, (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
            cv2.imshow('Apex Coach - Bicep Curl', image)
            if good_reps >= target_reps or cv2.waitKey(10) & 0xFF == ord('q'): break
                
    cap.release()
    cv2.destroyAllWindows()
    plot_analytics(angles_history, rep_durations, 'BICEP: TIME PER REP (SEC)', 'BICEP: ELBOW ANGLE TRACKER', bad_reps_list)