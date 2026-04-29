# bicep.py
import cv2
import mediapipe as mp
import time
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from utils import calculate_angle, get_torso_lean

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def start_workout(target_reps=10):
    cap = cv2.VideoCapture(0)
    
    good_reps, bad_reps = 0, 0
    stage = "Waiting..."
    current_rep_form = "Good"
    
    angles_history, rep_durations = [], []
    rep_start_time = time.time()
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Process Frame
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.RGB2BGR)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Joint Coordinates
                l_sh = [landmarks[11].x, landmarks[11].y]
                l_el = [landmarks[13].x, landmarks[13].y]
                l_wr = [landmarks[15].x, landmarks[15].y]
                l_hip = [landmarks[23].x, landmarks[23].y]
                
                r_sh = [landmarks[12].x, landmarks[12].y]
                r_el = [landmarks[14].x, landmarks[14].y]
                r_wr = [landmarks[16].x, landmarks[16].y]
                r_hip = [landmarks[24].x, landmarks[24].y]
                
                # Kinematics
                avg_angle = (calculate_angle(l_sh, l_el, l_wr) + calculate_angle(r_sh, r_el, r_wr)) / 2.0
                angles_history.append(avg_angle)
                
                avg_drift = (calculate_angle(l_hip, l_sh, l_el) + calculate_angle(r_hip, r_sh, r_el)) / 2.0
                avg_sway = (get_torso_lean(l_sh, l_hip) + get_torso_lean(r_sh, r_hip)) / 2.0
                
                warning_text, color = "FORM: GOOD", (0, 255, 255) # Yellowish green
                
                # Form Check
                if avg_sway > 15 or avg_drift > 30:
                    current_rep_form = "Bad"
                    warning_text, color = "WARNING: CHEATING!", (0, 0, 255) # Red
                    
                # State Machine
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
                    
                # UI Overlay
                cv2.rectangle(image, (0,0), (320, 160), (0,0,0), -1)
                cv2.putText(image, f"Good Reps: {good_reps}/{target_reps}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(image, f"Bad Reps: {bad_reps}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(image, f"Stage: {stage}", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(image, warning_text, (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
            cv2.imshow('Apex Coach - Bicep Curl', image)
            
            # Exit condition
            if good_reps >= target_reps or cv2.waitKey(10) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Workout Complete! Good: {good_reps} | Bad: {bad_reps}")
    plot_charts(angles_history, rep_durations)

def plot_charts(angles, durations):
    if not angles: return
    smoothed = gaussian_filter1d(angles, sigma=3)
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.canvas.manager.set_window_title('Workout Analytics')
    
    if durations:
        ax1.bar(range(1, len(durations)+1), durations, color='#00ffcc')
        ax1.set_title('TIME PER REP (SEC)')
        ax1.set_xlabel('Rep #')
        
    ax2.plot(smoothed, color='#00ffcc', linewidth=2)
    ax2.fill_between(range(len(smoothed)), smoothed, color='#00ffcc', alpha=0.2)
    ax2.set_title('ELBOW ANGLE TRACKER')
    ax2.set_xlabel('Frame Index')
    ax2.set_ylabel('Degrees')
    
    plt.tight_layout()
    plt.show()