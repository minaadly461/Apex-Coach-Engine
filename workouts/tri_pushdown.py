import cv2
import mediapipe as mp
import time
import pyttsx3
import threading
from utils import calculate_angle, get_torso_lean, plot_analytics

# --- Async Voice Engine ---
def speak_async(text):
    """Runs the voice engine in a separate thread to prevent camera lag."""
    def run_speech():
        engine = pyttsx3.init()
        engine.setProperty('rate', 170)
        engine.say(text)
        engine.runAndWait()
    
    threading.Thread(target=run_speech, daemon=True).start()

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def start_workout(target_reps=10):
    cap = cv2.VideoCapture(0)
    good_reps, bad_reps = 0, 0
    stage, current_rep_form = "Waiting...", "Good"
    bad_rep_reason = ""
    
    # Analytics Storage
    angles_history, rep_durations, bad_reps_list = [], [], []
    
    # State & Timers
    is_ready = False
    countdown = 3
    last_rep_time = time.time()
    rep_start_time = time.time()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Image Processing
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # 1. Countdown Logic
            if not is_ready:
                cv2.putText(image, f"TRICEPS READY? {countdown}", (80, 240), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4)
                cv2.imshow('Apex Coach - Tricep Pushdown', image)
                if cv2.waitKey(1000) & 0xFF == ord('q'): break
                countdown -= 1
                if countdown == 0:
                    is_ready = True
                    speak_async("Go!")
                    rep_start_time = time.time()
                    last_rep_time = time.time()
                continue

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Point Coordinates
                l_sh, l_el, l_wr = [landmarks[11].x, landmarks[11].y], [landmarks[13].x, landmarks[13].y], [landmarks[15].x, landmarks[15].y]
                r_sh, r_el, r_wr = [landmarks[12].x, landmarks[12].y], [landmarks[14].x, landmarks[14].y], [landmarks[16].x, landmarks[16].y]
                l_hip, r_hip = [landmarks[23].x, landmarks[23].y], [landmarks[24].x, landmarks[24].y]

                # Core Calculations
                avg_angle = (calculate_angle(l_sh, l_el, l_wr) + calculate_angle(r_sh, r_el, r_wr)) / 2.0
                avg_drift = (calculate_angle(l_hip, l_sh, l_el) + calculate_angle(r_hip, r_sh, r_el)) / 2.0
                avg_sway = (get_torso_lean(l_sh, l_hip) + get_torso_lean(r_sh, r_hip)) / 2.0
                angles_history.append(avg_angle)

                # --- 2. Form Checking (Instant Capture) ---
                warning_text, color = "FORM: GOOD", (204, 255, 0)
                if avg_sway > 15:
                    warning_text, color = "WARNING: DON'T SWING!", (60, 0, 255)
                    if current_rep_form == "Good":
                        current_rep_form = "Bad"
                        bad_rep_reason = "Excessive body sway"
                elif avg_drift > 15:
                    warning_text, color = "WARNING: TUCK ELBOWS!", (60, 0, 255)
                    if current_rep_form == "Good":
                        current_rep_form = "Bad"
                        bad_rep_reason = "Elbows drifted"

                # --- 3. Rep Counting Logic (Top -> Down -> Top) ---
                
                # الحالة 1: اليد في الأعلى (Position: TOP / Angle < 65)
                if avg_angle < 65:
                    if stage == "up": # عودة من وضعية الفرد التام
                        duration = (time.time() - rep_start_time)
                        rep_durations.append(duration)
                        
                        if current_rep_form == "Good":
                            good_reps += 1
                            speak_async(str(good_reps))
                        else:
                            bad_reps += 1
                            speak_async("Wrong")
                            bad_reps_list.append(f"Rep #{good_reps+bad_reps}: {bad_rep_reason}")
                        
                        # Reset for next rep
                        stage = "down" 
                        current_rep_form = "Good"
                        bad_rep_reason = ""
                        rep_start_time = time.time()
                        last_rep_time = time.time()
                        
                    elif stage == "Waiting...":
                        stage = "down" # البداية من فوق، الجهاز ينتظر النزول

                # الحالة 2: اليد في الأسفل (Position: FULL EXTENSION / Angle > 150)
                if avg_angle > 150 and stage == "down":
                    stage = "up" # وصل لأقصى نقطة نزول، الآن ينتظر العودة للأعلى

                # 4. Progress Bar (Visualizing the 65-150 range)
                per = int(max(0, min(100, ((avg_angle - 65) / (150 - 65)) * 100)))
                cv2.rectangle(image, (20, 100), (35, 350), (100, 100, 100), 2)
                bar_h = int(250 * (per / 100))
                cv2.rectangle(image, (20, 350 - bar_h), (35, 350), (0, 255, 204), -1)
                cv2.putText(image, f"{per}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # UI Overlay
                cv2.rectangle(image, (0,0), (380, 180), (0,0,0), -1)
                cv2.putText(image, f"Tricep Reps: {good_reps}/{target_reps}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(image, f"Bad Reps: {bad_reps}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(image, f"Stage: {stage}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(image, warning_text, (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow('Apex Coach - Tricep Pushdown', image)
            
            # Exit Conditions (Target hit, 5s timeout, or 'q')
            if good_reps >= target_reps or (good_reps+bad_reps > 0 and time.time() - last_rep_time > 5) or cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    
    status = "Target Reached!" if good_reps >= target_reps else "Workout Complete."
    speak_async(status)
    
    plot_analytics(angles_history, rep_durations, 'TRICEP SPEED (SEC)', 'EXTENSION TRACKER', bad_reps_list)

if __name__ == "__main__":
    start_workout(5)