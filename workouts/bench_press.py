import cv2
import mediapipe as mp
import time
import pyttsx3
import threading
from utils import calculate_angle, plot_analytics

# --- Bench Press Configuration ---
UPPER_THRESH = 145      # Arms fully extended (Lockout)
LOWER_THRESH = 55       # Bar down to chest
REVERSAL_TOLERANCE = 15 # Degrees allowed before flagging a half-rep

# --- Async Voice Engine ---
def speak_async(text):
    """Prevents the camera feed from stuttering while the coach speaks."""
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
    # States: Waiting... -> lowering -> pushing
    stage = "Waiting..." 
    current_rep_form = "Good"
    extreme_angle = 0 # Tracks deepest/highest point to detect half-reps
    
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
            
            # Flip image for a mirror effect and process
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # 1. Workout Countdown
            if not is_ready:
                cv2.putText(image, f"BENCH PRESS READY? {countdown}", (80, 240), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 204), 4)
                cv2.imshow('Apex Coach - Bench Press', image)
                if cv2.waitKey(1000) & 0xFF == ord('q'): break
                countdown -= 1
                if countdown == 0:
                    is_ready = True
                    speak_async("Unrack the bar and go!")
                    rep_start_time = time.time()
                    last_rep_time = time.time()
                continue

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Shoulder, Elbow, Wrist points for both arms
                l_sh, l_el, l_wr = [landmarks[11].x, landmarks[11].y], [landmarks[13].x, landmarks[13].y], [landmarks[15].x, landmarks[15].y]
                r_sh, r_el, r_wr = [landmarks[12].x, landmarks[12].y], [landmarks[14].x, landmarks[14].y], [landmarks[16].x, landmarks[16].y]

                # Calculate average elbow extension
                l_angle = calculate_angle(l_sh, l_el, l_wr)
                r_angle = calculate_angle(r_sh, r_el, r_wr)
                avg_angle = (l_angle + r_angle) / 2.0
                angles_history.append(avg_angle)

                # --- 2. Advanced Rep Logic (State Machine) ---
                status_msg = "STABILIZE"
                warning_color = (255, 255, 255)

                if stage == "Waiting...":
                    # Detect when the user unracks the bar
                    if avg_angle >= UPPER_THRESH - 15:
                        stage = "lowering"
                        extreme_angle = avg_angle
                        current_rep_form = "Good"
                        speak_async("Lower the bar")
                    else:
                        status_msg = "UNRACK THE BAR"

                elif stage == "lowering":
                    extreme_angle = min(extreme_angle, avg_angle) # Find deepest point
                    status_msg = "LOWERING"
                    
                    if avg_angle <= LOWER_THRESH:
                        stage = "pushing"
                        extreme_angle = avg_angle
                    
                    # Half Rep Detection: Reversed direction before touching chest
                    elif extreme_angle < UPPER_THRESH - 15 and avg_angle > extreme_angle + REVERSAL_TOLERANCE:
                        if current_rep_form == "Good":
                            current_rep_form = "Bad"
                            bad_reps += 1
                            speak_async("Half rep! Touch your chest")
                            bad_reps_list.append(f"Rep #{good_reps+bad_reps}: Short range (Bottom)")
                        stage = "pushing"
                        extreme_angle = avg_angle

                elif stage == "pushing":
                    extreme_angle = max(extreme_angle, avg_angle) # Find highest point
                    status_msg = "PUSHING UP"
                    
                    if avg_angle >= UPPER_THRESH:
                        # Success! Full rep cycle completed
                        duration = time.time() - rep_start_time
                        rep_durations.append(duration)
                        
                        if current_rep_form == "Good":
                            good_reps += 1
                            speak_async(str(good_reps))
                        
                        # Reset state for next rep
                        stage = "lowering"
                        current_rep_form = "Good"
                        extreme_angle = avg_angle
                        rep_start_time = time.time()
                        last_rep_time = time.time()
                    
                    # Half Rep Detection: Dropped bar before full lockout
                    elif extreme_angle > LOWER_THRESH + 15 and avg_angle < extreme_angle - REVERSAL_TOLERANCE:
                        if current_rep_form == "Good":
                            current_rep_form = "Bad"
                            bad_reps += 1
                            speak_async("Wrong, lockout your elbows")
                            bad_reps_list.append(f"Rep #{good_reps+bad_reps}: Short range (Top)")
                        stage = "lowering"
                        extreme_angle = avg_angle

                # --- 3. UI and Visuals ---
                # Calculate progress (Lowering bar = progress increases)
                # We map the range 55 (100%) to 145 (0%)
                per = int(max(0, min(100, ((UPPER_THRESH - avg_angle) / (UPPER_THRESH - LOWER_THRESH)) * 100)))
                
                # Draw Progress Bar
                cv2.rectangle(image, (20, 100), (40, 350), (50, 50, 50), -1)
                bar_color = (0, 255, 204) if stage == "pushing" else (0, 165, 255)
                cv2.rectangle(image, (20, 350 - int(2.5 * per)), (40, 350), bar_color, -1)

                # Info Overlay
                cv2.rectangle(image, (0,0), (450, 150), (0,0,0), -1)
                cv2.putText(image, f"BENCH REPS: {good_reps}/{target_reps}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(image, f"BAD REPS: {bad_reps}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(image, f"STATUS: {status_msg}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # Draw skeleton for visual feedback
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow('Apex Coach - Bench Press', image)
            
            # Auto-stop if target reached or idle for 5 seconds
            if good_reps >= target_reps or (good_reps > 0 and time.time() - last_rep_time > 5) or cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    
    speak_async("Workout Complete. Rack the bar.")
    # Show end-of-session charts
    plot_analytics(angles_history, rep_durations, 'BENCH SPEED (SEC)', 'ELBOW EXTENSION', bad_reps_list)

if __name__ == "__main__":
    start_workout(5)