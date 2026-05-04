import cv2
import mediapipe as mp
import time
import pyttsx3
import threading
from utils import calculate_angle, get_torso_lean, plot_analytics

# --- Constants من كولاب بالظبط ---
UPPER_THRESH = 150
LOWER_THRESH = 45
REVERSAL_TOLERANCE = 20

# --- Async Voice Engine ---
def speak_async(text):
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
    stage = "Waiting..."
    current_rep_form = "Good"
    extreme_angle = 0 # لمتابعة أعلى/أقل زاوية وكشف الـ Half Reps
    
    angles_history, rep_durations, bad_reps_list = [], [], []
    
    is_ready = False
    countdown = 3
    last_rep_time = time.time()
    rep_start_time = time.time()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # 1. Countdown
            if not is_ready:
                cv2.putText(image, f"SHOULDER PRESS: {countdown}", (100, 240), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 204), 4)
                cv2.imshow('Apex Coach - Shoulder Press', image)
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
                
                # النقاط المطلوبة (كتاف، كوع، معصم، وسط)
                l_sh, l_el, l_wr = [landmarks[11].x, landmarks[11].y], [landmarks[13].x, landmarks[13].y], [landmarks[15].x, landmarks[15].y]
                r_sh, r_el, r_wr = [landmarks[12].x, landmarks[12].y], [landmarks[14].x, landmarks[14].y], [landmarks[16].x, landmarks[16].y]
                l_hip, r_hip = [landmarks[23].x, landmarks[23].y], [landmarks[24].x, landmarks[24].y]

                avg_angle = (calculate_angle(l_sh, l_el, l_wr) + calculate_angle(r_sh, r_el, r_wr)) / 2.0
                avg_sway = (get_torso_lean(l_sh, l_hip) + get_torso_lean(r_sh, r_hip)) / 2.0
                angles_history.append(avg_angle)

                # --- 2. Form Checking (Arching Back) ---
                warning_text, color = "FORM: GOOD", (204, 255, 0)
                if avg_sway > 20:
                    if current_rep_form == "Good":
                        current_rep_form = "Bad"
                        bad_reps += 1
                        speak_async("Don't arch your back")
                        bad_reps_list.append(f"Rep #{good_reps+bad_reps}: Back arched")
                    warning_text, color = "WARNING: ARCHING BACK!", (60, 0, 255)

                # --- 3. Full Rep & Half Rep Logic (State Machine) ---
                if stage == "Waiting...":
                    # البداية لازم تكون من تحت (الوزن عند الأكتاف)
                    if avg_angle <= LOWER_THRESH + 15:
                        stage = "pushing_up"
                        extreme_angle = avg_angle
                        current_rep_form = "Good"
                    else:
                        warning_text = "Lower hands to start!"

                elif stage == "pushing_up":
                    extreme_angle = max(extreme_angle, avg_angle)
                    
                    if avg_angle >= UPPER_THRESH:
                        stage = "pulling_down"
                        extreme_angle = avg_angle
                    
                    # كشف الـ Half Rep (عكست الحركة قبل ما توصل لفوق)
                    elif extreme_angle > LOWER_THRESH + 20 and avg_angle < extreme_angle - REVERSAL_TOLERANCE:
                        if current_rep_form == "Good":
                            current_rep_form = "Bad"
                            bad_reps += 1
                            speak_async("Wrong, push all the way up")
                            bad_reps_list.append(f"Rep #{good_reps+bad_reps}: Half rep (not high enough)")
                        stage = "pulling_down"
                        extreme_angle = avg_angle

                elif stage == "pulling_down":
                    extreme_angle = min(extreme_angle, avg_angle)
                    
                    if avg_angle <= LOWER_THRESH:
                        # اكتمال العدة!
                        duration = time.time() - rep_start_time
                        rep_durations.append(duration)
                        
                        if current_rep_form == "Good":
                            good_reps += 1
                            speak_async(str(good_reps))
                        
                        # ريست للعدة الجاية
                        stage = "pushing_up"
                        current_rep_form = "Good"
                        extreme_angle = avg_angle
                        rep_start_time = time.time()
                        last_rep_time = time.time()
                    
                    # كشف الـ Half Rep (طلعت تاني قبل ما تنزل للأخر)
                    elif extreme_angle < UPPER_THRESH - 20 and avg_angle > extreme_angle + REVERSAL_TOLERANCE:
                        if current_rep_form == "Good":
                            current_rep_form = "Bad"
                            bad_reps += 1
                            speak_async("Wrong, lower all the way down")
                            bad_reps_list.append(f"Rep #{good_reps+bad_reps}: Half rep (not low enough)")
                        stage = "pushing_up"
                        extreme_angle = avg_angle

                # 4. Progress Bar (45-150)
                per = int(max(0, min(100, ((avg_angle - LOWER_THRESH) / (UPPER_THRESH - LOWER_THRESH)) * 100)))
                cv2.rectangle(image, (20, 100), (35, 350), (100, 100, 100), 2)
                bar_h = int(250 * (per / 100))
                bar_color = (204, 255, 0) if stage == "pushing_up" else (60, 0, 255)
                cv2.rectangle(image, (20, 350 - bar_h), (35, 350), bar_color, -1)

                # UI Overlay
                cv2.rectangle(image, (0,0), (420, 180), (0,0,0), -1)
                cv2.putText(image, f"Shoulder Reps: {good_reps}/{target_reps}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(image, f"Bad Reps: {bad_reps}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(image, f"Stage: {stage}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(image, warning_text, (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow('Apex Coach - Shoulder Press', image)
            
            # Exit
            if good_reps >= target_reps or (good_reps+bad_reps > 0 and time.time() - last_rep_time > 5) or cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    
    status = "Target Reached!" if good_reps >= target_reps else "Workout Complete."
    speak_async(status)
    
    plot_analytics(angles_history, rep_durations, 'SHOULDER PRESS SPEED', 'ELBOW EXTENSION', bad_reps_list)

if __name__ == "__main__":
    start_workout(5)