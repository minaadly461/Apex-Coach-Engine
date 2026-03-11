import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from ultralytics import YOLO
from collections import deque

# --- 1. دالة حساب الزاوية ---
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# --- 2. دالة الرسم ---
def draw_text(img, text, pos, color=(255, 255, 255)):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2, cv2.LINE_AA)

# --- 3. الدالة الرئيسية للموديول ---
def process_leg_extension(input_video_path, output_video_path, chart_output_path):
    print(f"[ENGINE] Analyzing Leg Extension (Completion Logic): {input_video_path}")
    
    model = YOLO('models/yolov8n-pose.pt') 
    cap = cv2.VideoCapture(input_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # --- نظام التثبيت (Stabilizer) ---
    SMOOTH_FRAMES = 15 
    keypoints_history = deque(maxlen=SMOOTH_FRAMES)
    CONF_THRESHOLD = 0.5 
    
    reps = 0
    stage = None # "extended" (الرجل مفرودة فوق), "bent" (الرجل متنية تحت)
    angles_history = []
    rep_durations = []
    rep_start_frame = 0

    # زوايا مينا المعدلة
    EXTENSION_THRESHOLD = 160 
    FLEXION_THRESHOLD = 105    

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        results = model(frame, verbose=False)
        
        if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
            if results[0].keypoints.conf is not None and results[0].keypoints.conf.shape[1] > 16:
                raw_kpts = results[0].keypoints.xy[0].cpu().numpy()
                conf = results[0].keypoints.conf[0].cpu().numpy()
                
                # Stabilizer
                if len(keypoints_history) > 0:
                    last_avg = np.mean(keypoints_history, axis=0)
                    for i in range(len(raw_kpts)):
                        if conf[i] < CONF_THRESHOLD:
                            raw_kpts[i] = last_avg[i]

                keypoints_history.append(raw_kpts)
                smoothed_kpts = np.mean(keypoints_history, axis=0)
                
                # نقاط الرجل: Hip (11,12), Knee (13,14), Ankle (15,16)
                r_leg = [smoothed_kpts[12], smoothed_kpts[14], smoothed_kpts[16]]
                l_leg = [smoothed_kpts[11], smoothed_kpts[13], smoothed_kpts[15]]
                
                base_color = (60, 60, 150)
                for leg in [r_leg, l_leg]:
                    if leg[0][0] != 0:
                        cv2.line(frame, tuple(leg[0].astype(int)), tuple(leg[1].astype(int)), base_color, 4, cv2.LINE_AA)
                        cv2.line(frame, tuple(leg[1].astype(int)), tuple(leg[2].astype(int)), base_color, 4, cv2.LINE_AA)
                        for pt in leg: cv2.circle(frame, tuple(pt.astype(int)), 6, (255, 255, 255), -1, cv2.LINE_AA)

                target = r_leg if np.mean(conf[[12,14,16]]) > np.mean(conf[[11,13,15]]) else l_leg
                
                if target[0][0] != 0:
                    angle = calculate_angle(target[0], target[1], target[2])
                    angles_history.append(angle)
                    curr_f = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

                    # --- منطق مينا (فرد ثم ثني = عدة) ---
                    # 1. لو فرد رجله فوق الـ 170، بنسجل إنه وصل للـ Peak
                    if angle > EXTENSION_THRESHOLD:
                        if stage is None or stage == "bent":
                            stage = "extended"
                            rep_start_frame = curr_f # حساب وقت العدة من الفرد للثني
                    
                    # 2. لو رجع تناها تحت الـ 105 وكان أصلاً فارِد (Extended)، هنا تزيد العدة
                    elif angle < FLEXION_THRESHOLD and stage == "extended":
                        reps += 1
                        stage = "bent"
                        duration = (curr_f - rep_start_frame) / fps
                        rep_durations.append(round(duration, 2))

                    draw_text(frame, f"Angle: {int(angle)} deg", (30, 60), color=(0, 255, 255))
                    draw_text(frame, f"Leg Ext Reps: {reps}", (30, 110), color=(0, 255, 0))
                    cv2.putText(frame, f"STATUS: {stage}", (30, height-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        out.write(frame)

    cap.release()
    out.release()

    # --- رسم الداش بورد ---
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor('#0a0a0a')
    for ax in [ax1, ax2]: ax.set_facecolor('#0a0a0a')

    if rep_durations:
        bars = ax1.bar(range(1, len(rep_durations) + 1), rep_durations, color='#ff003c', edgecolor='#ffffff')
        ax1.set_title('TIME PER REP (SECONDS)', fontweight='bold')
        for bar in bars:
            yval = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f"{round(yval, 1)}s", ha='center', color='white')

    if angles_history:
        smoothed_angles = gaussian_filter1d(angles_history, sigma=3)
        ax2.plot(smoothed_angles, color='#00ffcc', linewidth=3)
        ax2.axhline(y=EXTENSION_THRESHOLD, color='yellow', linestyle='--', alpha=0.3)
        ax2.axhline(y=FLEXION_THRESHOLD, color='green', linestyle='--', alpha=0.3)
        ax2.set_title('KNEE ANGLE WAVEFORM', fontweight='bold')

    plt.tight_layout()
    plt.savefig(chart_output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return {
        "total_reps": reps,
        "total_time": round(total_frames / fps, 2),
        "rep_durations": rep_durations
    }