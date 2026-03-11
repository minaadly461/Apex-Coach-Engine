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

# --- 2. دالة الرسم (ثيم البايسيبس النيون) ---
def draw_text(img, text, pos, color=(255, 255, 255)):
    # ظل أسود
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 5, cv2.LINE_AA)
    # النص الملون (أصفر أو أخضر نيون)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2, cv2.LINE_AA)

# --- 3. الدالة الرئيسية للموديول ---
def process_shoulder_press(input_video_path, output_video_path, chart_output_path):
    print(f"[ENGINE] Starting Shoulder Press Analysis: {input_video_path}")
    
    model = YOLO('models/yolov8n-pose.pt')
    cap = cv2.VideoCapture(input_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # متغيرات المنطق والتنعيم
    SMOOTH_FRAMES = 5
    keypoints_history = deque(maxlen=SMOOTH_FRAMES)
    reps = 0
    stage = None 
    angles_history = []
    rep_durations = []
    rep_start_frame = 0

    # Thresholds للكتف بناءً على كود التيست بتاعك
    EXTENSION_THRESHOLD = 140 # مفرود فوق
    FLEXION_THRESHOLD = 70    # متني عند الكتف

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        results = model(frame, verbose=False)
        
        if results[0].keypoints is not None and len(results[0].keypoints.xy[0]) > 10:
            raw_kpts = results[0].keypoints.xy[0].cpu().numpy()
            conf = results[0].keypoints.conf[0].cpu().numpy()
            
            keypoints_history.append(raw_kpts)
            smoothed_kpts = np.mean(keypoints_history, axis=0)
            
            # تحديد النقط (5,7,9 للشمال و 6,8,10 لليمين)
            r_arm = [smoothed_kpts[6], smoothed_kpts[8], smoothed_kpts[10]]
            l_arm = [smoothed_kpts[5], smoothed_kpts[7], smoothed_kpts[9]]

            # الرسم (ثيم بايسيبس: خطوط حمراء باهتة ونقط بيضاء)
            base_color = (60, 60, 150)
            for arm in [r_arm, l_arm]:
                if arm[0][0] != 0:
                    cv2.line(frame, tuple(arm[0].astype(int)), tuple(arm[1].astype(int)), base_color, 3, cv2.LINE_AA)
                    cv2.line(frame, tuple(arm[1].astype(int)), tuple(arm[2].astype(int)), base_color, 3, cv2.LINE_AA)
                    for pt in arm: cv2.circle(frame, tuple(pt.astype(int)), 6, (255, 255, 255), -1, cv2.LINE_AA)

            # اختيار الدراع الأوضح
            target = r_arm if np.mean(conf[[6,8,10]]) > np.mean(conf[[5,7,9]]) else l_arm
            
            if target[0][0] != 0:
                angle = calculate_angle(target[0], target[1], target[2])
                angles_history.append(angle)
                curr_f = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

                # --- منطق العد (Shoulder Press) ---
                if angle > EXTENSION_THRESHOLD:
                    if stage is None or stage == "down":
                        stage = "up"
                        rep_start_frame = curr_f
                
                elif angle < FLEXION_THRESHOLD and stage == "up":
                    reps += 1
                    stage = "down"
                    duration = (curr_f - rep_start_frame) / fps
                    rep_durations.append(round(duration, 2))

                # كتابة البيانات (أصفر للزاوية وأخضر للعد)
                draw_text(frame, f"Angle: {int(angle)} deg", (30, 60), color=(0, 255, 255))
                draw_text(frame, f"Shoulder Reps: {reps}", (30, 110), color=(0, 255, 0))

        out.write(frame)

    cap.release()
    out.release()

    # --- رسم الداش بورد (Gym Theme) ---
    
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor('#0a0a0a')
    ax1.set_facecolor('#0a0a0a')
    ax2.set_facecolor('#0a0a0a')

    if rep_durations:
        bars = ax1.bar(range(1, len(rep_durations) + 1), rep_durations, color='#ff003c', edgecolor='#ffffff')
        ax1.set_title('TIME PER REP (SHOULDER)', fontweight='bold', pad=15)
        for bar in bars:
            yval = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f"{round(yval, 1)}s", ha='center', color='white')

    if angles_history:
        smoothed_angles = gaussian_filter1d(angles_history, sigma=3)
        time_axis = np.linspace(0, total_frames/fps, len(smoothed_angles))
        ax2.plot(time_axis, smoothed_angles, color='#00ffcc', linewidth=3)
        ax2.fill_between(time_axis, smoothed_angles, color='#00ffcc', alpha=0.1)
        ax2.set_title('SHOULDER ANGLE TRACKER', fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig(chart_output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return {
        "total_reps": reps,
        "total_time": round(total_frames / fps, 2),
        "rep_durations": rep_durations
    }