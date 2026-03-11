import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from ultralytics import YOLO
from collections import deque

# --- 1. دالة حساب الزاوية ---
def calculate_angle(a, b, c):
    a = np.array(a) # Shoulder
    b = np.array(b) # Elbow
    c = np.array(c) # Wrist
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# دالة مساعدة لكتابة نص بحدود سوداء عشان يكون مقروء على أي خلفية
def draw_text(img, text, pos, color=(255, 255, 255)):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 5, cv2.LINE_AA) # Shadow/Outline
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2, cv2.LINE_AA) # Text


# --- 2. الدالة الرئيسية للموديول ---
def process_bicep_curl(input_video_path, output_video_path, chart_output_path):
    print(f"Starting to process: {input_video_path}")
    
    # --- إعدادات YOLOv8 ---
    # لو حاطط الموديل جوه فولدر model، خليها 'model/yolov8n-pose.pt'
    model = YOLO('models/yolov8n-pose.pt') 

    cap = cv2.VideoCapture(input_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # --- متغيرات التنعيم ---
    SMOOTH_FRAMES = 5
    keypoints_history = deque(maxlen=SMOOTH_FRAMES)

    reps = 0
    stage = None
    angles_history = []
    rep_durations = []
    rep_start_frame = 0

    # --- 3. معالجة الفيديو بـ YOLO ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        
        if results[0].keypoints is not None and len(results[0].keypoints.xy[0]) > 10:
            raw_keypoints = results[0].keypoints.xy[0].cpu().numpy()
            keypoints_conf = results[0].keypoints.conf[0].cpu().numpy()
            
            keypoints_history.append(raw_keypoints)
            smoothed_keypoints = np.mean(keypoints_history, axis=0)
            
            r_shoulder = smoothed_keypoints[6]
            r_elbow = smoothed_keypoints[8]
            r_wrist = smoothed_keypoints[10]
            
            l_shoulder = smoothed_keypoints[5]
            l_elbow = smoothed_keypoints[7]
            l_wrist = smoothed_keypoints[9]

            conf_right = np.mean([keypoints_conf[6], keypoints_conf[8], keypoints_conf[10]])
            conf_left = np.mean([keypoints_conf[5], keypoints_conf[7], keypoints_conf[9]])

            # توحيد الألوان بناءً على طلبك 
            base_color = (60, 60, 150) # اللون المطلوب BGR
            thickness = 3 # سمك موحد للدراعين
            
            # رسم الدواير البيضا
            for pt in [r_shoulder, r_elbow, r_wrist, l_shoulder, l_elbow, l_wrist]:
                if pt[0] != 0: 
                    cv2.circle(frame, tuple(pt.astype(int)), 6, (255, 255, 255), -1, cv2.LINE_AA)

            # رسم الخطوط للدراعين بنفس اللون والسمك
            if r_shoulder[0] != 0:
                cv2.line(frame, tuple(r_shoulder.astype(int)), tuple(r_elbow.astype(int)), base_color, thickness, cv2.LINE_AA)
                cv2.line(frame, tuple(r_elbow.astype(int)), tuple(r_wrist.astype(int)), base_color, thickness, cv2.LINE_AA)
            if l_shoulder[0] != 0:
                cv2.line(frame, tuple(l_shoulder.astype(int)), tuple(l_elbow.astype(int)), base_color, thickness, cv2.LINE_AA)
                cv2.line(frame, tuple(l_elbow.astype(int)), tuple(l_wrist.astype(int)), base_color, thickness, cv2.LINE_AA)

            # تحديد الدراع الأوضح عشان نحسب منه الزاوية اللي هتتعرض
            if conf_right > conf_left:
                target_angle_points = (r_shoulder, r_elbow, r_wrist)
            else:
                target_angle_points = (l_shoulder, l_elbow, l_wrist)

            # حساب الزاوية، عرضها فوق على الشمال، والعد
            if target_angle_points[0][0] != 0: 
                angle = calculate_angle(target_angle_points[0], target_angle_points[1], target_angle_points[2])
                angles_history.append(angle)

                # منطق العد 
                current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                if angle > 160: 
                    if stage == "up": 
                        reps += 1
                        frames_taken = current_frame - rep_start_frame
                        rep_durations.append(frames_taken / fps)
                        stage = "down" 
                        rep_start_frame = current_frame 
                    elif stage is None:
                        stage = "down" 
                        rep_start_frame = current_frame
                        
                if angle < 30 and stage == 'down': 
                    stage = "up" 

                # طباعة الزاوية والعداد فوق على الشمال
                draw_text(frame, f"Angle: {int(angle)} deg", (30, 60), color=(0, 255, 255)) 
                draw_text(frame, f"Reps: {reps}", (30, 110), color=(0, 255, 0)) 

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Video processing done.")

    # --- 4. رسم وحفظ الـ Charts بستايل الـ Gym Theme ---
    total_time_taken = total_frames / fps if fps > 0 else 0
    if not angles_history:
        angles_history = [0] * int(total_frames)
    smoothed_angles = gaussian_filter1d(angles_history, sigma=3)

    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor('#0a0a0a')
    ax1.set_facecolor('#0a0a0a')
    ax2.set_facecolor('#0a0a0a')

    # Bar Chart (وقت كل عدة)
    if rep_durations:
        bars = ax1.bar(range(1, len(rep_durations) + 1), rep_durations, color='#ff003c', edgecolor='#ffffff', linewidth=1)
        ax1.set_title('TIME PER REP', fontsize=16, fontweight='bold', color='#ffffff', pad=15)
        ax1.set_xlabel('REP NUMBER', color='#888888', fontweight='bold', labelpad=10)
        ax1.set_ylabel('SECONDS', color='#888888', fontweight='bold', labelpad=10)
        ax1.set_xticks(range(1, len(rep_durations) + 1))
        
        for bar in bars:
            yval = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f"{round(yval, 1)}s", ha='center', va='bottom', color='white', fontweight='bold', fontsize=10)
    else:
        ax1.text(0.5, 0.5, 'NO REPS COMPLETED', ha='center', va='center', fontsize=16, color='#ff003c', fontweight='bold')
        ax1.set_title('TIME PER REP')

    ax1.grid(True, linestyle=':', color='#333333', axis='y')

    # Waveform Chart (تتبع الزاوية)
    time_axis = np.linspace(0, total_time_taken, len(smoothed_angles))
    ax2.plot(time_axis, smoothed_angles, color='#00ffcc', linewidth=3)
    ax2.fill_between(time_axis, smoothed_angles, color='#00ffcc', alpha=0.1) 
    ax2.set_title('ELBOW ANGLE TRACKER', fontsize=16, fontweight='bold', color='#ffffff', pad=15)
    ax2.set_xlabel('TIME (SECONDS)', color='#888888', fontweight='bold', labelpad=10)
    ax2.set_ylabel('ANGLE (DEGREES)', color='#888888', fontweight='bold', labelpad=10)
    ax2.grid(True, linestyle='--', color='#222222')

    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#555555')
        ax.spines['bottom'].set_color('#555555')

    plt.tight_layout()
    
    # حفظ الصورة بدل عرضها
    plt.savefig(chart_output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Charts saved to: {chart_output_path}")
    
    # بنرجع الإحصائيات عشان لو التيم حب يستخدم الأرقام
    return {
        "total_reps": reps,
        "total_time": round(total_time_taken, 2),
        "rep_durations": [round(d, 2) for d in rep_durations]
    }