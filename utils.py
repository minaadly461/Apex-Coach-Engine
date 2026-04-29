import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

def calculate_angle(a, b, c):
    """Calculates the angle between three joints."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

def get_torso_lean(shoulder, hip):
    """Calculates how far the torso leans from a vertical axis."""
    dx = shoulder[0] - hip[0]
    dy = hip[1] - shoulder[1]
    return np.abs(np.arctan2(dx, dy) * 180.0 / np.pi)

def plot_analytics(angles, durations, title_time, title_angle, bad_reps=None):
    """Prints the fault report and generates the matplotlib performance charts."""
    if bad_reps is not None:
        print("\n" + "="*50)
        print("🚨 FAULT REPORT")
        print("="*50)
        if bad_reps:
            for entry in bad_reps: print(f"❌ {entry}")
        else:
            print("✅ PERFECT FORM! No faults recorded.")
        print("="*50 + "\n")

    if not angles: return
    smoothed = gaussian_filter1d(angles, sigma=3)
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.canvas.manager.set_window_title('Apex Coach Analytics')
    
    if durations:
        ax1.bar(range(1, len(durations)+1), durations, color='#00ffcc')
        ax1.set_title(title_time)
        ax1.set_xlabel('Rep #')
    else:
        ax1.text(0.5, 0.5, 'No complete reps', color='#ff003c', ha='center', va='center')
        ax1.set_title(title_time)
        
    ax2.plot(smoothed, color='#00ffcc', linewidth=2)
    ax2.fill_between(range(len(smoothed)), smoothed, color='#00ffcc', alpha=0.2)
    ax2.set_title(title_angle)
    ax2.set_xlabel('Frame Index')
    ax2.set_ylabel('Degrees')
    
    plt.tight_layout()
    plt.show()