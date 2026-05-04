import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

def calculate_angle(a, b, c):
    """Calculates the angle between three points (e.g., shoulder, elbow, wrist)."""
    a = np.array(a) 
    b = np.array(b) 
    c = np.array(c) 
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def get_torso_lean(shoulder, hip):
    """Calculates torso lean angle to detect swinging or cheating."""
    dx = shoulder[0] - hip[0]
    dy = hip[1] - shoulder[1]
    return np.abs(np.arctan2(dx, dy) * 180.0 / np.pi)

def plot_analytics(angles, durations, title1, title2, bad_reps_list):
    """Generates professional dark-themed analytics charts."""
    if not angles:
        print("No data recorded to plot.")
        return

    # Apply Gaussian smoothing for a clean waveform
    smoothed_angles = gaussian_filter1d(angles, sigma=3)

    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.patch.set_facecolor('#0a0a0a')

    # Chart 1: Time per Repetition
    if durations:
        ax1.bar(range(1, len(durations)+1), durations, color='#ff003c')
        ax1.set_title(title1, color='white', fontweight='bold')
        ax1.set_xlabel('Rep #', color='#aaa')
        ax1.set_ylabel('Seconds', color='#aaa')
    else:
        ax1.text(0.5, 0.5, 'Complete some reps to see speed data', color='#ff003c', ha='center', va='center')

    # Chart 2: Angle Tracking Waveform
    ax2.plot(smoothed_angles, color='#00ffcc', linewidth=2)
    ax2.fill_between(range(len(smoothed_angles)), smoothed_angles, color='#00ffcc', alpha=0.2)
    ax2.set_title(title2, color='white', fontweight='bold')
    ax2.set_xlabel('Time (Frames)', color='#aaa')
    ax2.set_ylabel('Degrees', color='#aaa')

    plt.tight_layout()
    plt.show()

    # Console feedback for form errors
    if bad_reps_list:
        print("\n📝 FORM FEEDBACK:")
        for note in bad_reps_list:
            print(f"- {note}")