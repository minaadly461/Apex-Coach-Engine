
from utils import calculate_angle, get_torso_lean 
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

def plot_leg_analytics(angles, durations, bad_reps_list):
    """ speical version for legs"""
    if not angles:
        print("No leg data recorded.")
        return

    smoothed = gaussian_filter1d(angles, sigma=3)
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.patch.set_facecolor('#0a0a0a')

    if durations:
        ax1.bar(range(1, len(durations)+1), durations, color='#00ffcc')
        ax1.set_title('LEG EXTENSION: REP SPEED', color='white')
    
    ax2.plot(smoothed, color='#00ffcc', linewidth=2)
    ax2.set_title('KNEE ANGLE TRACKER', color='white')

    plt.tight_layout()
    plt.show()