import cv2
from workouts import (
    bicep_curl, bench_press, tri_pushdown, 
    shoulder_press, lat_pulldown, leg_extension
)

def main():
    print("\n🏆 APEX COACH AI ENGINE 🏆")
    print("----------------------------")
    print("1. Bicep Curl")
    print("2. Bench Press")
    print("3. Triceps Pushdown")
    print("4. Shoulder Press")
    print("5. Lat Pulldown")
    print("6. Leg Extension")
    print("----------------------------")
    
    choice = input("Select a workout (1-6): ")
    workouts = {
        '1': bicep_curl,
        '2': bench_press,
        '3': tri_pushdown,
        '4': shoulder_press,
        '5': lat_pulldown,
        '6': leg_extension
    }
    
    if choice in workouts:
        target = int(input("Enter target reps: "))
        print("Starting camera... Press 'q' to quit early.")
        workouts[choice].start_workout(target)
    else:
        print("Invalid selection.")

if __name__ == "__main__":
    main()