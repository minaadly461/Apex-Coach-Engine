import os
import argparse
import json

from workouts.Bicep import process_bicep_curl
from workouts.Tricep import process_tricep_pushdown
from workouts.Bench import process_bench_press
from workouts.Lat import process_lat_pulldown
from workouts.Shoulder import process_shoulder_press
from workouts.LegExtension import process_leg_extension

def main():
    parser = argparse.ArgumentParser(description="Apex Coach - Video Processing Engine")
    
    parser.add_argument('-v', '--video', type=str, required=True, help="Path to the input video file.")
    parser.add_argument('-t', '--type', type=str, required=True, 
                        choices=['bicep', 'tricep', 'bench', 'lat', 'shoulder', 'legext'], 
                        help="Target exercise type for analysis.")
    parser.add_argument('-o', '--output', type=str, default="output", help="Directory to save output files.")

    args = parser.parse_args()

    input_video = args.video
    exercise_type = args.type.lower()
    output_dir = args.output

    if not os.path.exists(input_video):
        print(json.dumps({"status": "error", "message": f"Video file not found: {input_video}"}))
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    base_name = os.path.splitext(os.path.basename(input_video))[0]
    output_video_path = os.path.join(output_dir, f"{base_name}_{exercise_type}_out.mp4")
    output_chart_path = os.path.join(output_dir, f"{base_name}_{exercise_type}_chart.png")

    results = None

    try:
        if exercise_type == 'bicep':
            results = process_bicep_curl(input_video, output_video_path, output_chart_path)
        elif exercise_type == 'tricep':
            results = process_tricep_pushdown(input_video, output_video_path, output_chart_path)
        elif exercise_type == 'bench':
            results = process_bench_press(input_video, output_video_path, output_chart_path)
        elif exercise_type == 'lat':
            results = process_lat_pulldown(input_video, output_video_path, output_chart_path)
        elif exercise_type == 'shoulder':
            results = process_shoulder_press(input_video, output_video_path, output_chart_path)
        elif exercise_type == 'legext':
            results = process_leg_extension(input_video, output_video_path, output_chart_path)
        
        if results:
            # Structuring the response for backend API integration
            response = {
                "status": "success",
                "exercise": exercise_type,
                "data": {
                    "total_reps": results.get('total_reps', 0),
                    "total_time_seconds": results.get('total_time', 0),
                    "rep_durations": results.get('rep_durations', [])
                },
                "files": {
                    "video_path": output_video_path,
                    "chart_path": output_chart_path
                }
            }
            print(json.dumps(response))
            
    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}))

if __name__ == "__main__":
    main()