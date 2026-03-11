# Apex Coach: Vision Engine

A computer vision and kinematic analysis pipeline designed for automated workout tracking. This engine leverages YOLOv8 pose estimation to extract human skeletal keypoints, track exercise biomechanics, and compute repetition metrics from raw video inputs. 

Designed as a headless backend utility, it processes video streams and outputs structured JSON telemetry alongside annotated media artifacts.

## Core Capabilities

* **Multi-Class Exercise Tracking:** Biomechanical heuristic algorithms tailored for 6 specific modalities:
  * Bicep Curl (`bicep`)
  * Tricep Pushdown (`tricep`)
  * Bench Press (`bench`)
  * Lat Pulldown (`lat`)
  * Shoulder Press (`shoulder`)
  * Leg Extension (`legext`)
* **High-Fidelity Pose Estimation:** Utilizes `yolov8n-pose.pt` for real-time spatial tracking of 17 human keypoints.
* **Artifact Generation:** Automatically renders annotated video files (skeletal overlays, rep counters) and kinematic telemetry charts (angle vs. time).
* **System Integration:** Emits standard JSON payloads via `stdout` for seamless downstream consumption by backend APIs.

## Usage Interface

The pipeline is invoked via a Command Line Interface (CLI). Ensure all dependencies specified in `requirements.txt` are installed prior to execution.

### Syntax
```bash
python app.py -v <input_video_path> -t <exercise_modality>
