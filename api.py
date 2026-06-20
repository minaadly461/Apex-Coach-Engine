# api.py
import json
import base64
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from engine import EXERCISES

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    with open("static/index.html") as f:
        return HTMLResponse(f.read())


@app.get("/health")
async def health():
    return {"status": "ok", "exercises": list(EXERCISES.keys())}


@app.websocket("/ws/{exercise}/{target_reps}")
async def workout_websocket(
    websocket: WebSocket,
    exercise: str,
    target_reps: int
):
    await websocket.accept()

    # Validate exercise
    if exercise not in EXERCISES:
        await websocket.send_json({
            "event": "error",
            "message": f"Unknown exercise: {exercise}. Available: {list(EXERCISES.keys())}"
        })
        await websocket.close()
        return

    # Init engine
    engine = EXERCISES[exercise](target_reps=target_reps)

    await websocket.send_json({
        "event": "ready",
        "exercise": exercise,
        "target_reps": target_reps,
        "say": f"Starting {exercise.replace('_', ' ')}. Get ready!"
    })

    try:
        while True:
            # Receive frame from browser
            data = await websocket.receive_json()

            if data.get("type") == "frame":
                # Decode base64 frame
                frame_bytes = base64.b64decode(data["frame"])

                # Process frame through engine
                event = engine.process_frame(frame_bytes)

                if event is None:
                    continue

                # Send event back to browser
                await websocket.send_json(event)

                # If workout complete, close connection
                if event.get("event") == "workout_complete":
                    await asyncio.sleep(1)
                    await websocket.close()
                    break

            elif data.get("type") == "stop":
                analytics = engine.build_analytics()
                analytics["say"] = "Workout stopped."
                await websocket.send_json(analytics)
                await asyncio.sleep(1)
                await websocket.close()
                break

    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)