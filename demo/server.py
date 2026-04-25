import json
import os
import time
import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

# Enable CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

LOG_FILE = "../training_log.jsonl"

async def tail_log_file():
    """Continuously read the log file and yield new lines."""
    if not os.path.exists(LOG_FILE):
        # Create empty log if missing
        with open(LOG_FILE, "w") as f:
            pass
            
    with open(LOG_FILE, "r") as f:
        # Go to end of file
        f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if not line:
                await asyncio.sleep(0.5)
                continue
            yield line

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Dashboard connected.")
    try:
        # Send last 20 lines initially
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r") as f:
                lines = f.readlines()[-20:]
                for line in lines:
                    await websocket.send_text(line)
        
        # Stream new lines
        async for line in tail_log_file():
            await websocket.send_text(line)
    except Exception as e:
        print(f"WebSocket Error: {e}")
    finally:
        print("Dashboard disconnected.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
