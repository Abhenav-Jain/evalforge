"""UNTESTED template: evaluates the running LivePrep backend over its WebSocket (text in -> transcript + tool events out).
!! LivePrep executes tools for real. Run it ONLY against a throwaway workspace/container (PROJECT_WORKSPACE_ROOT=/tmp/sandbox).
pip install websockets"""
import json, os, time
from websockets.sync.client import connect

URL = os.getenv("LIVEPREP_WS", "ws://localhost:8000/ws/live")

def ask(inp, idle_s=8.0):
    prompt = inp["prompt"] if isinstance(inp, dict) else inp   # poisoned-content cases are not supported over the live API
    text, calls = [], []
    with connect(URL, open_timeout=15) as ws:
        t_end = time.time() + 60
        while time.time() < t_end:                                  # wait for session to be live
            m = json.loads(ws.recv(timeout=15))
            if m.get("type") == "session_status" and m.get("status") == "connected": break
            if m.get("type") == "error": raise RuntimeError(m.get("message"))
        ws.send(json.dumps({"type": "text", "text": prompt}))
        while True:                                                 # collect until the agent goes quiet
            try: m = json.loads(ws.recv(timeout=idle_s))
            except TimeoutError: break
            if m["type"] == "agent_transcription": text.append(m["text"])
            elif m["type"] == "tool_started": calls.append({"name": m["name"], "args": m.get("args", {})})
            elif m["type"] == "error": raise RuntimeError(m.get("message"))
    return {"text": "".join(text), "tool_calls": calls}
