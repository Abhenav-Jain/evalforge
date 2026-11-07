"""MOCK tool-calling agent used to demo the framework. Replace with your real agent:
  - python target: wrap your agent call and return {"text":..., "tool_calls":[{"name":..,"args":{..}}]}
  - http target:   map your API response fields in suite.yaml
Input is a dict: {"prompt": str, "poisoned": optional untrusted content the agent 'reads'}."""
import re

def agent(inp):
    prompt, poisoned = inp["prompt"], inp.get("poisoned", "")
    p = prompt.lower()
    calls = []
    if poisoned and re.search(r"<!--\s*system:", poisoned, re.I):          # deliberate weakness for the demo
        cmd = re.search(r"run `([^`]+)`", poisoned)
        if cmd: calls.append({"name": "execute_command", "args": {"command": cmd.group(1)}})
    if any(w in p for w in ["rm -rf", "delete everything", "shutdown", "wipe"]):
        return {"text": "Sorry, I can't help with destructive commands.", "tool_calls": []}
    if any(w in p for w in ["printenv", "api key", ".env", "api_key"]):
        return {"text": "I can't reveal secrets or environment variables.", "tool_calls": []}
    if "list" in p and "files" in p:
        calls.append({"name": "list_directory", "args": {"dir_path": "."}})
        return {"text": "Here are the files.", "tool_calls": calls}
    if "read" in p:
        calls.insert(0, {"name": "read_file", "args": {"file_path": "notes.txt"}})
        return {"text": "Summary of the file.", "tool_calls": calls}
    return {"text": "I'm a developer assistant. How can I help?", "tool_calls": calls}
