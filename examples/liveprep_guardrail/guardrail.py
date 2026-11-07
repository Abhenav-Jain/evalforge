"""Function under test: a verbatim copy of LivePrep's regex guardrail (backend/tools.py).
Point the suite at your real file instead by changing the callable, e.g. 'tools:is_command_safe'."""
import re
PATTERNS = [r"\brm\s+-[rf]{1,2}\s+[/~]", r"\bdel\s+/[sfaq]{1,4}\s+[c-zC-Z]:\\", r"\bformat\s+[c-zC-Z]:", r"\bFormat-Volume\b",
            r"\bdiskpart\b", r"\bStop-Computer\b", r"\bRestart-Computer\b", r"\bshutdown\b", r"\breg\s+delete\b",
            r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:", r"\bdrop\s+database\b", r"\bmkfs\b", r"\bdd\s+if=.*of=/dev/sd"]

def decide(cmd):
    return "blocked" if any(re.search(p, cmd, re.I) for p in PATTERNS) else "allowed"
