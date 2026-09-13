"""Remembered answers -- reused only when the person can see and change them.

Answering "who is the audience?" for the fifth time is tedious, so previous
answers are worth offering again. But quietly applying them is worse than not
remembering at all: the plan silently reflects a decision made for a different
task, and there is no visible reason why.

So everything here is *offered*, never applied. The caller gets the value plus
where it came from, and is expected to show both.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import PROJECT_ROOT

RECALL_FILE = PROJECT_ROOT / ".orchestrator-recall.json"
#: Keep the store small and relevant; old answers are more likely to mislead.
MAX_ENTRIES = 200


def _load() -> Dict[str, Any]:
    if not RECALL_FILE.exists():
        return {"answers": {}}
    try:
        data = json.loads(RECALL_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {"answers": {}}
    except (json.JSONDecodeError, OSError):
        # A corrupt recall file must never break planning.
        return {"answers": {}}


def _save(data: Dict[str, Any]) -> None:
    answers = data.get("answers", {})
    if len(answers) > MAX_ENTRIES:
        oldest = sorted(answers.items(), key=lambda kv: kv[1].get("at", 0))
        for key, _ in oldest[: len(answers) - MAX_ENTRIES]:
            answers.pop(key, None)
    try:
        RECALL_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        pass  # remembering is a convenience, never a hard requirement


def remember(name: str, value: Any, task: str = "") -> None:
    """Record one answer so it can be *offered* next time."""
    if value in (None, ""):
        return
    data = _load()
    data.setdefault("answers", {})[name] = {
        "value": value, "at": time.time(), "task": task[:120],
    }
    _save(data)


def remember_all(answers: Dict[str, Any], task: str = "") -> None:
    for name, value in (answers or {}).items():
        remember(name, value, task)


def recall(name: str) -> Optional[Dict[str, Any]]:
    """The remembered entry for one question name, if any."""
    return _load().get("answers", {}).get(name)


def annotate(questions: List[Any], task: str = "") -> List[Dict[str, Any]]:
    """Serialise questions, attaching any remembered answer as a *suggestion*.

    The returned dicts carry ``remembered`` and ``remembered_from`` so the UI
    can pre-fill the field while showing plainly that the value came from a
    previous task and can be changed.
    """
    annotated = []
    for question in questions:
        payload = question.to_dict()
        previous = recall(question.name)
        if previous:
            payload["remembered"] = previous["value"]
            payload["remembered_from"] = previous.get("task", "")
        annotated.append(payload)
    return annotated


def forget_all() -> None:
    """Drop every remembered answer."""
    _save({"answers": {}})
