"""Resumable per-session state: atomic writes, option hashes, stage markers.

A stage is "done" for a session when ``status.json`` holds its name with the
hash of the options it ran with. Running it again with the same options is a
no-op; with other options it runs again. Interrupting a stage can therefore
never leave a session half-marked: the marker is written last.
"""

import hashlib
import json
import os
import time
from dataclasses import asdict, is_dataclass


def atomic_write(path, data, mode="w"):
    """Write ``data`` to ``path`` through a temporary file in the same folder."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, mode, encoding=None if "b" in mode else "utf-8") as f:
        f.write(data)
    os.replace(tmp, path)


def atomic_json(path, obj):
    atomic_write(path, json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True, default=_jsonable))


def _jsonable(obj):
    if is_dataclass(obj):
        return asdict(obj)
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def options_hash(*objects):
    """Short stable hash of options (dicts, dataclasses, strings, numbers)."""
    blob = json.dumps(objects, sort_keys=True, default=_jsonable).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]


def session_dir(work_root, session):
    return os.path.join(work_root, session)


def list_sessions(work_root):
    """Sessions that have a work folder, sorted."""
    if not os.path.isdir(work_root):
        return []
    return sorted(d for d in os.listdir(work_root) if os.path.isdir(os.path.join(work_root, d)) and not d.startswith("."))


class SessionStatus:
    """Stage markers of one session, stored in ``<work_dir>/status.json``."""

    FILE = "status.json"

    def __init__(self, work_dir):
        self.work_dir = work_dir
        self.path = os.path.join(work_dir, self.FILE)
        self.stages = {}
        if os.path.isfile(self.path):
            with open(self.path, encoding="utf-8") as f:
                self.stages = json.load(f)

    def get(self, stage):
        return self.stages.get(stage)

    def is_done(self, stage, options=None):
        """Whether ``stage`` finished with these options (any options if None)."""
        entry = self.stages.get(stage)
        if not entry:
            return False
        return options is None or entry.get("hash") == options_hash(options)

    def mark(self, stage, options=None, **info):
        entry = {"hash": options_hash(options), "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
        entry.update(info)
        self.stages[stage] = entry
        atomic_json(self.path, self.stages)
        return entry

    def clear(self, stage):
        if stage in self.stages:
            del self.stages[stage]
            atomic_json(self.path, self.stages)
