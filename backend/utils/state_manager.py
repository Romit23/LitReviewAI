"""
State Manager — job state, WebSocket connection registry, and stage checkpoints.

WebSocket broadcasting
──────────────────────
Every call to update() or add_log() immediately pushes a JSON event to every
WebSocket client subscribed to that job.  The WS message schema is:

  { "type": "progress",  "progress": 42, "current_agent": "...", "status": "running" }
  { "type": "log",       "timestamp": "...", "level": "info", "message": "..." }
  { "type": "papers",    "papers": [...] }
  { "type": "completed"  }
  { "type": "failed",    "error": "..." }

Stage checkpoints (for error recovery)
───────────────────────────────────────
The pipeline saves its output after each stage:
  stage 1 → papers_raw
  stage 2 → papers_extracted
  stage 3 → papers_summarized
  stage 4 → analysis
  stage 5 → completed

On retry, run_pipeline() reads the checkpoint and resumes from the next stage.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket connection registry (per job_id)
# ─────────────────────────────────────────────────────────────────────────────

class _WSRegistry:
    """Thread-safe registry mapping job_id → set of connected WebSocket objects."""

    def __init__(self):
        self._connections: Dict[str, Set[Any]] = {}
        self._lock = threading.Lock()

    def subscribe(self, job_id: str, ws: Any) -> None:
        with self._lock:
            self._connections.setdefault(job_id, set()).add(ws)

    def unsubscribe(self, job_id: str, ws: Any) -> None:
        with self._lock:
            bucket = self._connections.get(job_id, set())
            bucket.discard(ws)

    def get_connections(self, job_id: str) -> List[Any]:
        with self._lock:
            return list(self._connections.get(job_id, set()))


_ws_registry = _WSRegistry()


def ws_subscribe(job_id: str, ws: Any) -> None:
    _ws_registry.subscribe(job_id, ws)


def ws_unsubscribe(job_id: str, ws: Any) -> None:
    _ws_registry.unsubscribe(job_id, ws)


async def _broadcast(job_id: str, payload: Dict) -> None:
    """Push a JSON payload to every WS client watching job_id. Fire-and-forget."""
    conns = _ws_registry.get_connections(job_id)
    if not conns:
        return
    text = json.dumps(payload)
    dead: List[Any] = []
    for ws in conns:
        try:
            await ws.send_text(text)
        except Exception:
            # Connection already closed; mark for removal
            dead.append(ws)
    for ws in dead:
        _ws_registry.unsubscribe(job_id, ws)


def _fire(job_id: str, payload: Dict) -> None:
    """Schedule a broadcast on whatever event loop is running (or silently skip)."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(_broadcast(job_id, payload))
    except RuntimeError:
        pass  # No event loop — background thread context; skip WS


# ─────────────────────────────────────────────────────────────────────────────
# State Manager
# ─────────────────────────────────────────────────────────────────────────────

class StateManager:
    """
    Central in-memory store for all pipeline jobs.

    Job dict keys
    ─────────────
    job_id, topic, max_papers
    status          : queued | running | paused | failed | completed
    progress        : 0–100
    current_agent   : display string
    logs            : list of {timestamp, level, message}
    papers_found    : lightweight list for the UI (title/year/paperId)
    error           : error string when failed

    Checkpoint keys (for resume)
    ─────────────────────────────
    checkpoint      : 0–5  (0=not started, 1=search done, …, 5=complete)
    _papers_raw     : full paper dicts after search
    _papers_extracted : after pdf_agent
    _papers_summarized: after summarization
    _analysis       : after comparison

    Result keys
    ───────────
    result (markdown), latex, apa
    """

    def __init__(self):
        self._jobs: Dict[str, Dict] = {}
        self._lock  = threading.Lock()

    # ── Job lifecycle ─────────────────────────────────────────────────────────

    def create_job(self, job_id: str, topic: str, max_papers: int) -> Dict:
        with self._lock:
            job: Dict[str, Any] = {
                # Identity
                "job_id":     job_id,
                "topic":      topic,
                "max_papers": max_papers,
                # Status
                "status":        "queued",
                "progress":      0,
                "current_agent": "",
                "error":         None,
                # UI data
                "logs":         [],
                "papers_found": [],
                # Results
                "result": None,
                "latex":  None,
                "apa":    None,
                # Recovery checkpoints
                "checkpoint":          0,
                "_papers_raw":         None,
                "_papers_extracted":   None,
                "_papers_summarized":  None,
                "_analysis":           None,
                # Timestamps
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
            }
            self._jobs[job_id] = job
            return job

    def get_status(self, job_id: str) -> Optional[Dict]:
        with self._lock:
            return self._jobs.get(job_id)

    # ── Mutations (all broadcast to WS clients) ───────────────────────────────

    def update(self, job_id: str, **kwargs) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job.update(kwargs)
            job["updated_at"] = datetime.utcnow().isoformat()

        # Build and broadcast progress event (exclude internal/result keys)
        _fire(job_id, {
            "type":          "progress",
            "progress":      kwargs.get("progress",      job.get("progress", 0)),
            "current_agent": kwargs.get("current_agent", job.get("current_agent", "")),
            "status":        kwargs.get("status",        job.get("status", "running")),
        })

    def add_log(self, job_id: str, message: str, level: str = "info") -> None:
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level":     level,
            "message":   message,
        }
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job["logs"].append(entry)
            job["updated_at"] = datetime.utcnow().isoformat()

        _fire(job_id, {"type": "log", **entry})

    def set_papers(self, job_id: str, papers: list) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job["papers_found"] = papers
            job["updated_at"] = datetime.utcnow().isoformat()

        _fire(job_id, {"type": "papers", "papers": papers})

    def set_result(self, job_id: str, markdown: str, latex: str, apa: str = "") -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job.update({
                "result":   markdown,
                "latex":    latex,
                "apa":      apa,
                "status":   "completed",
                "progress": 100,
                "checkpoint": 5,
                "updated_at": datetime.utcnow().isoformat(),
            })

        _fire(job_id, {"type": "completed"})

    def set_error(self, job_id: str, error: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job.update({
                "status":     "failed",
                "error":      error,
                "updated_at": datetime.utcnow().isoformat(),
            })

        _fire(job_id, {"type": "failed", "error": error})

    # ── Checkpoint helpers (called by planner_agent) ──────────────────────────

    def save_checkpoint(self, job_id: str, stage: int, data_key: str, data: Any) -> None:
        """
        Save pipeline intermediate data so a failed job can be resumed.
        stage: 1=search, 2=pdf, 3=summarize, 4=analysis
        data_key: one of _papers_raw, _papers_extracted, _papers_summarized, _analysis
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job["checkpoint"] = stage
            job[data_key]     = data
            job["updated_at"] = datetime.utcnow().isoformat()

    def get_checkpoint(self, job_id: str) -> int:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.get("checkpoint", 0) if job else 0

    def get_checkpoint_data(self, job_id: str, data_key: str) -> Any:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.get(data_key) if job else None

    def reset_for_retry(self, job_id: str) -> bool:
        """
        Mark a failed job as ready to resume.
        Returns False if job doesn't exist or isn't in a failed state.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job["status"] not in ("failed",):
                return False
            job.update({
                "status":        "queued",
                "error":         None,
                "current_agent": "",
                "updated_at":    datetime.utcnow().isoformat(),
            })
            return True


state_manager = StateManager()
