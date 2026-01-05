# utils/tracking_utils.py
from __future__ import annotations
from typing import Any, Dict

def get_tracker(config: Dict[str, Any]):
    """
    Returns a wandb-like tracker (Trackio) or None if disabled.
    """
    tcfg = (config or {}).get("tracking", {}) or {}
    if not tcfg.get("enabled", False):
        return None

    import trackio as wandb  # Trackio wandb-like API

    wandb.init(
        project=tcfg.get("project", "gut-microbiome"),
        name=tcfg.get("run_name"),
        config=config,
        space_id=tcfg.get("space_id"),
    )
    return wandb


def safe_log(tracker, data: Dict[str, Any]):
    if tracker is None:
        return
    cleaned = {}
    for k, v in data.items():
        try:
            cleaned[k] = v
        except Exception:
            cleaned[k] = str(v)
    tracker.log(cleaned)