"""Active-learning resume: save / load / infer interrupted run state."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

STATE_NAME = "al_resume_state.json"
STATE_VERSION = 1


def _state_path(output_dir: str) -> Path:
    return Path(output_dir) / "training_logs" / STATE_NAME


def _enabled(cfg) -> bool:
    try:
        return bool(getattr(getattr(cfg.TRAINER, "COOPAL", object()), "AL_RESUME_ENABLE", True))
    except Exception:
        return True


def _infer_from_query_log(output_dir: Path) -> Optional[Tuple[int, List[int]]]:
    """Infer labeled pool from conference SaE query_rounds.jsonl."""
    log_path = output_dir / "training_logs" / "query_rounds.jsonl"
    if not log_path.is_file():
        return None
    last_rec: Optional[dict] = None
    max_round = -1
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        r = int(rec.get("round", -1))
        if r >= max_round:
            max_round = r
            last_rec = rec
    if last_rec is None:
        return None
    labeled = last_rec.get("labeled_global_idx_after_query")
    if labeled is None:
        return None
    return max_round, [int(x) for x in labeled]


def infer_resume_state(
    output_dir: str,
    cfg,
    total_n: int,
    n_query: int,
    total_rounds: int,
) -> Optional[Dict[str, Any]]:
    """Rebuild resume state from legacy artifacts when json missing."""
    out = Path(output_dir)
    summary = out / "training_logs" / "run_summary.json"
    if summary.is_file():
        return None

    explicit = load_resume_state(output_dir)
    if explicit is not None:
        return explicit

    inferred = _infer_from_query_log(out)
    if inferred is None:
        return None
    completed, labeled = inferred
    u_index = [i for i in range(total_n) if i not in set(labeled)]
    seed = int(getattr(cfg, "SEED", 1))
    return {
        "version": STATE_VERSION,
        "source": "inferred",
        "dataset_name": str(getattr(cfg.DATASET, "NAME", "")),
        "seed": seed,
        "total_train_n": int(total_n),
        "n_query": int(n_query),
        "total_rounds": int(total_rounds),
        "completed_round": int(completed),
        "next_round": int(completed) + 1,
        "resume_epoch": 0,
        "labeled_global_idx": labeled,
        "u_index": u_index,
    }


def load_resume_state(output_dir: str) -> Optional[Dict[str, Any]]:
    path = _state_path(output_dir)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if int(data.get("version", 0)) != STATE_VERSION:
        return None
    return data


def validate_resume_state(
    state: Dict[str, Any],
    cfg,
    total_n: int,
    n_query: int,
    total_rounds: int,
) -> bool:
    if int(state.get("total_train_n", -1)) != int(total_n):
        return False
    if int(state.get("n_query", -1)) != int(n_query):
        return False
    if int(state.get("total_rounds", -1)) != int(total_rounds):
        return False
    ds = str(getattr(cfg.DATASET, "NAME", ""))
    if state.get("dataset_name") and str(state["dataset_name"]) != ds:
        return False
    if int(state.get("seed", getattr(cfg, "SEED", -1))) != int(getattr(cfg, "SEED", -1)):
        return False
    labeled = state.get("labeled_global_idx") or []
    u_index = state.get("u_index") or []
    if len(labeled) + len(u_index) != int(total_n):
        return False
    return True


def get_resume_plan(
    output_dir: str,
    cfg,
    total_n: int,
    n_query: int,
    total_rounds: int,
) -> Optional[Dict[str, Any]]:
    if not _enabled(cfg):
        return None
    if (Path(output_dir) / "training_logs" / "run_summary.json").is_file():
        return None

    state = infer_resume_state(output_dir, cfg, total_n, n_query, total_rounds)
    if state is None:
        return None
    if not validate_resume_state(state, cfg, total_n, n_query, total_rounds):
        print("[al_resume] state mismatch with current config — starting fresh")
        return None
    return state


def save_resume_state(
    output_dir: str,
    *,
    cfg,
    total_n: int,
    n_query: int,
    total_rounds: int,
    labeled_global_idx: List[int],
    u_index: List[int],
    next_round: int,
    resume_epoch: int = 0,
    completed_round: Optional[int] = None,
    note: str = "",
) -> Path:
    log_dir = Path(output_dir) / "training_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    if completed_round is None:
        completed_round = next_round - 1 if resume_epoch == 0 else next_round
    payload = {
        "version": STATE_VERSION,
        "source": "checkpoint",
        "note": note,
        "dataset_name": str(getattr(cfg.DATASET, "NAME", "")),
        "seed": int(getattr(cfg, "SEED", 1)),
        "total_train_n": int(total_n),
        "n_query": int(n_query),
        "total_rounds": int(total_rounds),
        "completed_round": int(completed_round),
        "next_round": int(next_round),
        "resume_epoch": int(resume_epoch),
        "labeled_global_idx": [int(x) for x in labeled_global_idx],
        "u_index": [int(x) for x in u_index],
    }
    path = _state_path(output_dir)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def clear_resume_state(output_dir: str) -> None:
    path = _state_path(output_dir)
    if path.is_file():
        path.unlink()


def apply_labeled_pool(
    unlabeled_dst,
    labeled_global_idx: List[int],
    u_index: List[int],
) -> Tuple[list, list]:
    train_x = [unlabeled_dst[i] for i in labeled_global_idx]
    return train_x, list(u_index)
