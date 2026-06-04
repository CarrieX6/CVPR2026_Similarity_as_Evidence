"""Prune per-epoch AL checkpoints to avoid filling disk."""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple


def _epoch_from_name(path: Path) -> int:
    m = re.search(r"model\.pth\.tar-(\d+)$", path.name)
    return int(m.group(1)) if m else -1


def list_epoch_checkpoints(model_dir: Path) -> List[Tuple[int, Path]]:
    rows: List[Tuple[int, Path]] = []
    if not model_dir.is_dir():
        return rows
    for p in model_dir.glob("model.pth.tar-*"):
        ep = _epoch_from_name(p)
        if ep >= 0:
            rows.append((ep, p))
    rows.sort(key=lambda x: x[0])
    return rows


def latest_epoch_checkpoint(model_dir: Path) -> Optional[Tuple[int, Path]]:
    """Return (epoch, path) for the most recently written checkpoint."""
    rows = list_epoch_checkpoints(model_dir)
    if not rows:
        return None
    ep, path = max(rows, key=lambda item: item[1].stat().st_mtime)
    return ep, path


def _sync_checkpoint_pointer(model_dir: Path, path: Path) -> None:
    ptr = model_dir / "checkpoint"
    try:
        ptr.write_text(f"{path.name}\n", encoding="utf-8")
    except OSError:
        pass


def prune_epoch_checkpoints(model_dir: Path, keep_last: int = 2, dry_run: bool = False) -> Tuple[int, int]:
    """Delete old model.pth.tar-* keeping newest ``keep_last`` files by mtime.

    Uses modification time (not epoch index) so stale high-epoch files from an
    earlier AL round are removed while mid-round resume checkpoints are kept.

    Returns (deleted_files, freed_bytes).
    """
    keep_last = max(int(keep_last), 0)
    rows = list_epoch_checkpoints(model_dir)
    if keep_last <= 0:
        to_delete = rows
    elif len(rows) <= keep_last:
        to_delete = []
    else:
        rows_by_mtime = sorted(rows, key=lambda item: item[1].stat().st_mtime)
        to_delete = rows_by_mtime[: len(rows) - keep_last]

    deleted = 0
    freed = 0
    for _, path in to_delete:
        try:
            sz = path.stat().st_size
        except OSError:
            sz = 0
        if dry_run:
            deleted += 1
            freed += sz
            continue
        try:
            path.unlink()
            deleted += 1
            freed += sz
        except OSError:
            pass

    if not dry_run and keep_last > 0:
        latest = latest_epoch_checkpoint(model_dir)
        if latest is not None:
            _sync_checkpoint_pointer(model_dir, latest[1])

    return deleted, freed


def prune_output_dir(output_dir: Path, keep_last: int = 2, dry_run: bool = False) -> Tuple[int, int]:
    deleted = 0
    freed = 0
    out = Path(output_dir)
    for sub in ("prompt_learner", "meh_net"):
        d, f = prune_epoch_checkpoints(out / sub, keep_last=keep_last, dry_run=dry_run)
        deleted += d
        freed += f
    return deleted, freed


def purge_all_checkpoints(output_dir: Path, dry_run: bool = False) -> Tuple[int, int]:
    """Remove all per-epoch AL checkpoints after a run completes."""
    return prune_output_dir(output_dir, keep_last=0, dry_run=dry_run)
