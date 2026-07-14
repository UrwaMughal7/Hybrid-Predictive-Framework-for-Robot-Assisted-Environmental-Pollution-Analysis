"""
utils/helpers.py
================
Shared utility functions used across the entire pipeline:
  - seed locking
  - directory creation
  - figure saving
  - metric computation
  - animation helpers
"""

import os
import random
import shutil
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf

from config.settings import ALL_DIRS, DIR_FIG, SEED


# ── Reproducibility ────────────────────────────────────────────────────────────
def set_all_seeds(seed: int = SEED) -> None:
    """Lock numpy, Python random, and TensorFlow random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


# ── Directory management ───────────────────────────────────────────────────────
def ensure_dirs() -> None:
    """Create every output directory before any save is attempted."""
    for d in ALL_DIRS:
        os.makedirs(d, exist_ok=True)
    print("  ✓ Output folders ready")


# ── Figure saving ──────────────────────────────────────────────────────────────
def savefig(fig, filename: str, folder: str = None) -> None:
    """Save a matplotlib figure to *folder* (default DIR_FIG) and close it."""
    folder = folder or DIR_FIG
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 {path}")


# ── Metrics ────────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, label: str = "") -> dict:
    """Return dict with MAE, MSE, RMSE, R2."""
    y_true = np.asarray(y_true, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()
    mae    = mean_absolute_error(y_true, y_pred)
    mse    = mean_squared_error(y_true, y_pred)
    rmse   = np.sqrt(mse)
    r2     = r2_score(y_true, y_pred)
    m      = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}
    if label:
        print(f"\n  ┌─ [{label}] " + "─" * 38)
        for k, v in m.items():
            arrow = "↓ lower=better" if k != "R2" else "↑ higher=better"
            print(f"  │  {k:<6}: {v:>10.5f}   {arrow}")
        print(f"  └" + "─" * 50)
    return m


# ── Animation helpers ──────────────────────────────────────────────────────────
def detect_animation_writer():
    """Return (writer_name, extension) for animations."""
    if shutil.which("ffmpeg"):
        print("  ✓ ffmpeg detected — videos will be MP4")
        return "ffmpeg", ".mp4"
    print("  ⚠ ffmpeg not found — videos will be GIF (install ffmpeg for MP4)")
    return "pillow", ".gif"


def save_animation(anim_obj, path_no_ext: str, fps: int = 6, dpi: int = 90):
    """Save animation as MP4 (ffmpeg) or GIF (pillow fallback)."""
    for writer_name, ext in [("ffmpeg", ".mp4"), ("pillow", ".gif")]:
        path = path_no_ext + ext
        try:
            if writer_name == "ffmpeg" and not shutil.which("ffmpeg"):
                continue
            writer = (
                animation.FFMpegWriter(fps=fps, bitrate=1800)
                if writer_name == "ffmpeg"
                else animation.PillowWriter(fps=fps)
            )
            anim_obj.save(path, writer=writer, dpi=dpi)
            print(f"  🎬 {path}")
            return path
        except Exception as e:
            print(f"  ⚠ {writer_name} failed: {e}")
    print("  ⚠ Animation save failed — no writer available")
    return None


def make_frames(N: int, max_frames: int = 120):
    """Return (frame_indices, step) for animations."""
    step = max(1, N // max_frames)
    return list(range(0, N, step)), step
