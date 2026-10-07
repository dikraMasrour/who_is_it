"""Paths and persisted user settings.

Nothing here is tied to one machine. Every path is derived from this file's own
location, and each one can be overridden with an environment variable, so the
folder can be copied or checked out anywhere and still work.

    FR_MODELS_DIR   where to look for .pt checkpoints   (default ../models)
    FR_DATA_DIR     where to look for datasets          (default ../data)
    FR_ASSETS_DIR   where galleries are written         (default ./assets)
"""
import json
import os

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(APP_DIR)


def _dir(env_var, *default_parts):
    override = os.environ.get(env_var)
    return os.path.abspath(override if override else os.path.join(*default_parts))


MODELS_DIR = _dir("FR_MODELS_DIR", PROJECT_DIR, "models")
DATA_DIR = _dir("FR_DATA_DIR", PROJECT_DIR, "data")
ASSETS_DIR = _dir("FR_ASSETS_DIR", APP_DIR, "assets")

GALLERY_DIR = os.path.join(ASSETS_DIR, "galleries")
EXTRACT_DIR = os.path.join(ASSETS_DIR, "datasets")
SETTINGS_PATH = os.path.join(APP_DIR, "user_settings.json")

# Calibrated operating point for the Glint360K TopoFR-R100 checkpoint (10-fold,
# 14,000 identities). Every model has its own threshold - this is a starting
# value, not a constant, and must be recalibrated for a different checkpoint.
VERIF_THRESHOLD = 0.225

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
WEIGHT_EXTS = (".pt", ".pth", ".ckpt")


# ------------------------------------------------------------------- settings

def load_settings():
    """Last-used weights / dataset / gallery. Missing or corrupt file -> {}."""
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_settings(settings):
    os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
    with open(SETTINGS_PATH, "w", encoding="utf-8") as fh:
        json.dump(settings, fh, indent=2)


# -------------------------------------------------------------------- lookups

def find_weights():
    """Every checkpoint under MODELS_DIR, newest first. Empty list is fine -
    the UI always also accepts a typed path."""
    found = []
    for root, _dirs, files in os.walk(MODELS_DIR):
        for f in files:
            if f.lower().endswith(WEIGHT_EXTS):
                found.append(os.path.join(root, f))
    return sorted(found, key=lambda p: -os.path.getmtime(p))


def find_datasets():
    """Immediate sub-directories of DATA_DIR and of the extracted-zip folder."""
    out = []
    for base in (DATA_DIR, EXTRACT_DIR):
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            path = os.path.join(base, name)
            if os.path.isdir(path):
                out.append(path)
    return out


def gallery_path(weights_id, dataset_id, n_enrolled):
    """One file per (weights, dataset, size) combination, so several can coexist."""
    os.makedirs(GALLERY_DIR, exist_ok=True)
    return os.path.join(GALLERY_DIR, f"{weights_id}__{dataset_id}__n{n_enrolled}.npz")
