"""Preprocessing, scoring, timing and gallery I/O - everything that is the same
whatever device the model runs on.

Detection + alignment now happen here too: `detect_and_align` runs InsightFace's
RetinaFace detector to get a 5-point landmark, then affine-warps the face to the
112x112 ArcFace template that TopoFR was trained against. A raw, un-aligned photo
can go straight into `detect_and_align`; its 112x112 output is what `to_tensor`
still expects. Callers that already hand in a pre-aligned crop can skip straight
to `to_tensor` as before - detection is an added first step, not a replacement.
"""
import json
import os
from time import perf_counter

import cv2
import numpy as np

GALLERY_FORMAT = 3      # multiple reference templates per identity


# --------------------------------------------------------- detection & alignment

_detectors = {}  # {(ctx_id, det_size): insightface.app.FaceAnalysis}, detection module only


class NoFaceDetected(RuntimeError):
    """No face scored above `det_thresh`. Never fall back to a guessed alignment -
    a wrong warp silently corrupts the embedding, a raised error doesn't."""


def get_detector(det_size=(640, 640), ctx_id=0):
    """Lazily load and cache the InsightFace RetinaFace detector.

    `allowed_modules=["detection"]` loads only the detection model out of the
    buffalo_l pack (skips recognition/landmark-106/genderage), so repeated calls
    are cheap after the first. Cached per `ctx_id` (0 = GPU via onnxruntime-gpu,
    -1 = CPU) so the app's CPU/GPU toggle applies to detection too, the same way
    it does to the embedder - insightface itself falls back to CPU when the GPU
    provider isn't available.
    """
    key = (ctx_id, det_size)
    if key not in _detectors:
        from insightface.app import FaceAnalysis
        det = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"])
        det.prepare(ctx_id=ctx_id, det_size=det_size)
        _detectors[key] = det
    return _detectors[key]


def detect_and_align(bgr, image_size=112, det_thresh=0.5, ctx_id=0):
    """Detect the largest face and affine-align it to a 112x112 ArcFace crop.

    RetinaFace gives a 5-point landmark per face (eyes, nose, mouth corners).
    When several faces are found, the largest bounding box wins - the usual
    assumption for a benchmark/demo probe, where the subject fills the frame.
    `face_align.norm_crop` then solves the similarity transform onto
    InsightFace's standard 112x112 template, the same one ArcFace-family models
    (TopoFR included) were trained against, and warps the face into it.

    Returns (aligned_bgr, bbox, kps). Raises NoFaceDetected if nothing clears
    `det_thresh` - callers should catch this and surface it rather than pass a
    black frame or an unaligned crop into `to_tensor`.
    """
    from insightface.utils import face_align

    det = get_detector(ctx_id=ctx_id)
    det.det_thresh = det_thresh
    faces = det.get(bgr)
    if not faces:
        raise NoFaceDetected("no face found above the detection threshold")

    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    aligned = face_align.norm_crop(bgr, landmark=face.kps, image_size=image_size)
    return aligned, face.bbox, face.kps


def detect_and_align_all(bgr, image_size=112, det_thresh=0.5, ctx_id=0):
    """Detect and align every face, ordered from largest to smallest."""
    from insightface.utils import face_align

    det = get_detector(ctx_id=ctx_id)
    det.det_thresh = det_thresh
    faces = det.get(bgr)
    if not faces:
        raise NoFaceDetected("no face found above the detection threshold")

    faces = sorted(
        faces,
        key=lambda face: (face.bbox[2] - face.bbox[0]) *
                         (face.bbox[3] - face.bbox[1]),
        reverse=True,
    )
    return [
        (face_align.norm_crop(bgr, landmark=face.kps, image_size=image_size),
         face.bbox, face.kps)
        for face in faces
    ]


# --------------------------------------------------------------- preprocessing

def to_tensor(bgr):
    """Benchmark preprocessing: resize 112 -> RGB -> CHW -> [-1, 1]."""
    img = cv2.resize(bgr, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)
    return (img / 255.0 - 0.5) / 0.5


def normalize(a):
    """L2-normalise rows of an (n, d) array."""
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def cosine(a, b):
    """Cosine similarity between two already-normalised (1, d) embeddings."""
    return float((a * b).sum())


# ---------------------------------------------------------------------- timing

def timed(fn, sync=None):
    """Run fn, return (result, elapsed_ms).

    `sync` must be the active device's synchronise callable. GPU kernels are
    queued asynchronously, so without it the timer measures the launch rather
    than the work and the GPU looks impossibly fast. Syncing *before* starting
    the clock also keeps work queued by a previous step out of this measurement.
    """
    if sync is not None:
        sync()
    t0 = perf_counter()
    out = fn()
    if sync is not None:
        sync()
    return out, (perf_counter() - t0) * 1000.0


# --------------------------------------------------------------------- gallery

def save_gallery(path, emb, names, rel_paths, meta):
    """Store embeddings plus enough metadata to know what they are.

    Paths are stored *relative to the dataset root*, and the root is stored
    separately. That is what makes a gallery survive being copied to another
    machine - absolute paths would not.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    meta = dict(meta, format=GALLERY_FORMAT)
    np.savez_compressed(
        path,
        emb=np.asarray(emb, dtype=np.float32),
        names=np.array([str(n) for n in names]),
        rel_paths=np.array([str(p) for p in rel_paths]),
        meta=np.array(json.dumps(meta)),
    )
    return path


def load_gallery(path):
    """Pre-computed, pre-normalised gallery. Returns None when absent.

    Both the embedding and the L2 normalisation happen at enrollment time, so a
    search costs only: embed the probe + one matrix product. No database work is
    charged to the query, which is what keeps the 1:N figure comparable to 1:1.
    """
    if not path or not os.path.exists(path):
        return None
    data = np.load(path, allow_pickle=False)
    try:
        meta = json.loads(str(data["meta"]))
    except (KeyError, ValueError):
        meta = {}
    if meta.get("format") != GALLERY_FORMAT:
        return None
    return {
        "path": path,
        "emb": data["emb"].astype(np.float32),      # (templates, 512), normalised
        "names": [str(n) for n in data["names"]],
        "rel_paths": [str(p) for p in data["rel_paths"]],
        "root": meta.get("root", ""),
        "meta": meta,
    }


def image_path(gallery, index):
    """Absolute path of an enrolled image, resolved against the stored root."""
    rel = gallery["rel_paths"][index]
    if os.path.isabs(rel):
        return rel
    return os.path.normpath(os.path.join(gallery["root"], rel))


def read_gallery_image(gallery, index):
    """Enrolled image as BGR, or None if the dataset has moved or been deleted.

    Never raises: a gallery is still perfectly usable for scoring when its
    thumbnails are gone, so a missing file must not take the page down.
    """
    try:
        path = image_path(gallery, index)
    except (IndexError, KeyError):
        return None
    if not path or not os.path.exists(path):
        return None
    return cv2.imread(path)


def gallery_matches(gallery, weights_id):
    """True when this gallery was enrolled with the currently selected weights.

    Embeddings from two different models are not comparable, so searching a
    gallery built by another checkpoint silently returns nonsense.
    """
    return bool(gallery) and gallery["meta"].get("weights_id") == weights_id


def search(gallery_emb, probe_emb):
    """Search templates and return one best-scoring row per identity.

    Returns template scores, identity-level scores, identity names, and the
    template row that explains each identity-level result.
    """
    template_scores = (probe_emb @ gallery_emb["emb"].T).ravel()
    names = np.asarray(gallery_emb["names"])
    unique_names, inverse = np.unique(names, return_inverse=True)
    identity_scores = np.full(len(unique_names), -np.inf, dtype=np.float32)
    best_templates = np.full(len(unique_names), -1, dtype=np.int64)
    for row, identity_index in enumerate(inverse):
        if template_scores[row] > identity_scores[identity_index]:
            identity_scores[identity_index] = template_scores[row]
            best_templates[identity_index] = row
    return template_scores, identity_scores, unique_names, best_templates
