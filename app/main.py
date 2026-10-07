import os
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException


from .engine import Embedder, inspect_weights
from .pipeline import detect_and_align, to_tensor, normalize, search, load_gallery, NoFaceDetected

WEIGHTS_PATH = os.environ.get("FR_WEIGHTS_PATH", "models/Glint360K_R100_TopoFR_9760.pt")
GALLERY_PATH = os.environ.get("FR_GALLERY_PATH", "gallery/b8730886ae__640ba9de6e__n130.npz")
VERIF_THRESHOLD = float(os.environ.get("FR_VERIF_THRESHOLD", 0.225))

app = FastAPI(title="Who is it? Face recognition API", version='1.0.0')

embedder = None
gallery = None

@app.on_event("startup")
def load_model():
    global embedder, gallery
    embedder = Embedder(WEIGHTS_PATH, device='cuda').warmup()
    gallery = load_gallery(GALLERY_PATH)
    if gallery is None:
        raise RuntimeError(f"Gallery not found or invalid at {GALLERY_PATH}")
    
@app.get("/health")
def health():
    return {"status":"ok", "weights_id": embedder.id if embedder else None}

@app.post("/identify")
async def identify(file: UploadFile = File(...)):
    raw = await file.read()
    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, "Could not decode image")
    
    try:
        aligned, bbox, kps = detect_and_align(bgr)
    except NoFaceDetected:
        return {"match": None, "reason": "no_face_detected"}
    
    tensor = to_tensor(aligned)
    emb = normalize(embedder.embed(tensor))
    
    _, identify_scores, names, _ = search(gallery, emb)
    best_idx = int(np.argmax(identify_scores))
    best_score = float(identify_scores[best_idx])
    
    return {
        "match": names[best_idx] if best_score >= VERIF_THRESHOLD else None,
        "similarity": best_score,
        "threshold": VERIF_THRESHOLD,
        "bbox": bbox.tolist()
    }