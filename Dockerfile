FROM python:3.12.2-slim

# system dependencies opencv and onnxruntime need
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY models/ models/
COPY gallery/ gallery/

ENV FR_WEIGHTS_PATH=models/Glint360K_R100_TopoFR_9760.pt
ENV FR_GALLERY_PATH=gallery/b8730886ae__640ba9de6e__n130.npz
ENV FR_VERIF_THRESHOLD=0.225

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
