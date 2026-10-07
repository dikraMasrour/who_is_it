"""Device detection and the PyTorch inference engine.

One framework - PyTorch - for both CPU and GPU, and the `.pt` checkpoint is loaded
directly with no conversion of any kind. Two consequences, both deliberate:

  * the numbers measure the model *as distributed*, not a re-exported copy of it;
  * CPU and GPU differ only by the device, never by the runtime, so the speed-up
    ratio means what it says.

GPU support therefore follows PyTorch's own support matrix:

    NVIDIA, CUDA          full support - the reference configuration
    AMD on Linux, ROCm    works, exposed through the same torch.cuda API
    Intel Arc, XPU        works with an XPU-enabled torch build; integrated Intel
                          GPUs are usually not supported
    AMD on Windows        no PyTorch GPU backend exists - CPU only

Run `python check_devices.py` to see what the current machine offers.
"""
import hashlib
import os
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

BACKBONES = ("r100", "r50", "r200")
_LAYER_CFG = {"r50": [3, 4, 14, 3], "r100": [3, 13, 30, 3], "r200": [6, 26, 60, 6]}


# --------------------------------------------------------------------- devices

def gpu_backend():
    """'cuda', 'xpu' or None - which torch GPU backend this machine can use."""
    try:
        import torch
    except ImportError:
        return None
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    xpu = getattr(torch, "xpu", None)          # torch >= 2.5
    try:
        if xpu is not None and xpu.is_available():
            return "xpu"
    except Exception:
        pass
    return None


def available_devices():
    """Logical devices the app can offer: always cpu, plus gpu when torch has one."""
    return ["cpu", "gpu"] if gpu_backend() else ["cpu"]


def gpu_name():
    """Adapter name for display, or None when there is no torch GPU."""
    import torch
    backend = gpu_backend()
    try:
        if backend == "cuda":
            return torch.cuda.get_device_name(0)
        if backend == "xpu":
            return torch.xpu.get_device_name(0)
    except Exception:
        pass
    return backend.upper() if backend else None


def device_label(device):
    if device != "gpu":
        return "CPU"
    backend = gpu_backend()
    if not backend:
        return "GPU (unavailable)"
    api = {"cuda": "CUDA", "xpu": "XPU"}[backend]
    return f"GPU - {gpu_name()} ({api})"


def torch_device(device):
    """Logical device -> torch.device."""
    import torch
    if device == "gpu":
        backend = gpu_backend()
        if not backend:
            raise RuntimeError(
                "No PyTorch GPU on this machine. PyTorch reaches a GPU through CUDA "
                "(NVIDIA, or AMD on Linux via ROCm) or XPU (Intel Arc); none is "
                "available here. Run `python check_devices.py` for details."
            )
        return torch.device(f"{backend}:0")
    return torch.device("cpu")


def make_sync(device):
    """Return a callable that blocks until queued GPU work has finished.

    Essential for timing: CUDA and XPU kernels are launched asynchronously, so a
    timer stopped without a sync measures the launch, not the computation, and
    reports an absurdly fast GPU. On CPU this is a no-op.
    """
    if device != "gpu":
        return lambda: None
    import torch
    backend = gpu_backend()
    if backend == "cuda":
        return torch.cuda.synchronize
    if backend == "xpu":
        return torch.xpu.synchronize
    return lambda: None


def describe_environment():
    """Everything check_devices.py and the sidebar want to report."""
    info = {"torch": None, "cuda_build": None, "backend": gpu_backend(),
            "gpu_name": None, "devices": ["cpu"], "notes": []}
    try:
        import torch
    except ImportError:
        info["notes"].append("PyTorch is not installed - run "
                             "`pip install -r requirements.txt`.")
        return info

    info["torch"] = torch.__version__
    info["cuda_build"] = getattr(torch.version, "cuda", None)
    info["devices"] = available_devices()
    if info["backend"]:
        info["gpu_name"] = gpu_name()
        return info

    if info["cuda_build"] is None:
        info["notes"].append(
            "This is a CPU-only PyTorch build (torch.version.cuda is None). On an "
            "NVIDIA machine, reinstall from the CUDA index - see README.")
    else:
        info["notes"].append(
            f"PyTorch was built for CUDA {info['cuda_build']} but no GPU is visible. "
            "Check the NVIDIA driver with `nvidia-smi`.")
    if os.name == "nt":
        info["notes"].append(
            "On Windows, AMD GPUs have no PyTorch backend at all - CPU only.")
    return info


# ------------------------------------------------------------------ checkpoint

def weights_fingerprint(path):
    """Short stable id for a checkpoint: size plus its first and last megabyte.

    Hashing a full 1 GB file on every rerun would be visible in the UI; this is
    enough to tell two checkpoints apart and to detect one being replaced.
    """
    size = os.path.getsize(path)
    h = hashlib.sha1(str(size).encode())
    with open(path, "rb") as fh:
        h.update(fh.read(1 << 20))
        if size > (2 << 20):
            fh.seek(-(1 << 20), os.SEEK_END)
            h.update(fh.read(1 << 20))
    return h.hexdigest()[:10]


def load_state_dict(path):
    """Read a TopoFR checkpoint into a plain {name: tensor} dict."""
    import torch
    if not os.path.exists(path):
        raise FileNotFoundError(f"weights not found: {path}")
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        # Older checkpoints may be pickled with non-tensor objects.
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "state_dict" in ckpt and "weight" not in ckpt:
        ckpt = ckpt["state_dict"]
    if not isinstance(ckpt, dict):
        raise ValueError(
            f"{os.path.basename(path)} does not contain a state dict "
            f"(got {type(ckpt).__name__}).")
    # Checkpoints saved under DataParallel carry a 'module.' prefix.
    if any(k.startswith("module.") for k in ckpt):
        ckpt = {k[len("module."):] if k.startswith("module.") else k: v
                for k, v in ckpt.items()}
    return ckpt


def infer_backbone(state_dict):
    """Recover r50 / r100 / r200 by counting the blocks in each stage."""
    counts = []
    for stage in range(1, 5):
        prefix = f"layer{stage}."
        idxs = {int(k[len(prefix):].split(".")[0])
                for k in state_dict if k.startswith(prefix)}
        counts.append(max(idxs) + 1 if idxs else 0)
    for name, cfg in _LAYER_CFG.items():
        if cfg == counts:
            return name
    return None


def inspect_weights(path):
    """Cheap metadata about a checkpoint, for the setup screen."""
    sd = load_state_dict(path)
    head = sd.get("weight")
    return {
        "path": path,
        "name": os.path.basename(path),
        "id": weights_fingerprint(path),
        "backbone": infer_backbone(sd),
        "train_identities": int(head.shape[0]) if head is not None else None,
        "embedding_dim": int(head.shape[1]) if head is not None else 512,
        "size_mb": os.path.getsize(path) / 1e6,
    }


# ---------------------------------------------------------------------- engine

class Embedder:
    """TopoFR on one device. `embed` is the batch-1 call the UI times."""

    def __init__(self, weights_path, device="cpu", backbone=None):
        import torch
        from .iresnet import build_model

        self.weights_path = weights_path
        self.device = device
        self.torch_device = torch_device(device)
        self.sync = make_sync(device)

        sd = load_state_dict(weights_path)
        self.backbone = backbone or infer_backbone(sd)
        if self.backbone not in _LAYER_CFG:
            raise ValueError(
                f"Could not tell which backbone {os.path.basename(weights_path)} is. "
                f"Pick one explicitly - expected one of {', '.join(BACKBONES)}.")

        head = sd.get("weight")
        net = build_model(self.backbone, num_classes=int(head.shape[0]) if head is not None else 1)
        missing, unexpected = net.load_state_dict(sd, strict=False)
        blocking = [k for k in missing if k != "weight"]
        if blocking or unexpected:
            raise ValueError(
                f"{os.path.basename(weights_path)} does not match backbone "
                f"'{self.backbone}': {len(blocking)} missing / {len(unexpected)} "
                f"unexpected tensors. Try a different backbone.")

        # The classifier head is training-only; dropping it saves ~150 MB of VRAM.
        net.weight = torch.nn.Parameter(torch.empty(0))
        self.net = net.eval().to(self.torch_device)
        for p in self.net.parameters():
            p.requires_grad_(False)

        self.id = weights_fingerprint(weights_path)
        self.label = device_label(device)

    # -- inference -----------------------------------------------------------

    def embed(self, arr):
        """One (3, 112, 112) float32 array -> (1, 512) embedding."""
        return self.embed_batch([arr])

    def embed_batch(self, arrs):
        import torch
        x = torch.from_numpy(np.ascontiguousarray(np.stack(arrs)))
        x = x.to(self.torch_device, non_blocking=False)
        with torch.inference_mode():
            out = self.net(x, phase="infer")
        return out.float().cpu().numpy()

    def warmup(self, runs=5):
        """Discard the first inferences.

        The first CUDA call initialises the context and picks convolution
        algorithms; unwarmed, it reports several hundred ms instead of a few.
        """
        dummy = np.zeros((3, 112, 112), dtype=np.float32)
        for _ in range(runs):
            self.embed(dummy)
        self.sync()
        return self

    def free(self):
        """Release VRAM when the app switches device or model."""
        import torch
        self.net = None
        if self.device == "gpu":
            try:
                if gpu_backend() == "cuda":
                    torch.cuda.empty_cache()
                elif gpu_backend() == "xpu":
                    torch.xpu.empty_cache()
            except Exception:
                pass
