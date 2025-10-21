import os, io, base64, uuid, requests
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage import img_as_float
import cv2

CLOUD = os.getenv("CLOUDINARY_CLOUD", "")
PRESET = os.getenv("CLOUDINARY_PRESET", "")

app = FastAPI(title="Skinmo IA", version="0.1")

class AnalyzeIn(BaseModel):
    image_a_url: str
    image_b_url: str
    pathology: str  # vitiligo|psoriasis|eczema|acne|cicatrice

def _download(url):
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot download image: {e}")

def _pil2cv(im):
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

def _cv2pil(m):
    return Image.fromarray(cv2.cvtColor(m, cv2.COLOR_BGR2RGB))

def _resize_pair(a, b, max_side=640):
    def scale(x):
        h, w = x.shape[:2]
        s = max(h, w)
        if s <= max_side:
            return x
        r = max_side / s
        return cv2.resize(x, (int(w * r), int(h * r)), interpolation=cv2.INTER_AREA)
    a2, b2 = scale(a), scale(b)
    h = min(a2.shape[0], b2.shape[0])
    w = min(a2.shape[1], b2.shape[1])
    return a2[:h, :w], b2[:h, :w]

def _align(a_gray, b_gray):
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        _, warp = cv2.findTransformECC(a_gray, b_gray, warp, cv2.MOTION_EUCLIDEAN,
                                       (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-6))
        return cv2.warpAffine(b_gray, warp, (a_gray.shape[1], a_gray.shape[0]),
                              flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    except cv2.error:
        return b_gray

def _heatmap(diff01):
    diff8 = (diff01 * 255).astype("uint8")
    return cv2.applyColorMap(diff8, cv2.COLORMAP_JET)

def _upload_heat(pil):
    if not CLOUD or not PRESET:
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    url = f"https://api.cloudinary.com/v1_1/{CLOUD}/image/upload"
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    payload = {
        "file": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
        "upload_preset": PRESET,
        "public_id": f"heatmaps/{uuid.uuid4().hex}"
    }
    r = requests.post(url, data=payload, timeout=30)
    r.raise_for_status()
    return r.json()["secure_url"]

@app.get("/health")
def health():
    return {"status": "ok"}

class Out(BaseModel):
    delta_percent: float
    confidence: float
    status_flag: str
    heatmap_url: str

@app.post("/analyze", response_model=Out)
def analyze(inp: AnalyzeIn):
    A = _pil2cv(_download(inp.image_a_url))
    B = _pil2cv(_download(inp.image_b_url))
    A, B = _resize_pair(A, B)
    a = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(B, cv2.COLOR_BGR2GRAY)
    b_al = _align(a, b)

    a_f, b_f = img_as_float(a), img_as_float(b_al)
    s, diff = ssim(a_f, b_f, full=True)
    diff = 1 - diff
    mae = float(np.mean(np.abs(a_f - b_f)))
    delta = max(0.0, min(100.0, (1.0 - mae) * 100.0))
    conf = float(max(0.0, min(1.0, s)))
    status = "green" if s >= 0.75 else ("orange" if s >= 0.55 else "red")

    overlay = cv2.addWeighted(cv2.cvtColor(b_al, cv2.COLOR_GRAY2BGR), 0.6, _heatmap(diff), 0.4, 0)
    heat_url = _upload_heat(_cv2pil(overlay))
    return Out(delta_percent=round(delta, 1),
               confidence=round(conf, 3),
               status_flag=status,
               heatmap_url=heat_url)
