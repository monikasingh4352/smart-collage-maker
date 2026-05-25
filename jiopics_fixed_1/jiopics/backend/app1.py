from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import base64, hashlib, io, os, pickle, time, traceback, atexit
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageFilter
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
ROOT         = os.path.dirname(BASE_DIR)
FRONTEND_DIR = os.path.join(ROOT, "frontend")
ASSETS_DIR   = os.path.join(ROOT, "assets", "collage_templates")
CACHE_FILE   = os.path.join(BASE_DIR, ".roi_cache.pkl")

app = Flask(__name__,
            template_folder=os.path.join(FRONTEND_DIR, "templates"),
            static_folder=os.path.join(FRONTEND_DIR, "static"))
CORS(app)

# ── In-memory caches (also persisted to disk) ──────────────────────────────────
_saliency_cache: dict[str, tuple[float, float]] = {}
_quality_cache:  dict[str, float]               = {}
_face_cache:     dict[str, list]                = {}
_sal_map_cache:  dict[str, np.ndarray]          = {}   


# ════════════════════════════════════════════════════════════════════════════════
#  DISK CACHE  — survives server restarts; same photo = instant on reload
# ════════════════════════════════════════════════════════════════════════════════

def _load_caches():
    if not os.path.exists(CACHE_FILE):
        return
    try:
        with open(CACHE_FILE, "rb") as f:
            data = pickle.load(f)
        _saliency_cache.update(data.get("saliency", {}))
        _quality_cache.update(data.get("quality",  {}))
        _face_cache.update(data.get("faces",    {}))
        print(f"[Cache] Loaded — {len(_saliency_cache)} ROIs, "
              f"{len(_quality_cache)} quality scores, "
              f"{len(_face_cache)} face results")
    except Exception as e:
        print(f"[Cache] Load failed (safe to ignore): {e}")


def _save_caches():
    try:
        with open(CACHE_FILE, "wb") as f:
            pickle.dump({
                "saliency": _saliency_cache,
                "quality":  _quality_cache,
                "faces":    _face_cache,
            }, f)
        print(f"[Cache] Saved — {len(_saliency_cache)} entries")
    except Exception as e:
        print(f"[Cache] Save failed: {e}")


_load_caches()
atexit.register(_save_caches)


# ── Static assets ──────────────────────────────────────────────────────────────
@app.route("/assets/collage_templates/<path:filename>")
def serve_asset(filename):
    return send_from_directory(ASSETS_DIR, filename)


# ════════════════════════════════════════════════════════════════════════════════
#  TEMPLATE LOADER
# ════════════════════════════════════════════════════════════════════════════════

def _load_templates() -> dict:
    import json
    json_path = os.path.join(BASE_DIR, "templates.json")
    if not os.path.exists(json_path):
        print(f"[ERROR] templates.json not found at {json_path}")
        return {}
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    templates = {}
    for key, tmpl in raw.items():
        if key.startswith("_"):
            continue
        if "canvas_ratio" in tmpl:
            tmpl["canvas_ratio"] = tuple(tmpl["canvas_ratio"])
        if "bg" in tmpl:
            tmpl["bg"] = tuple(tmpl["bg"])
        if "slots_def" in tmpl:
            tmpl["slots_def"] = [tuple(s) for s in tmpl["slots_def"]]
        templates[key] = tmpl
    print(f"[Templates] Loaded {len(templates)} templates")
    return templates


COLLAGE_TEMPLATES = _load_templates()
POLAROID_ANGLES   = [-8, 6, -4, 7, -3, 5, -6, 4, -5, 8]


# ════════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ════════════════════════════════════════════════════════════════════════════════

def _img_hash(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.resize((32, 32), Image.NEAREST).save(buf, "JPEG", quality=30)
    return hashlib.md5(buf.getvalue()).hexdigest()


def decode_image(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64.split(",")[-1]))).convert("RGB")


# ════════════════════════════════════════════════════════════════════════════════
#  FACE DETECTION  —  MTCNN via facenet_pytorch
# ════════════════════════════════════════════════════════════════════════════════

_blazeface_detector = None
_blazeface_device   = None


def _get_blazeface():
    global _blazeface_detector, _blazeface_device
    if _blazeface_detector is not None:
        return _blazeface_detector
    try:
        import torch
        from facenet_pytorch import MTCNN
        _blazeface_device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _blazeface_detector = MTCNN(
            keep_all=True,
            device=_blazeface_device,
            min_face_size=10,
            thresholds=[0.5, 0.6, 0.6],
            post_process=False,
        )
        print(f"[Face] MTCNN loaded on {_blazeface_device}")
        return _blazeface_detector
    except ImportError:
        print("[Face] facenet_pytorch not available — saliency fallback active")
        return None


def _run_blazeface(pil_img: Image.Image) -> list:
    """
    Returns list of face dicts: [{x1,y1,x2,y2,w,h,cx,cy,conf}, ...]
    Sorted by face area descending. Cached by image hash.
    """
    h = _img_hash(pil_img)
    if h in _face_cache:
        return _face_cache[h]

    orig_w, orig_h = pil_img.size
    boxes_out      = []
    detector       = _get_blazeface()

    if detector is None:
        _face_cache[h] = boxes_out
        return boxes_out

    try:
        import torch
        scale   = min(1.0, 640 / orig_w)
        det_img = pil_img.resize(
            (max(1, int(orig_w * scale)), max(1, int(orig_h * scale))),
            Image.BILINEAR)

        with torch.no_grad():
            raw_boxes, confidences = detector.detect(det_img)

        if raw_boxes is not None:
            inv = 1.0 / scale
            for i, box in enumerate(raw_boxes):
                conf = float(confidences[i]) if confidences is not None else 1.0
                if conf < 0.70:
                    continue
                x1 = int(max(0,      box[0] * inv))
                y1 = int(max(0,      box[1] * inv))
                x2 = int(min(orig_w, box[2] * inv))
                y2 = int(min(orig_h, box[3] * inv))
                boxes_out.append({
                    "x1": float(x1), "y1": float(y1),
                    "x2": float(x2), "y2": float(y2),
                    "conf": conf,
                    "w":  float(x2 - x1),
                    "h":  float(y2 - y1),
                    "cx": float((x1 + x2) / 2 / orig_w),
                    "cy": float((y1 + y2) / 2 / orig_h),
                })

        boxes_out.sort(key=lambda b: b["w"] * b["h"], reverse=True)

    except Exception as e:
        print(f"[Face] Detection error: {e}")

    _face_cache[h] = boxes_out
    return boxes_out


# ════════════════════════════════════════════════════════════════════════════════
#  U²-NET SALIENCY
# ════════════════════════════════════════════════════════════════════════════════

_u2net_model  = None
_u2net_device = None


_U2NET_MAX_EDGE = 400


def _run_u2net(pil_img: Image.Image) -> np.ndarray:
    """Returns H×W float32 saliency map [0-1]. Cached by image hash."""
    global _u2net_model, _u2net_device

    h = _img_hash(pil_img)
    if h in _sal_map_cache:
        return _sal_map_cache[h]

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torchvision import transforms
    except ImportError:
        result = _heuristic_saliency_map(pil_img)
        _sal_map_cache[h] = result
        return result

    # ── Model definition (inline, no external file needed) ───────────────────
    class REBNCONV(nn.Module):
        def __init__(self, i=3, o=3, d=1):
            super().__init__()
            self.c = nn.Conv2d(i, o, 3, padding=d, dilation=d)
            self.b = nn.BatchNorm2d(o)
            self.r = nn.ReLU(inplace=True)
        def forward(self, x): return self.r(self.b(self.c(x)))

    class RSU4F(nn.Module):
        def __init__(self, i, m, o):
            super().__init__()
            self.i_ = REBNCONV(i, o);   self.c1 = REBNCONV(o, m, 1)
            self.c2 = REBNCONV(m, m, 2); self.c3 = REBNCONV(m, m, 4)
            self.c4 = REBNCONV(m, m, 8)
            self.c3d = REBNCONV(m*2, m, 4)
            self.c2d = REBNCONV(m*2, m, 2)
            self.c1d = REBNCONV(m*2, o, 1)
        def forward(self, x):
            hx = self.i_(x)
            h1 = self.c1(hx); h2 = self.c2(h1); h3 = self.c3(h2); h4 = self.c4(h3)
            return (self.c1d(torch.cat((
                self.c2d(torch.cat((
                    self.c3d(torch.cat((h4, h3), 1)), h2), 1)),
                h1), 1)) + hx)

    class U2NETP(nn.Module):
        def __init__(self):
            super().__init__()
            self.s1  = RSU4F(3, 16, 64);  self.p12 = nn.MaxPool2d(2, 2, ceil_mode=True)
            self.s2  = RSU4F(64, 16, 64); self.p23 = nn.MaxPool2d(2, 2, ceil_mode=True)
            self.s3  = RSU4F(64, 16, 64); self.p34 = nn.MaxPool2d(2, 2, ceil_mode=True)
            self.s4  = RSU4F(64, 16, 64)
            self.s3d = RSU4F(128, 16, 64)
            self.s2d = RSU4F(128, 16, 64)
            self.s1d = RSU4F(128, 16, 64)
            self.d1  = nn.Conv2d(64, 1, 3, padding=1)
            self.d2  = nn.Conv2d(64, 1, 3, padding=1)
            self.d3  = nn.Conv2d(64, 1, 3, padding=1)
            self.d4  = nn.Conv2d(64, 1, 3, padding=1)
            self.out = nn.Conv2d(4, 1, 1)

        def forward(self, x):
            ip = lambda t, s: F.interpolate(t, s, mode='bilinear', align_corners=False)
            h1  = self.s1(x)
            h2  = self.s2(self.p12(h1))
            h3  = self.s3(self.p23(h2))
            h4  = self.s4(self.p34(h3))
            h3d = self.s3d(torch.cat((ip(h4,  h3.shape[2:]), h3), 1))
            h2d = self.s2d(torch.cat((ip(h3d, h2.shape[2:]), h2), 1))
            h1d = self.s1d(torch.cat((ip(h2d, h1.shape[2:]), h1), 1))
            d1  = self.d1(h1d)
            d2  = ip(self.d2(h2d), x.shape[2:])
            d3  = ip(self.d3(h3d), x.shape[2:])
            d4  = ip(self.d4(h4),  x.shape[2:])
            return torch.sigmoid(self.out(torch.cat((d1, d2, d3, d4), 1))), torch.sigmoid(d1)

    # ── Load weights once ─────────────────────────────────────────────────────
    wp = os.path.join(BASE_DIR, "u2netp.pth")
    if _u2net_model is None:
        if not os.path.exists(wp):
            import urllib.request
            print("[U²-Net] Downloading weights (~4 MB)…")
            urllib.request.urlretrieve(
                "https://github.com/xuebinqin/U-2-Net/releases/download/v1.0/u2netp.pth", wp)
            print("[U²-Net] Download complete")
        _u2net_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = U2NETP()
        net.load_state_dict(torch.load(wp, map_location=_u2net_device))
        net.to(_u2net_device).eval()
        _u2net_model = net
        print(f"[U²-Net] Model loaded on {_u2net_device}")

    # ── Inference ─────────────────────────────────────────────────────────────
    ow, oh  = pil_img.size
    me      = _U2NET_MAX_EDGE
    if max(ow, oh) > me:
        scale = me / max(ow, oh)
        proc  = pil_img.resize((int(ow * scale), int(oh * scale)), Image.BILINEAR)
    else:
        proc  = pil_img

    tf = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    tensor = tf(proc.convert("RGB")).unsqueeze(0).to(_u2net_device)

    with torch.no_grad():
        sal, _ = _u2net_model(tensor)

    sal    = sal.squeeze().cpu().numpy()
    sal    = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    result = np.array(
        Image.fromarray((sal * 255).astype(np.uint8)).resize((ow, oh), Image.BILINEAR)
    ).astype(np.float32) / 255.0

    _sal_map_cache[h] = result
    return result


# ════════════════════════════════════════════════════════════════════════════════
#  HEURISTIC SALIENCY  — CPU-only fallback, ~5 ms/image
# ════════════════════════════════════════════════════════════════════════════════

def _heuristic_saliency_map(pil_img: Image.Image) -> np.ndarray:
    SZ   = 160
    ow, oh = pil_img.size
    arr  = np.asarray(pil_img.convert("RGB").resize((SZ, SZ), Image.BILINEAR), dtype=np.float32)
    H, W = arr.shape[:2]
    Y, X = np.mgrid[0:H, 0:W]
    gauss = np.exp(-((X - W/2)**2 / (2*(W*.40)**2) + (Y - H/2)**2 / (2*(H*.40)**2)))
    gray  = arr.mean(2)
    edge  = np.hypot(np.abs(np.diff(gray, axis=0, prepend=gray[:1])),
                     np.abs(np.diff(gray, axis=1, prepend=gray[:, :1])))
    edge /= edge.max() + 1e-6
    cmax  = np.maximum(np.maximum(arr[:,:,0], arr[:,:,1]), arr[:,:,2])
    cmin  = np.minimum(np.minimum(arr[:,:,0], arr[:,:,1]), arr[:,:,2])
    sat   = (cmax - cmin) / (cmax + 1e-6) / 255.0
    sal   = gauss*.40 + edge*.35 + sat*.25
    sal  /= sal.max() + 1e-6
    return np.array(
        Image.fromarray((sal * 255).astype(np.uint8)).resize((ow, oh), Image.BILINEAR)
    ).astype(np.float32) / 255.0


def _saliency_centroid(sal_map: np.ndarray) -> tuple[float, float]:
    H, W  = sal_map.shape
    Y, X  = np.mgrid[0:H, 0:W]
    tot   = sal_map.sum() + 1e-8
    cx    = float(np.clip((X * sal_map).sum() / tot / W, 0.05, 0.95))
    cy    = float(np.clip((Y * sal_map).sum() / tot / H, 0.05, 0.95))
    return cx, cy


# ════════════════════════════════════════════════════════════════════════════════
#  SHARPNESS SCORE
# ════════════════════════════════════════════════════════════════════════════════

def _sharpness(pil_img: Image.Image) -> float:
    small = np.array(pil_img.convert("L").resize((200, 200)), dtype=np.float32)
    try:
        import cv2
        return min(1.0, float(cv2.Laplacian(small.astype(np.uint8), cv2.CV_64F).var()) / 800.0)
    except ImportError:
        gy = np.abs(np.diff(small, axis=0))
        gx = np.abs(np.diff(small, axis=1))
        return min(1.0, (gy.var() + gx.var()) * 5 / 800.0)


# ════════════════════════════════════════════════════════════════════════════════
#  RULE-OF-THIRDS NUDGE
# ════════════════════════════════════════════════════════════════════════════════

THIRDS          = [1/3, 2/3]
THIRDS_STRENGTH = 0.55   # 0 = no nudge, 1 = full alignment


def _rule_of_thirds_nudge(cx: float, cy: float) -> tuple[float, float]:
    
    tx = min(THIRDS, key=lambda t: abs(cx - t))
    ty = min(THIRDS, key=lambda t: abs(cy - t))
    return (cx - tx) * THIRDS_STRENGTH, (cy - ty) * THIRDS_STRENGTH


# ════════════════════════════════════════════════════════════════════════════════
#  SUBJECT ANCHOR  —  get_saliency_roi()
# ════════════════════════════════════════════════════════════════════════════════

def get_saliency_roi(pil_img: Image.Image,
                     slot_w: int = 0,
                     slot_h: int = 0) -> tuple[float, float]:
    
    h = _img_hash(pil_img)
    if h in _saliency_cache:
        return _saliency_cache[h]

    ow, oh      = pil_img.size
    slot_ratio  = (slot_w / slot_h) if slot_h > 0 else 1.0

    try:    boxes = _run_blazeface(pil_img)
    except: boxes = []

    if boxes:
        # ── FACE PATH ─────────────────────────────────────────────────────────
        grp_x1 = min(b["x1"] for b in boxes)
        grp_y1 = min(b["y1"] for b in boxes)
        grp_x2 = max(b["x2"] for b in boxes)
        grp_y2 = max(b["y2"] for b in boxes)

        total_area  = sum(b["w"] * b["h"] for b in boxes)
        cx          = sum(b["cx"] * b["w"] * b["h"] for b in boxes) / total_area
        cx          = float(np.clip(cx, 0.05, 0.95))

        grp_h_ratio = (grp_y2 - grp_y1) / oh
        grp_top     = grp_y1 / oh
        grp_mid     = (grp_y1 + grp_y2) / 2 / oh

        if grp_h_ratio > 0.35:     cy = grp_mid
        elif grp_h_ratio > 0.15:   cy = grp_top + grp_h_ratio * 0.25
        else:                       cy = grp_top

        headroom = grp_h_ratio * 0.35
        cy = float(np.clip(cy - headroom, 0.04, 0.90))

        if slot_ratio < 0.75 and grp_h_ratio < 0.35:
            cy = float(np.clip(cy - 0.05, 0.04, 0.90))

    else:
       
        try:    sal_map = _run_u2net(pil_img)
        except: sal_map = _heuristic_saliency_map(pil_img)

        cx, cy = _saliency_centroid(sal_map)
        cy     = float(np.clip(cy - 0.04, 0.08, 0.90))

    _saliency_cache[h] = (cx, cy)
    return cx, cy


# ════════════════════════════════════════════════════════════════════════════════
#  PHOTO QUALITY SCORE  —  drives Hungarian assignment
# ════════════════════════════════════════════════════════════════════════════════

def get_photo_quality(pil_img: Image.Image) -> float:
   
    h = _img_hash(pil_img)
    if h in _quality_cache:
        return _quality_cache[h]

    ow, oh = pil_img.size

    try:    boxes = _run_blazeface(pil_img)   # hits cache if ROI already ran
    except: boxes = []

    if boxes:
        face_area  = sum(b["w"] * b["h"] for b in boxes)
        face_score = min(1.0, face_area / (ow * oh) * 5.0)
        if len(boxes) == 1 and boxes[0]["w"] * boxes[0]["h"] / (ow * oh) > 0.05:
            face_score = min(1.0, face_score * 1.15)
    else:
        face_score = 0.0

    try:    sal_score = float(_run_u2net(pil_img).mean()) 
    except: sal_score = 0.3

    sharp_score = _sharpness(pil_img)
    if boxes:
      score = 0.60 * face_score + 0.25 * sal_score + 0.15 * sharp_score
    else:
     score = 0.70 * sal_score + 0.30 * sharp_score
    
    _quality_cache[h] = score
    return score



def _precompute_all(pil_imgs: list[Image.Image]) -> None:
   
    def _process(pil: Image.Image):
        h = _img_hash(pil)
        # Skip entirely if all three are already cached
        if h in _saliency_cache and h in _quality_cache and h in _face_cache:
            return
        try:
            _run_blazeface(pil)       
            _run_u2net(pil)           
            get_saliency_roi(pil)    
            get_photo_quality(pil)    
        except Exception as e:
            print(f"[Precompute] Error on image: {e}")

    workers = min(len(pil_imgs), 4)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_process, pil_imgs))

    print(f"[Precompute] Done — {len(pil_imgs)} images processed in parallel")

#  HUNGARIAN SLOT ASSIGNMENT

def match_images_to_slots(pil_images: list, slots_def: list,
                          canvas_w: int, canvas_h: int) -> list[int]:
    n = min(len(slots_def), len(pil_images))
    if n == 0:
        return []

    img_ratios   = [img.size[0] / img.size[1] for img in pil_images[:n]]
    slot_ratios, slot_areas = [], []
    for (xf, yf, wf, hf) in slots_def[:n]:
        sw = max(1, wf * canvas_w); sh = max(1, hf * canvas_h)
        slot_ratios.append(sw / sh); slot_areas.append(sw * sh)

    max_area         = max(slot_areas)
    slot_importances = [a / max_area for a in slot_areas]
    qualities        = [get_photo_quality(img) for img in pil_images[:n]]   # cached

    try:    face_counts = [len(_run_blazeface(img)) for img in pil_images[:n]]  # cached
    except: face_counts = [0] * n

    cost = np.zeros((n, n), dtype=np.float64)
    for ii in range(n):
        for jj in range(n):
            rd       = abs(img_ratios[ii] - slot_ratios[jj])
            has_face = face_counts[ii] > 0
            if has_face:
                ratio_pen   = rd**2 * 1.0
                quality_rew = qualities[ii] * slot_importances[jj] * 3.0
            else:
                ratio_pen   = rd**2 * 2.5
                quality_rew = qualities[ii] * slot_importances[jj] * 1.8
            cost[ii][jj] = ratio_pen - quality_rew

    try:
        from scipy.optimize import linear_sum_assignment
        pi, si = linear_sum_assignment(cost)
        order  = [0] * n
        for p, s in zip(pi, si):
            order[int(s)] = int(p)
        print(f"[Assignment] Hungarian optimal — cost: {cost[pi, si].sum():.3f}")
        return order
    except ImportError:
        print("[Assignment] scipy not installed — greedy fallback")
        used, order = set(), [0] * n
        for si in sorted(range(n), key=lambda i: -slot_areas[i]):
            best = min((ii for ii in range(n) if ii not in used),
                       key=lambda ii: cost[ii][si], default=None)
            if best is None:
                best = next(i for i in range(n) if i not in used)
            order[si] = best
            used.add(best)
        return order


# ════════════════════════════════════════════════════════════════════════════════
#  CROP-FIRST v2 PLACEMENT
# ════════════════════════════════════════════════════════════════════════════════

def _subject_crop_v2(pil_img:  Image.Image,
                     target_w: int, target_h: int,
                     cx: float, cy: float,
                     has_face: bool = False) -> Image.Image:
    
    src_w, src_h = pil_img.size
    slot_ratio   = target_w / target_h
    src_ratio    = src_w    / src_h

    if src_ratio > slot_ratio:
        crop_h = src_h; crop_w = int(src_h * slot_ratio)
    else:
        crop_w = src_w; crop_h = int(src_w / slot_ratio)
    crop_w = max(1, crop_w); crop_h = max(1, crop_h)

    cx_px = cx * src_w
    cy_px = cy * src_h

    dx, dy         = _rule_of_thirds_nudge(cx, cy)
    thirds_shift_x = dx * crop_w
    thirds_shift_y = dy * crop_h

    face_padding_y = 0.0
    if has_face and slot_ratio < 0.8:
        face_padding_y = crop_h * 0.20 * 0.30

    ideal_left = cx_px - crop_w / 2 + thirds_shift_x
    ideal_top  = cy_px - crop_h / 2 + thirds_shift_y - face_padding_y

    left = int(max(0, min(ideal_left, src_w - crop_w)))
    top  = int(max(0, min(ideal_top,  src_h - crop_h)))

    return pil_img.crop((left, top, left + crop_w, top + crop_h))


def place_photo(pil_img:  Image.Image,
                target_w: int, target_h: int,
                cx: float, cy: float,
                zoom:  float = 1.0,
                pan_x: float = 0.0,
                pan_y: float = 0.0) -> Image.Image:
   
    if target_w <= 0 or target_h <= 0:
        return pil_img.resize((max(1, target_w), max(1, target_h)), Image.BILINEAR)

    try:    boxes = _run_blazeface(pil_img)   # always cached by here
    except: boxes = []
    has_face = len(boxes) > 0

    cropped        = _subject_crop_v2(pil_img, target_w, target_h,
                                       cx, cy, has_face=has_face)
    crop_w, crop_h = cropped.size

    if zoom > 1.0 or pan_x != 0.0 or pan_y != 0.0:
        zoomed_w = max(target_w, int(crop_w * zoom))
        zoomed_h = max(target_h, int(crop_h * zoom))
        zoomed   = cropped.resize((zoomed_w, zoomed_h), Image.BILINEAR)
        base_x   = (zoomed_w - target_w) // 2
        base_y   = (zoomed_h - target_h) // 2
        off_x    = int(pan_x * target_w)
        off_y    = int(pan_y * target_h)
        win_x    = max(0, min(base_x + off_x, zoomed_w - target_w))
        win_y    = max(0, min(base_y + off_y, zoomed_h - target_h))
        cropped  = zoomed.crop((win_x, win_y, win_x + target_w, win_y + target_h))

    if cropped.size != (target_w, target_h):
        cropped = cropped.resize((target_w, target_h), Image.BILINEAR)

    return cropped

#  PASTE HELPERS


def _mask_rounded(w: int, h: int, r: int) -> Image.Image:
    from PIL import ImageDraw
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, w-1, h-1], radius=r, fill=255)
    return m


def paste_slot(canvas: Image.Image, img: Image.Image,
               x: int, y: int, w: int, h: int, corner: int = 0):
    img = img.resize((w, h), Image.BILINEAR)
    canvas.paste(img, (x, y), _mask_rounded(w, h, corner)) if corner > 0 \
        else canvas.paste(img, (x, y))


def paste_circle(canvas: Image.Image, img: Image.Image,
                 x: int, y: int, w: int, h: int):
    from PIL import ImageDraw
    img  = img.resize((w, h), Image.BILINEAR)
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).ellipse([0, 0, w-1, h-1], fill=255)
    canvas.paste(img, (x, y), mask)


def paste_polaroid(canvas: Image.Image, photo: Image.Image,
                   x: int, y: int, w: int, h: int, angle: int = 0):
    border = int(w * .06); bot = int(h * .18)
    pw, ph = w - border*2, h - border - bot
    frame  = Image.new("RGB", (w, h), (252, 248, 240))
    cx, cy = get_saliency_roi(photo, pw, ph)
    inner  = place_photo(photo, pw, ph, cx, cy)
    frame.paste(inner, (border, border))
    if angle:
        frame = frame.rotate(angle, expand=True, resample=Image.BICUBIC,
                             fillcolor=(160, 148, 132))
    fw, fh = frame.size
    canvas.paste(frame, (x - fw//2 + w//2, y - fh//2 + h//2))

#  COLLAGE BUILDER


def build_collage(images_b64: list[str],
                  template_key: str,
                  adjustments: list | None = None,
                  assignment:  list | None = None) -> tuple:
    t0   = time.perf_counter()
    tmpl = COLLAGE_TEMPLATES[template_key]
    CW, CH  = tmpl["canvas_ratio"]
    gap     = tmpl.get("gap", 6)
    bg_col  = tmpl.get("bg", (255, 255, 255))
    corner  = tmpl.get("corner", 0)
    is_pol  = tmpl.get("polaroid", False)
    circ_sl = tmpl.get("circle_slot", -1)
    slots   = tmpl["slots_def"]

    
    pil_imgs = [decode_image(b) for b in images_b64]

    
    _precompute_all(pil_imgs)

    if assignment is None:
        assignment = match_images_to_slots(pil_imgs, slots, CW, CH)

    canvas   = Image.new("RGB", (CW, CH), bg_col)
    roi_list = []

    for i, (xf, yf, wf, hf) in enumerate(slots):
        if i >= len(assignment): break
        idx = assignment[i]
        if idx >= len(pil_imgs): break

        pil = pil_imgs[idx]
        ts  = time.perf_counter()

        sx = int(xf * CW) + gap // 2;  sw = max(4, int(wf * CW) - gap)
        sy = int(yf * CH) + gap // 2;  sh = max(4, int(hf * CH) - gap)

        cx, cy = get_saliency_roi(pil, sw, sh)   # instant — cached

        adj   = adjustments[i] if adjustments and i < len(adjustments) \
                else {"zoom": 100, "panX": 0, "panY": 0}
        zoom  = max(1.0, adj.get("zoom",  100) / 100.0)
        pan_x = adj.get("panX", 0) / 100.0
        pan_y = adj.get("panY", 0) / 100.0

        if is_pol:
            paste_polaroid(canvas, pil, sx, sy, sw, sh,
                           angle=POLAROID_ANGLES[i % len(POLAROID_ANGLES)])
        elif i == circ_sl:
            size   = min(sw, sh)
            placed = place_photo(pil, size, size, cx, cy, zoom, pan_x, pan_y)
            paste_circle(canvas, placed, sx + (sw-size)//2, sy + (sh-size)//2, size, size)
        else:
            placed = place_photo(pil, sw, sh, cx, cy, zoom, pan_x, pan_y)
            paste_slot(canvas, placed, sx, sy, sw, sh, corner)

        roi_list.append({
            "slot":    i + 1,
            "cx":      round(cx * 100, 1),
            "cy":      round(cy * 100, 1),
            "time_ms": round((time.perf_counter() - ts) * 1000, 1),
        })

    elapsed = round(time.perf_counter() - t0, 3)
    print(f"[Build] Complete in {elapsed}s")
    return canvas, roi_list, elapsed, assignment


def collage_to_b64(canvas: Image.Image, quality: int = 98) -> str:
    buf = io.BytesIO()
    canvas.save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

#  ROUTES


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/templates")
def api_templates():
    return jsonify({
        k: {
            "name":        v["name"],
            "slots":       v["slots"],
            "preview_img": v["preview_img"],
            "circle_slot": v.get("circle_slot", -1),
        }
        for k, v in COLLAGE_TEMPLATES.items()
    })


@app.route("/api/slot_defs/<tk>")
def api_slot_defs(tk):
    if tk not in COLLAGE_TEMPLATES:
        return jsonify({"error": "Unknown template"}), 400
    return jsonify([
        {"xf": s[0], "yf": s[1], "wf": s[2], "hf": s[3]}
        for s in COLLAGE_TEMPLATES[tk]["slots_def"]
    ])


@app.route("/api/saliency", methods=["POST"])
def api_saliency():
    d = request.get_json(force=True)
    try:
        pil    = decode_image(d.get("image", ""))
        cx, cy = get_saliency_roi(pil)
        return jsonify({"cx": round(cx, 4), "cy": round(cy, 4)})
    except Exception as e:
        return jsonify({"cx": 0.5, "cy": 0.45, "error": str(e)})


@app.route("/api/saliency_batch", methods=["POST"])
def api_saliency_batch():
    imgs = request.get_json(force=True).get("images", [])
    if not imgs:
        return jsonify({"results": []}), 200

    pil_imgs = []
    for b64 in imgs:
        try:   pil_imgs.append(decode_image(b64))
        except: pil_imgs.append(None)

    valid = [p for p in pil_imgs if p is not None]
    if valid:
        _precompute_all(valid)   # parallel warm-up

    results = []
    for p in pil_imgs:
        if p is None:
            results.append({"cx": 0.5, "cy": 0.45})
        else:
            try:
                cx, cy = get_saliency_roi(p)
                results.append({"cx": round(cx, 4), "cy": round(cy, 4)})
            except Exception:
                results.append({"cx": 0.5, "cy": 0.45})

    return jsonify({"results": results}), 200


@app.route("/api/debug_roi", methods=["POST"])
def api_debug_roi():
    d = request.get_json(force=True)
    try:
        pil    = decode_image(d.get("image", ""))
        boxes  = _run_blazeface(pil)
        cx, cy = get_saliency_roi(pil)
        q      = get_photo_quality(pil)
        ow, oh = pil.size
        fhr    = 0.0
        if boxes:
            fhr = (max(b["y2"] for b in boxes) - min(b["y1"] for b in boxes)) / oh
        ndx, ndy = _rule_of_thirds_nudge(cx, cy)
        return jsonify({
            "placement":      "crop_first_v2_optimised",
            "faces_detected": len(boxes),
            "face_boxes":     [{k: round(v, 2) if isinstance(v, float) else v
                                for k, v in b.items()} for b in boxes],
            "cx":             round(cx, 4),
            "cy":             round(cy, 4),
            "thirds_nudge_x": round(ndx, 4),
            "thirds_nudge_y": round(ndy, 4),
            "quality":        round(q, 4),
            "sharpness":      round(_sharpness(pil), 4),
            "path":           "face" if boxes else "saliency",
            "face_h_ratio":   round(fhr, 3),
            "cache_hits":     {
                "saliency": _img_hash(pil) in _saliency_cache,
                "quality":  _img_hash(pil) in _quality_cache,
                "face":     _img_hash(pil) in _face_cache,
            },
        })
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()})


def _gen(data: dict) -> tuple[dict, int]:
    imgs   = data.get("images", [])
    tk     = data.get("template", "modern_blocks")
    qual   = int(data.get("quality", 98))
    adjs   = data.get("adjustments", None)
    asgn   = data.get("assignment", None)

    if tk not in COLLAGE_TEMPLATES:
        return {"error": f"Unknown template: {tk}"}, 400

    needed = COLLAGE_TEMPLATES[tk]["slots"]
    if len(imgs) < needed:
        return {"error": f"Need {needed} images, got {len(imgs)}"}, 400

    try:
        canvas, rois, elapsed, used = build_collage(imgs[:needed], tk, adjs, asgn)
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}, 500

    CW, CH = COLLAGE_TEMPLATES[tk]["canvas_ratio"]
    return {
        "collage":    collage_to_b64(canvas, qual),
        "rois":       rois,
        "time_s":     elapsed,
        "canvas":     f"{CW}×{CH}",
        "template":   COLLAGE_TEMPLATES[tk]["name"],
        "slots":      needed,
        "assignment": used,
        "placement":  "crop_first_v2",
    }, 200


@app.route("/api/generate", methods=["POST"])
def api_generate():
    r, s = _gen(request.get_json(force=True))
    return jsonify(r), s


@app.route("/api/generate_adjusted", methods=["POST"])
def api_generate_adjusted():
    r, s = _gen(request.get_json(force=True))
    return jsonify(r), s


def _warmup_models():
   
    print("[Warmup] Pre-loading AI models…")
    _get_blazeface()
    try:
        dummy = Image.new("RGB", (64, 64), (128, 128, 128))
        _run_u2net(dummy)
        print("[Warmup] Models ready ✓")
    except Exception as e:
        print(f"[Warmup] U²-Net warm-up failed (will load on first request): {e}")










@app.route("/debug")
def debug_ui():
    html = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>JioPics — Saliency Debugger</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0f0f0f;color:#e0e0e0;min-height:100vh;padding:20px}
h1{font-size:18px;font-weight:500;margin-bottom:4px}
.sub{font-size:13px;color:#888;margin-bottom:20px}
.top-bar{display:flex;align-items:center;gap:10px;margin-bottom:16px;flex-wrap:wrap}
.top-bar input{background:#1c1c1c;border:1px solid #333;color:#e0e0e0;padding:7px 12px;border-radius:8px;font-size:13px;flex:1;min-width:180px}
.top-bar button{background:#1c1c1c;border:1px solid #444;color:#e0e0e0;padding:7px 14px;border-radius:8px;font-size:13px;cursor:pointer;white-space:nowrap;transition:background .15s}
.top-bar button:hover{background:#2a2a2a}
.conn-dot{width:9px;height:9px;border-radius:50%;background:#444;flex-shrink:0}
.conn-dot.ok{background:#1D9E75}.conn-dot.err{background:#E24B4A}
#conn-label{font-size:12px;color:#888}
.overlay-bar{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px;align-items:center}
.overlay-bar span{font-size:12px;color:#888}
.ot{font-size:12px;padding:4px 11px;border-radius:20px;cursor:pointer;border:1px solid #333;background:transparent;color:#888;transition:all .15s}
.ot.on{background:#0a3060;color:#6ab4ff;border-color:#1a5090}
.drop-zone{border:1.5px dashed #333;border-radius:12px;padding:30px;text-align:center;cursor:pointer;transition:background .15s;margin-bottom:16px}
.drop-zone:hover,.drop-zone.drag{background:#1a1a1a}
.drop-zone p{font-size:13px;color:#666;margin-top:8px}
.legend{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:16px;font-size:12px;color:#888;align-items:center}
.leg-dot{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:4px;vertical-align:middle}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:16px}
.img-card{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:12px;overflow:hidden}
.canvas-wrap{position:relative;background:#111;width:100%}
.canvas-wrap canvas{display:block;width:100%;height:auto}
.ov{position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none}
.card-body{padding:12px}
.pill-row{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px}
.pill{font-size:11px;font-weight:500;padding:3px 9px;border-radius:20px}
.p-blue{background:#0a3060;color:#6ab4ff}
.p-amber{background:#3d2000;color:#f5a623}
.p-teal{background:#00291d;color:#2dd4a0}
.p-gray{background:#222;color:#999}
.p-red{background:#3d0000;color:#ff6b6b}
.stat-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:8px}
.stat{background:#111;border-radius:8px;padding:8px 10px}
.stat .v{font-size:15px;font-weight:500;color:#e0e0e0}
.stat .l{font-size:11px;color:#666;margin-top:2px}
.bar-bg{height:4px;background:#2a2a2a;border-radius:2px;margin-top:5px;overflow:hidden}
.bar-fg{height:100%;border-radius:2px;transition:width .4s}
.log{font-family:monospace;font-size:11px;color:#666;background:#111;border-radius:6px;padding:8px 10px;max-height:90px;overflow-y:auto;line-height:1.7;border:1px solid #222}
.log .ok{color:#1D9E75}.log .err{color:#E24B4A}.log .info{color:#6ab4ff}
.spinner{width:16px;height:16px;border:2px solid #333;border-top-color:#6ab4ff;border-radius:50%;animation:spin .7s linear infinite;display:inline-block;vertical-align:middle;margin-right:6px}
@keyframes spin{to{transform:rotate(360deg)}}
.phase-bar{display:flex;gap:0;margin:8px 0;border-radius:6px;overflow:hidden;height:22px;font-size:11px}
.phase{display:flex;align-items:center;justify-content:center;flex:1;color:#000;font-weight:500;transition:opacity .3s}
.phase.dim{opacity:.3}
</style>
</head>
<body>
<h1>JioPics — Live Saliency Debugger</h1>
<p class="sub">Drop any photo to see exactly how your backend processes it — face detection, U²-Net saliency, ROI anchor, and crop preview.</p>

<div class="top-bar">
  <span style="font-size:13px;color:#888;white-space:nowrap">Server:</span>
  <input type="text" id="srv" value="http://127.0.0.1:5000">
  <button onclick="testConn()">Test connection</button>
  <span class="conn-dot" id="cdot"></span>
  <span id="clabel">not tested</span>
</div>

<div class="overlay-bar">
  <span>Overlays:</span>
  <button class="ot on" id="tog-heatmap"  onclick="tog('heatmap',this)">Saliency heatmap</button>
  <button class="ot on" id="tog-anchor"   onclick="tog('anchor',this)">Anchor (cx,cy)</button>
  <button class="ot on" id="tog-faces"    onclick="tog('faces',this)">Face boxes</button>
  <button class="ot on" id="tog-thirds"   onclick="tog('thirds',this)">Rule of thirds</button>
  <button class="ot on" id="tog-crop"     onclick="tog('crop',this)">Crop preview</button>
  <button class="ot on" id="tog-salmap"   onclick="tog('salmap',this)">Real sal-map</button>
</div>

<div class="drop-zone" id="dz" onclick="document.getElementById('fi').click()"
  ondragover="event.preventDefault();this.classList.add('drag')"
  ondragleave="this.classList.remove('drag')"
  ondrop="onDrop(event)">
  <input type="file" id="fi" accept="image/*" multiple style="display:none" onchange="handleFiles(this.files)">
  <svg width="32" height="32" fill="none" stroke="#555" stroke-width="1.5" viewBox="0 0 24 24"><path d="M4 16l4-4 4 4 4-6 4 6"/><rect x="3" y="3" width="18" height="18" rx="2"/></svg>
  <p>Drop photos here or click to browse — processes each through /api/debug_roi in real time</p>
</div>

<div class="legend">
  <span><span class="leg-dot" style="background:rgba(255,100,0,.8)"></span>High saliency</span>
  <span><span class="leg-dot" style="background:rgba(0,100,255,.6)"></span>Low saliency</span>
  <span><span class="leg-dot" style="background:#00ff88"></span>Anchor (cx,cy)</span>
  <span><span class="leg-dot" style="background:#ff4444;border-radius:2px"></span>Face box</span>
  <span><span class="leg-dot" style="background:rgba(255,255,255,.15);border:1px solid rgba(255,255,255,.4);border-radius:2px"></span>Crop frame</span>
</div>

<div class="grid" id="grid"></div>

<script>
const OV = {heatmap:true,anchor:true,faces:true,thirds:true,crop:true,salmap:true};
const CARDS = {};

function srv(){ return document.getElementById('srv').value.replace(/\/$/,''); }

async function testConn(){
  const dot=document.getElementById('cdot'), lbl=document.getElementById('clabel');
  lbl.textContent='testing…';
  try{
    const r=await fetch(srv()+'/api/templates',{signal:AbortSignal.timeout(4000)});
    if(r.ok){dot.className='conn-dot ok';lbl.textContent='connected ✓';}
    else throw new Error('HTTP '+r.status);
  }catch(e){dot.className='conn-dot err';lbl.textContent='failed — '+e.message;}
}

function tog(key,btn){
  OV[key]=!OV[key];
  btn.classList.toggle('on',OV[key]);
  Object.values(CARDS).forEach(c=>redraw(c));
}

function onDrop(e){
  e.preventDefault();
  document.getElementById('dz').classList.remove('drag');
  handleFiles(e.dataTransfer.files);
}

function handleFiles(files){
  [...files].slice(0,8).forEach(f=>{ if(f.type.startsWith('image/')) processFile(f); });
}

function b64(file){ return new Promise(res=>{ const r=new FileReader(); r.onload=e=>res(e.target.result); r.readAsDataURL(file); }); }

async function processFile(file){
  const id='c'+Date.now()+Math.random().toString(36).slice(2);
  const data64=await b64(file);

  const card=document.createElement('div');
  card.className='img-card';
  card.innerHTML=`
    <div class="canvas-wrap" id="wrap_${id}">
      <canvas id="base_${id}"></canvas>
      <canvas class="ov" id="ov_${id}"></canvas>
    </div>
    <div class="card-body">
      <div class="phase-bar" id="phases_${id}">
        <div class="phase dim" style="background:#1a4a8a" id="ph1_${id}">1. Decode</div>
        <div class="phase dim" style="background:#0a6060" id="ph2_${id}">2. MTCNN</div>
        <div class="phase dim" style="background:#5a3a00" id="ph3_${id}">3. U²-Net</div>
        <div class="phase dim" style="background:#3a0060" id="ph4_${id}">4. ROI</div>
        <div class="phase dim" style="background:#1a4000" id="ph5_${id}">5. Quality</div>
        <div class="phase dim" style="background:#5a2000" id="ph6_${id}">6. Done</div>
      </div>
      <div class="pill-row" id="pills_${id}">
        <span class="pill p-gray"><span class="spinner"></span>Calling /api/debug_roi…</span>
      </div>
      <div class="stat-grid" id="stats_${id}"></div>
      <div class="log" id="log_${id}"></div>
    </div>`;
  document.getElementById('grid').appendChild(card);

  const img=new Image();
  img.onload=()=>{
    const bc=document.getElementById('base_'+id);
    const oc=document.getElementById('ov_'+id);
    bc.width=img.naturalWidth; bc.height=img.naturalHeight;
    oc.width=img.naturalWidth; oc.height=img.naturalHeight;
    bc.getContext('2d').drawImage(img,0,0);
    CARDS[id]={data64,img,bc,oc,data:null,salImg:null};
    addLog(id,'info','→ Image loaded ('+img.naturalWidth+'×'+img.naturalHeight+')');
    phase(id,1);
    runDebug(id);
  };
  img.src=data64;
}

function phase(id,n){
  for(let i=1;i<=6;i++){
    const el=document.getElementById('ph'+i+'_'+id);
    if(el) el.classList.toggle('dim', i>n);
  }
}

function addLog(id,cls,msg){
  const el=document.getElementById('log_'+id);
  if(!el)return;
  const line=document.createElement('div');
  line.className=cls;
  line.textContent=msg;
  el.appendChild(line);
  el.scrollTop=el.scrollHeight;
}

async function runDebug(id){
  const c=CARDS[id];
  const t0=performance.now();
  addLog(id,'info','→ POST '+srv()+'/api/debug_roi');
  phase(id,2);

  try{
    const resp=await fetch(srv()+'/api/debug_roi',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({image:c.data64})
    });
    if(!resp.ok) throw new Error('HTTP '+resp.status);
    const d=await resp.json();
    const ms=(performance.now()-t0).toFixed(0);
    phase(id,3);
    addLog(id,'ok','← '+ms+'ms | path='+d.path+' | faces='+d.faces_detected);
    phase(id,4);
    addLog(id,'info','  cx='+d.cx+' cy='+d.cy);
    phase(id,5);
    addLog(id,'info','  quality='+d.quality+' sharpness='+d.sharpness);
    if(d.thirds_nudge_x!==undefined)
      addLog(id,'info','  thirds nudge x='+d.thirds_nudge_x+' y='+d.thirds_nudge_y);
    addLog(id,'ok','  cache: sal='+d.cache_hits?.saliency+' q='+d.cache_hits?.quality+' face='+d.cache_hits?.face);
    phase(id,6);

    CARDS[id].data=d;
    renderPills(id,d);
    renderStats(id,d);

    if(OV.salmap) await fetchSalMap(id);
    redraw(CARDS[id]);

  }catch(e){
    addLog(id,'err','✕ '+e.message);
    addLog(id,'err','Make sure Flask is running at '+srv());
    document.getElementById('pills_'+id).innerHTML='<span class="pill p-red">Connection failed — see log</span>';
  }
}

async function fetchSalMap(id){
  const c=CARDS[id];
  try{
    addLog(id,'info','→ POST /api/saliency_map (real U²-Net map)');
    const resp=await fetch(srv()+'/api/saliency_map',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({image:c.data64})
    });
    if(!resp.ok){ addLog(id,'err','  /api/saliency_map not found — using approx heatmap'); return; }
    const blob=await resp.blob();
    const url=URL.createObjectURL(blob);
    const img=new Image();
    await new Promise(res=>{ img.onload=res; img.src=url; });
    CARDS[id].salImg=img;
    addLog(id,'ok','  Real sal-map loaded ('+img.width+'×'+img.height+')');
  }catch(e){
    addLog(id,'err','  sal-map fetch failed: '+e.message);
  }
}

function renderPills(id,d){
  const el=document.getElementById('pills_'+id);
  const path=d.path==='face'
    ?'<span class="pill p-blue">MTCNN face path</span>'
    :'<span class="pill p-amber">U²-Net saliency path</span>';
  const cache=d.cache_hits?.saliency
    ?'<span class="pill p-teal">cache hit</span>'
    :'<span class="pill p-gray">fresh compute</span>';
  const faces=d.faces_detected>0
    ?'<span class="pill p-blue">'+d.faces_detected+' face'+(d.faces_detected>1?'s':'')+'</span>':'' ;
  el.innerHTML=path+cache+faces;
}

function renderStats(id,d){
  const el=document.getElementById('stats_'+id);
  const q=(d.quality*100).toFixed(1);
  const sh=(d.sharpness*100).toFixed(1);
  const fhr=d.face_h_ratio?((d.face_h_ratio*100).toFixed(1)+'%'):'—';
  el.innerHTML=`
    <div class="stat"><div class="v">${(d.cx*100).toFixed(1)}%, ${(d.cy*100).toFixed(1)}%</div><div class="l">Anchor (cx, cy)</div></div>
    <div class="stat"><div class="v">${q}%<div class="bar-bg"><div class="bar-fg" style="width:${q}%;background:#378ADD"></div></div></div><div class="l">Quality score</div></div>
    <div class="stat"><div class="v">${sh}%<div class="bar-bg"><div class="bar-fg" style="width:${sh}%;background:#1D9E75"></div></div></div><div class="l">Sharpness</div></div>
    <div class="stat"><div class="v">${d.faces_detected} &nbsp;<span style="font-size:12px;color:#666">face-h: ${fhr}</span></div><div class="l">Faces detected</div></div>`;
}

function redraw(c){
  if(!c.oc)return;
  const cv=c.oc, W=cv.width, H=cv.height;
  const ctx=cv.getContext('2d');
  ctx.clearRect(0,0,W,H);

  if(OV.thirds){
    ctx.save();
    ctx.strokeStyle='rgba(255,255,255,0.18)';
    ctx.lineWidth=Math.max(1,W*0.002);
    ctx.setLineDash([6,4]);
    [1/3,2/3].forEach(f=>{
      ctx.beginPath();ctx.moveTo(W*f,0);ctx.lineTo(W*f,H);ctx.stroke();
      ctx.beginPath();ctx.moveTo(0,H*f);ctx.lineTo(W,H*f);ctx.stroke();
    });
    ctx.restore();
  }

  const d=c.data;
  if(!d)return;

  if(OV.salmap && c.salImg){
    ctx.save();
    ctx.globalAlpha=0.55;
    ctx.drawImage(c.salImg,0,0,W,H);
    ctx.restore();
  } else if(OV.heatmap){
    drawApproxHeatmap(ctx,W,H,d.cx,d.cy);
  }

  if(OV.faces && d.face_boxes?.length){
    d.face_boxes.forEach(b=>{
      const lw=Math.max(2,W*0.004);
      ctx.strokeStyle='#ff4444';
      ctx.lineWidth=lw;
      ctx.strokeRect(b.x1,b.y1,b.x2-b.x1,b.y2-b.y1);
      ctx.fillStyle='rgba(255,68,68,0.12)';
      ctx.fillRect(b.x1,b.y1,b.x2-b.x1,b.y2-b.y1);
      ctx.font='bold '+Math.max(11,W*0.022)+'px monospace';
      ctx.fillStyle='#ff6666';
      ctx.fillText('conf '+b.conf.toFixed(2),b.x1+4,b.y1>16?b.y1-5:b.y2+14);
    });
  }

  if(OV.crop) drawCropFrames(ctx,W,H,d.cx,d.cy);

  if(OV.anchor){
    const cx=d.cx*W, cy=d.cy*H;
    const r=Math.max(10,W*0.018);
    ctx.strokeStyle='#00ff88'; ctx.lineWidth=Math.max(2,W*0.004);
    ctx.beginPath(); ctx.arc(cx,cy,r,0,Math.PI*2); ctx.stroke();
    ctx.fillStyle='#00ff88';
    ctx.beginPath(); ctx.arc(cx,cy,3,0,Math.PI*2); ctx.fill();
    const cross=r*1.7;
    ctx.strokeStyle='rgba(0,255,136,0.6)'; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(cx-cross,cy); ctx.lineTo(cx+cross,cy); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cx,cy-cross); ctx.lineTo(cx,cy+cross); ctx.stroke();
    ctx.font='bold '+Math.max(11,W*0.024)+'px monospace';
    ctx.fillStyle='#00ff88';
    ctx.shadowColor='#000'; ctx.shadowBlur=4;
    ctx.fillText('('+( d.cx*100).toFixed(1)+'%, '+(d.cy*100).toFixed(1)+'%)',cx+r+5,cy+4);
    ctx.shadowBlur=0;
  }
}

function drawApproxHeatmap(ctx,W,H,cx,cy){
  const N=60, cw=W/N, ch=H/N;
  for(let row=0;row<N;row++){
    for(let col=0;col<N;col++){
      const px=(col+.5)/N, py=(row+.5)/N;
      const g=Math.exp(-((px-.5)**2/(.40**2)+(py-.5)**2/(.40**2))/2);
      const s=Math.exp(-((px-cx)**2/(.22**2)+(py-cy)**2/(.22**2))/2);
      const v=Math.min(1,g*.35+s*.65);
      if(v<.04)continue;
      const r=Math.round(255*Math.min(1,v*2.2));
      const gb=Math.round(255*(1-v));
      ctx.fillStyle=`rgba(${r},${Math.round(r*.35)},${gb},${(v*.6).toFixed(2)})`;
      ctx.fillRect(col*cw,row*ch,cw+1,ch+1);
    }
  }
}

function drawCropFrames(ctx,W,H,cx,cy){
  [{r:1,label:'1:1',stroke:'rgba(255,255,255,0.7)',dash:[]},
   {r:4/3,label:'4:3',stroke:'rgba(100,200,255,0.8)',dash:[8,4]},
   {r:9/16,label:'9:16',stroke:'rgba(255,200,80,0.8)',dash:[4,4]}
  ].forEach(item=>{
    let cw,ch;
    if(W/H>item.r){ch=H;cw=H*item.r;}else{cw=W;ch=W/item.r;}
    cw=Math.round(cw);ch=Math.round(ch);
    const left=Math.max(0,Math.min(cx*W-cw/2,W-cw));
    const top =Math.max(0,Math.min(cy*H-ch/2,H-ch));
    const lw=Math.max(1.5,W*.003);
    ctx.strokeStyle=item.stroke; ctx.lineWidth=lw; ctx.setLineDash(item.dash);
    ctx.strokeRect(left+lw/2,top+lw/2,cw-lw,ch-lw);
    ctx.setLineDash([]);
    ctx.font='bold '+Math.max(11,W*.024)+'px monospace';
    ctx.fillStyle=item.stroke;
    ctx.shadowColor='#000';ctx.shadowBlur=4;
    ctx.fillText(item.label,left+6,top+Math.max(16,W*.028));
    ctx.shadowBlur=0;
  });
}
</script>
</body>
</html>"""
    return html





@app.route("/api/saliency_map", methods=["POST"])
def api_saliency_map():
    """Returns the actual U²-Net saliency heatmap as a colourised PNG."""
    import struct, zlib
    d = request.get_json(force=True)
    try:
        pil = decode_image(d.get("image", ""))
        try:
            sal = _run_u2net(pil)          # hits cache if already computed
        except Exception:
            sal = _heuristic_saliency_map(pil)

        h, w = sal.shape
        # Colourise: low=blue → mid=green → high=red  (BGR heat)
        r = np.clip(sal * 2 - 1, 0, 1)
        g = np.clip(1 - np.abs(sal * 2 - 1), 0, 1)
        b = np.clip(1 - sal * 2, 0, 1)
        rgb = (np.stack([r, g, b], axis=2) * 255).astype(np.uint8)

        out_img = Image.fromarray(rgb, "RGB").resize(pil.size, Image.BILINEAR)
        buf = io.BytesIO()
        out_img.save(buf, "PNG")
        buf.seek(0)
        from flask import send_file
        return send_file(buf, mimetype="image/png")
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    
#  MAIN

if __name__ == "__main__":
    print("=" * 60)
    print("  JIOPICS  —  AI Collage Studio  (production build)")
    print(f"  Templates    : {len(COLLAGE_TEMPLATES)}")
    print( "  Face detect  : MTCNN (facenet_pytorch)")
    print( "  Saliency     : U²-Net 400px → heuristic fallback")
    print( "  Crop         : crop-first v2 + rule-of-thirds nudge")
    print( "  Assignment   : Hungarian (scipy) → greedy fallback")
    print( "  Speed        : parallel ROI + disk cache")
    print(f"  URL          : http://127.0.0.1:5000")
    print("=" * 60)

    _warmup_models()

    app.run(
        debug=False,
        port=5000,
        host="0.0.0.0",
        use_reloader=False,
        threaded=True,
    )
