from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import base64, hashlib, io, os, time
from PIL import Image, ImageFilter
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
ROOT         = os.path.dirname(BASE_DIR)
FRONTEND_DIR = os.path.join(ROOT, "frontend")
ASSETS_DIR   = os.path.join(ROOT, "assets", "collage_templates")

app = Flask(__name__,
            template_folder=os.path.join(FRONTEND_DIR, "templates"),
            static_folder=os.path.join(FRONTEND_DIR, "static"))
CORS(app)


#CROP_MODE = "fit"       # ← change to "crop" to restore original behaviour
CROP_MODE = "crop"    # ← uncomment this line and comment above to crop

print(f"[CROP_MODE] active mode: '{CROP_MODE}'")


# ── Caches (in-memory, keyed by image hash) ───────────────────────────────────
_saliency_cache: dict[str, tuple[float, float]] = {}
_quality_cache:  dict[str, float]               = {}
_face_cache:     dict[str, list]                = {}
_sal_map_cache:  dict[str, np.ndarray]          = {}

# ── Static assets ──────────────────────────────────────────────────────────────
@app.route("/assets/collage_templates/<path:filename>")
def serve_asset(filename):
    return send_from_directory(ASSETS_DIR, filename)


# ════════════════════════════════════════════════════════════════════════════
#  TEMPLATE LOADER
# ════════════════════════════════════════════════════════════════════════════

def _load_templates() -> dict:
    import json
    json_path = os.path.join(BASE_DIR, "templates.json")
    if not os.path.exists(json_path):
        print(f"[WARNING] templates.json not found at {json_path}")
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
    print(f"[Templates] Loaded {len(templates)} templates from templates.json")
    return templates


COLLAGE_TEMPLATES = _load_templates()
POLAROID_ANGLES   = [-8, 6, -4, 7, -3, 5, -6, 4, -5, 8]


# ════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ════════════════════════════════════════════════════════════════════════════

def _img_hash(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.resize((32, 32), Image.NEAREST).save(buf, "JPEG", quality=30)
    return hashlib.md5(buf.getvalue()).hexdigest()


# ════════════════════════════════════════════════════════════════════════════
#  BLAZEFACE RUNNER
# ════════════════════════════════════════════════════════════════════════════

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
        return _blazeface_detector
    except ImportError:
        return None


def _run_blazeface(pil_img: Image.Image) -> list:
    import torch
    h = _img_hash(pil_img)
    if h in _face_cache:
        return _face_cache[h]

    orig_w, orig_h = pil_img.size
    boxes_out      = []

    detector = _get_blazeface()
    if detector is None:
        _face_cache[h] = boxes_out
        return boxes_out

    scale   = min(1.0, 640 / orig_w)
    det_w   = max(1, int(orig_w * scale))
    det_h   = max(1, int(orig_h * scale))
    det_img = pil_img.resize((det_w, det_h), Image.BILINEAR)

    with torch.no_grad():
        raw_boxes, _ = detector.detect(det_img)

    if raw_boxes is not None:
        inv = 1.0 / scale
        for box in raw_boxes:
            x1 = int(max(0,      box[0] * inv))
            y1 = int(max(0,      box[1] * inv))
            x2 = int(min(orig_w, box[2] * inv))
            y2 = int(min(orig_h, box[3] * inv))
            headroom = int((y2 - y1) * 0.25)
            y1       = max(0, y1 - headroom)
            boxes_out.append([float(x1), float(y1), float(x2), float(y2)])

    _face_cache[h] = boxes_out
    return boxes_out


# ════════════════════════════════════════════════════════════════════════════
#  U²-NET RUNNER
# ════════════════════════════════════════════════════════════════════════════

_u2net_model  = None
_u2net_device = None

def _run_u2net(pil_img: Image.Image) -> np.ndarray:
    """Run U²-Net saliency. Returns H×W float32 map (0-1). Cached by image hash."""
    global _u2net_model, _u2net_device

    h = _img_hash(pil_img)
    if h in _sal_map_cache:
        return _sal_map_cache[h]

    try:
        import torch
        import torch.nn.functional as F
        from torchvision import transforms
    except ImportError:
        result = _heuristic_saliency_map(pil_img)
        _sal_map_cache[h] = result
        return result

    import torch.nn as nn

    class REBNCONV(nn.Module):
        def __init__(self, in_ch=3, out_ch=3, dilate=1):
            super().__init__()
            self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=dilate, dilation=dilate)
            self.bn   = nn.BatchNorm2d(out_ch)
            self.relu = nn.ReLU(inplace=True)
        def forward(self, x): return self.relu(self.bn(self.conv(x)))

    class RSU4F(nn.Module):
        def __init__(self, in_ch, mid_ch, out_ch):
            super().__init__()
            self.in_  = REBNCONV(in_ch,   out_ch)
            self.c1   = REBNCONV(out_ch,  mid_ch, 1)
            self.c2   = REBNCONV(mid_ch,  mid_ch, 2)
            self.c3   = REBNCONV(mid_ch,  mid_ch, 4)
            self.c4   = REBNCONV(mid_ch,  mid_ch, 8)
            self.c3d  = REBNCONV(mid_ch*2, mid_ch, 4)
            self.c2d  = REBNCONV(mid_ch*2, mid_ch, 2)
            self.c1d  = REBNCONV(mid_ch*2, out_ch, 1)
        def forward(self, x):
            hx  = self.in_(x)
            h1  = self.c1(hx); h2 = self.c2(h1); h3 = self.c3(h2); h4 = self.c4(h3)
            h3d = self.c3d(torch.cat((h4, h3), 1))
            h2d = self.c2d(torch.cat((h3d, h2), 1))
            return self.c1d(torch.cat((h2d, h1), 1)) + hx

    class U2NETP(nn.Module):
        def __init__(self, in_ch=3, out_ch=1):
            super().__init__()
            self.s1  = RSU4F(in_ch, 16, 64); self.p12 = nn.MaxPool2d(2, 2, ceil_mode=True)
            self.s2  = RSU4F(64,    16, 64); self.p23 = nn.MaxPool2d(2, 2, ceil_mode=True)
            self.s3  = RSU4F(64,    16, 64); self.p34 = nn.MaxPool2d(2, 2, ceil_mode=True)
            self.s4  = RSU4F(64,    16, 64)
            self.s3d = RSU4F(128, 16, 64)
            self.s2d = RSU4F(128, 16, 64)
            self.s1d = RSU4F(128, 16, 64)
            self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
            self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
            self.side3 = nn.Conv2d(64, out_ch, 3, padding=1)
            self.side4 = nn.Conv2d(64, out_ch, 3, padding=1)
            self.out   = nn.Conv2d(4*out_ch, out_ch, 1)
        def forward(self, x):
            h1  = self.s1(x)
            h2  = self.s2(self.p12(h1))
            h3  = self.s3(self.p23(h2))
            h4  = self.s4(self.p34(h3))
            h3d = self.s3d(torch.cat((F.interpolate(h4,  h3.shape[2:], mode='bilinear', align_corners=False), h3), 1))
            h2d = self.s2d(torch.cat((F.interpolate(h3d, h2.shape[2:], mode='bilinear', align_corners=False), h2), 1))
            h1d = self.s1d(torch.cat((F.interpolate(h2d, h1.shape[2:], mode='bilinear', align_corners=False), h1), 1))
            d1  = self.side1(h1d)
            d2  = F.interpolate(self.side2(h2d), x.shape[2:], mode='bilinear', align_corners=False)
            d3  = F.interpolate(self.side3(h3d), x.shape[2:], mode='bilinear', align_corners=False)
            d4  = F.interpolate(self.side4(h4),  x.shape[2:], mode='bilinear', align_corners=False)
            return torch.sigmoid(self.out(torch.cat((d1, d2, d3, d4), 1))), torch.sigmoid(d1)

    weights_path = os.path.join(BASE_DIR, "u2netp.pth")
    if _u2net_model is None:
        if not os.path.exists(weights_path):
            import urllib.request
            url = "https://github.com/xuebinqin/U-2-Net/releases/download/v1.0/u2netp.pth"
            print("[U²-Net] Downloading weights (~4 MB)…")
            urllib.request.urlretrieve(url, weights_path)
            print("[U²-Net] Done.")
        _u2net_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = U2NETP(3, 1)
        net.load_state_dict(torch.load(weights_path, map_location=_u2net_device))
        net.to(_u2net_device).eval()
        _u2net_model = net

    orig_w, orig_h = pil_img.size
    max_edge = 800
    if max(orig_w, orig_h) > max_edge:
        scale    = max_edge / max(orig_w, orig_h)
        proc_img = pil_img.resize((int(orig_w * scale), int(orig_h * scale)), Image.BILINEAR)
    else:
        proc_img = pil_img

    tf = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    tensor = tf(proc_img.convert("RGB")).unsqueeze(0).to(_u2net_device)

    with torch.no_grad():
        sal, _ = _u2net_model(tensor)

    sal = sal.squeeze().cpu().numpy()
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    sal_pil = Image.fromarray((sal * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.BILINEAR)
    result  = np.array(sal_pil).astype(np.float32) / 255.0

    _sal_map_cache[h] = result
    return result


# ════════════════════════════════════════════════════════════════════════════
#  HEURISTIC SALIENCY  (fallback when torch unavailable)
# ════════════════════════════════════════════════════════════════════════════

_SAL_SIZE = 160

def _heuristic_saliency_map(pil_img: Image.Image) -> np.ndarray:
    orig_w, orig_h = pil_img.size
    thumb = pil_img.convert("RGB").resize((_SAL_SIZE, _SAL_SIZE), Image.BILINEAR)
    arr   = np.asarray(thumb, dtype=np.float32)
    H, W  = arr.shape[:2]
    Y, X  = np.mgrid[0:H, 0:W]

    gauss = np.exp(-((X - W/2)**2 / (2*(W*.40)**2) + (Y - H/2)**2 / (2*(H*.40)**2)))
    gray  = arr.mean(2)
    gy    = np.abs(np.diff(gray, axis=0, prepend=gray[:1]))
    gx    = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    edge  = np.hypot(gx, gy)
    edge /= edge.max() + 1e-6
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    cmax    = np.maximum(np.maximum(r, g), b)
    cmin    = np.minimum(np.minimum(r, g), b)
    sat     = (cmax - cmin) / (cmax + 1e-6) / 255.0
    sal     = gauss * .40 + edge * .35 + sat * .25
    sal    /= sal.max() + 1e-6

    sal_pil = Image.fromarray((sal * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.BILINEAR)
    return np.array(sal_pil).astype(np.float32) / 255.0


def _saliency_centroid(sal_map: np.ndarray) -> tuple[float, float]:
    """Centre-of-mass of saliency heatmap. Fallback (0.5, 0.45) if empty."""
    try:
        import cv2
        M  = cv2.moments(sal_map)
        if M["m00"] < 1e-6:
            return 0.5, 0.45
        cx = M["m10"] / M["m00"] / sal_map.shape[1]
        cy = M["m01"] / M["m00"] / sal_map.shape[0]
    except ImportError:
        H, W  = sal_map.shape
        Y, X  = np.mgrid[0:H, 0:W]
        total = sal_map.sum() + 1e-8
        cx    = float((X * sal_map).sum() / total / W)
        cy    = float((Y * sal_map).sum() / total / H)
    return float(np.clip(cx, 0.1, 0.9)), float(np.clip(cy, 0.1, 0.9))


# ════════════════════════════════════════════════════════════════════════════
#  CROP ANCHOR — FACE-HEAVY + SALIENCY FALLBACK
# ════════════════════════════════════════════════════════════════════════════

def get_saliency_roi(pil_img: Image.Image) -> tuple[float, float]:
    h = _img_hash(pil_img)
    if h in _saliency_cache:
        return _saliency_cache[h]

    orig_w, orig_h = pil_img.size

    try:
        boxes = _run_blazeface(pil_img)
    except Exception:
        boxes = []

    if boxes:
        avg_x1 = sum(b[0] for b in boxes) / len(boxes)
        avg_y1 = sum(b[1] for b in boxes) / len(boxes)
        avg_x2 = sum(b[2] for b in boxes) / len(boxes)
        avg_y2 = sum(b[3] for b in boxes) / len(boxes)

        cx = float(np.clip((avg_x1 + avg_x2) / 2 / orig_w, 0.05, 0.95))

        face_height_ratio = (avg_y2 - avg_y1) / orig_h
        face_top          = avg_y1 / orig_h
        face_mid          = (avg_y1 + avg_y2) / 2 / orig_h

        if face_height_ratio > 0.35:
            cy = face_mid
        elif face_height_ratio > 0.15:
            cy = face_top + face_height_ratio * 0.30
        else:
            cy = face_top

        cy = float(np.clip(cy - 0.08, 0.04, 0.90))

    else:
        try:
            sal_map = _run_u2net(pil_img)
        except Exception:
            sal_map = _heuristic_saliency_map(pil_img)

        cx, cy = _saliency_centroid(sal_map)
        cy     = float(np.clip(cy - 0.05, 0.08, 0.90))

    _saliency_cache[h] = (cx, cy)
    return cx, cy


# ════════════════════════════════════════════════════════════════════════════
#  PHOTO QUALITY SCORE
# ════════════════════════════════════════════════════════════════════════════

def get_photo_quality(pil_img: Image.Image) -> float:
    h = _img_hash(pil_img)
    if h in _quality_cache:
        return _quality_cache[h]

    orig_w, orig_h = pil_img.size
    img_area       = orig_w * orig_h

    try:
        boxes = _run_blazeface(pil_img)
    except Exception:
        boxes = []

    if boxes:
        face_area  = sum((b[2]-b[0]) * (b[3]-b[1]) for b in boxes)
        face_score = min(1.0, face_area / img_area * 4.0)
    else:
        face_score = 0.0

    try:
        sal_map   = _run_u2net(pil_img)
        sal_score = float(sal_map.mean())
    except Exception:
        sal_score = 0.3

    small = np.array(pil_img.convert("RGB").resize((200, 200)), dtype=np.float32)
    try:
        import cv2
        gray      = cv2.cvtColor(small.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        sharpness = min(1.0, float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 800.0)
    except ImportError:
        gray      = small.mean(2)
        gy        = np.abs(np.diff(gray, axis=0))
        gx        = np.abs(np.diff(gray, axis=1))
        sharpness = min(1.0, (gx.var() + gy.var()) * 5 / 800.0)

    score = 0.70 * face_score + 0.20 * sal_score + 0.10 * sharpness
    _quality_cache[h] = score
    return score


# ════════════════════════════════════════════════════════════════════════════
#  HUNGARIAN ALGORITHM ASSIGNMENT
# ════════════════════════════════════════════════════════════════════════════

def match_images_to_slots(pil_images, slots_def, canvas_w, canvas_h):
    n_slots  = len(slots_def)
    n_images = len(pil_images)
    n        = min(n_slots, n_images)
    if n == 0:
        return []

    img_ratios  = [img.size[0] / img.size[1] for img in pil_images[:n]]
    slot_ratios = []
    slot_areas  = []
    for (xf, yf, wf, hf) in slots_def[:n]:
        sw = max(1, wf * canvas_w)
        sh = max(1, hf * canvas_h)
        slot_ratios.append(sw / sh)
        slot_areas.append(sw * sh)

    max_area         = max(slot_areas) if slot_areas else 1.0
    slot_importances = [a / max_area for a in slot_areas]

    qualities = [get_photo_quality(img) for img in pil_images[:n]]
    has_face  = [bool(_run_blazeface(img)) for img in pil_images[:n]]

    cost_matrix = np.zeros((n, n), dtype=np.float64)

    for ii in range(n):
        for jj in range(n):
            ratio_diff = abs(img_ratios[ii] - slot_ratios[jj])

            if has_face[ii]:
                ratio_penalty  = ratio_diff ** 2 * 1.5
                quality_reward = qualities[ii] * slot_importances[jj] * 2.5
            else:
                ratio_penalty  = ratio_diff ** 2 * 3.0
                quality_reward = qualities[ii] * slot_importances[jj] * 1.5

            cost_matrix[ii][jj] = ratio_penalty - quality_reward

    try:
        from scipy.optimize import linear_sum_assignment
        photo_indices, slot_indices = linear_sum_assignment(cost_matrix)
        photo_indices = [int(i) for i in photo_indices]
        slot_indices  = [int(j) for j in slot_indices]
        order = [0] * n
        for photo_idx, slot_idx in zip(photo_indices, slot_indices):
            order[slot_idx] = photo_idx
        order = [int(x) for x in order]
        print(f"[Assignment] Hungarian optimal — total cost: "
              f"{cost_matrix[photo_indices, slot_indices].sum():.3f}")
        return order

    except ImportError:
        print("[Assignment] scipy not found — falling back to greedy.")
        slot_priority = sorted(range(n), key=lambda i: -slot_areas[i])
        used_imgs = set()
        order     = [0] * n
        for si in slot_priority:
            best_cost = float('inf')
            best_img  = -1
            for ii in range(n):
                if ii in used_imgs:
                    continue
                if cost_matrix[ii][si] < best_cost:
                    best_cost = cost_matrix[ii][si]
                    best_img  = ii
            if best_img == -1:
                best_img = next(i for i in range(n) if i not in used_imgs)
            order[si] = best_img
            used_imgs.add(best_img)
        order = [int(x) for x in order]
        return order


# ════════════════════════════════════════════════════════════════════════════
#
#  PLACEMENT FUNCTIONS
#  Two modes controlled by CROP_MODE at the top of this file.
#
# ════════════════════════════════════════════════════════════════════════════

# ── MODE: "fit" ───────────────────────────────────────────────────────────────
#
#  Photo fits entirely inside the slot — nothing is cut off at zoom=1.
#  Blurred version of the same photo fills any empty space behind it.
#  cx/cy from saliency positions the photo within the slot:
#    cx=0.5 → horizontally centred
#    cx=0.2 → shifted toward the left side
#  pan_x / pan_y (from user dragging) shift the photo further.
#  zoom > 1 scales the photo up; once larger than the slot, edges get cut.
#
def _place_fit(pil_img: Image.Image,
               target_w: int, target_h: int,
               cx: float, cy: float,
               zoom: float = 1.0,
               pan_x: float = 0.0,
               pan_y: float = 0.0) -> Image.Image:

    if target_w <= 0 or target_h <= 0:
        return pil_img.resize((max(1, target_w), max(1, target_h)), Image.BILINEAR)

    src_w, src_h = pil_img.size

    # Scale to fit inside (contain) then apply zoom
    base_scale = min(target_w / src_w, target_h / src_h)
    scale      = base_scale * zoom
    fg_w       = max(1, int(src_w * scale))
    fg_h       = max(1, int(src_h * scale))
    fg         = pil_img.resize((fg_w, fg_h), Image.BILINEAR)

    # Blurred background always fills the slot
    bg = pil_img.resize((target_w, target_h), Image.BILINEAR)
    bg = bg.filter(ImageFilter.GaussianBlur(radius=24))

    # Base position: cx/cy anchor within the remaining space after fit
    # e.g. cx=0.5 centres the photo horizontally
    space_x = target_w - fg_w
    space_y = target_h - fg_h
    base_x  = int(space_x * cx)
    base_y  = int(space_y * cy)

    # User pan: fraction of slot size, added on top of base position
    off_x = int(pan_x * target_w)
    off_y = int(pan_y * target_h)

    # Clamp so the photo never moves fully out of the slot
    px = max(min(base_x + off_x, target_w - 1), -(fg_w - 1))
    py = max(min(base_y + off_y, target_h - 1), -(fg_h - 1))

    bg.paste(fg, (px, py))
    return bg


# ── MODE: "crop" ──────────────────────────────────────────────────────────────
#
#  Original smart_crop: scales image to COVER the slot (no empty space),
#  then pans so cx/cy is centred in the visible window, cropping the edges.
#  Blur-bg kicks in only when aspect ratios differ by more than 0.5.
#
def _place_crop(pil_img: Image.Image,
                target_w: int, target_h: int,
                cx: float, cy: float,
                zoom: float = 1.0,
                pan_x: float = 0.0,
                pan_y: float = 0.0) -> Image.Image:

    if target_w <= 0 or target_h <= 0:
        return pil_img.resize((max(1, target_w), max(1, target_h)), Image.BILINEAR)

    src_w, src_h = pil_img.size

    # ── Check ratio diff — use blur-bg path if mismatch is large ──────────
    img_ratio  = src_w / src_h
    slot_ratio = target_w / target_h
    ratio_diff = abs(img_ratio - slot_ratio)

    if ratio_diff >= 0.5:
        # Large mismatch → blur-bg (fit-inside + blurred background)
        bg = pil_img.resize((target_w, target_h), Image.BILINEAR)
        bg = bg.filter(ImageFilter.GaussianBlur(radius=20))
        scale  = min(target_w / src_w, target_h / src_h)
        fg_w   = max(1, int(src_w * scale))
        fg_h   = max(1, int(src_h * scale))
        fg     = pil_img.resize((fg_w, fg_h), Image.BILINEAR)
        offset_x = (target_w - fg_w) // 2
        offset_y = (target_h - fg_h) // 2
        bg.paste(fg, (offset_x, offset_y))
        return bg

    # ── Small mismatch → cover + crop ────────────────────────────────────
    slot_w = max(4, int(target_w / zoom))
    slot_h = max(4, int(target_h / zoom))

    scale    = max(slot_w / src_w, slot_h / src_h)
    scaled_w = max(int(src_w * scale), slot_w)
    scaled_h = max(int(src_h * scale), slot_h)
    scaled   = pil_img.resize((scaled_w, scaled_h), Image.BILINEAR)

    salient_x = (cx + pan_x) * scaled_w
    salient_y = (cy + pan_y) * scaled_h

    is_tall_slot   = target_h > target_w * 1.25
    focus_in_upper = cy < 0.55

    if is_tall_slot and focus_in_upper:
        head_margin = slot_h * 0.25
        salient_y  -= (head_margin - slot_h / 2)
    elif focus_in_upper:
        head_margin = slot_h * 0.12
        salient_y  -= (head_margin - slot_h / 2)

    offset_x = salient_x - slot_w / 2
    offset_y = (0 if cy < 0.20
                else int(max(0, min(salient_y - slot_h / 2, scaled_h - slot_h))))
    offset_x = int(max(0, min(offset_x, scaled_w - slot_w)))

    view = scaled.crop((offset_x, offset_y,
                        offset_x + slot_w, offset_y + slot_h))
    if view.size != (target_w, target_h):
        view = view.resize((target_w, target_h), Image.BILINEAR)
    return view


# ── DISPATCHER — called everywhere instead of smart_crop/smart_crop_with_blur_bg
#
#  This is the only function build_collage needs to call.
#  It routes to _place_fit or _place_crop based on CROP_MODE.
#
def place_photo(pil_img: Image.Image,
                target_w: int, target_h: int,
                cx: float, cy: float,
                zoom: float = 1.0,
                pan_x: float = 0.0,
                pan_y: float = 0.0) -> Image.Image:

    if CROP_MODE == "fit":
        return _place_fit(pil_img, target_w, target_h, cx, cy, zoom, pan_x, pan_y)
    else:  # "crop"
        return _place_crop(pil_img, target_w, target_h, cx, cy, zoom, pan_x, pan_y)


# Keep old names as aliases so nothing else breaks if referenced elsewhere
def smart_crop(pil_img, target_w, target_h, cx, cy,
               zoom=1.0, pan_x=0.0, pan_y=0.0):
    return place_photo(pil_img, target_w, target_h, cx, cy, zoom, pan_x, pan_y)

def smart_crop_with_blur_bg(pil_img, target_w, target_h, cx, cy,
                             zoom=1.0, pan_x=0.0, pan_y=0.0):
    return place_photo(pil_img, target_w, target_h, cx, cy, zoom, pan_x, pan_y)


# ════════════════════════════════════════════════════════════════════════════
#  PASTE HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _mask_rounded(w: int, h: int, radius: int) -> Image.Image:
    from PIL import ImageDraw
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, w-1, h-1], radius=radius, fill=255)
    return mask


def paste_slot(canvas, img, x, y, w, h, corner=0):
    img = img.resize((w, h), Image.BILINEAR)
    if corner > 0:
        canvas.paste(img, (x, y), _mask_rounded(w, h, corner))
    else:
        canvas.paste(img, (x, y))


def paste_circle(canvas, img, x, y, w, h):
    from PIL import ImageDraw
    img  = img.resize((w, h), Image.BILINEAR)
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).ellipse([0, 0, w-1, h-1], fill=255)
    canvas.paste(img, (x, y), mask)


def paste_polaroid(canvas, photo, x, y, w, h, angle=0):
    border     = int(w * .06)
    bottom_pad = int(h * .18)
    pw, ph     = w - border*2, h - border - bottom_pad
    frame      = Image.new("RGB", (w, h), (252, 248, 240))
    # Polaroid always uses fit mode — we want the full photo in the frame
    inner = _place_fit(photo, pw, ph, 0.5, 0.5)
    frame.paste(inner, (border, border))
    if angle:
        frame = frame.rotate(angle, expand=True, resample=Image.BICUBIC,
                             fillcolor=(160, 148, 132))
    fw, fh = frame.size
    canvas.paste(frame, (x - fw//2 + w//2, y - fh//2 + h//2))


# ════════════════════════════════════════════════════════════════════════════
#  COLLAGE BUILDER
# ════════════════════════════════════════════════════════════════════════════

def decode_image(b64: str) -> Image.Image:
    raw = base64.b64decode(b64.split(",")[-1])
    return Image.open(io.BytesIO(raw)).convert("RGB")


def build_collage(images_b64, template_key, adjustments=None, assignment=None):
    t_start = time.perf_counter()
    tmpl    = COLLAGE_TEMPLATES[template_key]
    CW, CH  = tmpl["canvas_ratio"]
    gap     = tmpl.get("gap", 6)
    bg      = tmpl.get("bg", (255, 255, 255))
    corner  = tmpl.get("corner", 0)
    is_pol  = tmpl.get("polaroid", False)
    circ_sl = tmpl.get("circle_slot", -1)
    slots   = tmpl["slots_def"]

    pil_images = [decode_image(b64) for b64 in images_b64]

    if assignment is None:
        assignment = match_images_to_slots(pil_images, slots, CW, CH)

    canvas   = Image.new("RGB", (CW, CH), bg)
    roi_list = []

    for i, (xf, yf, wf, hf) in enumerate(slots):
        if i >= len(assignment):
            break
        img_idx = assignment[i]
        if img_idx >= len(pil_images):
            break
        pil = pil_images[img_idx]

        t0     = time.perf_counter()
        cx, cy = get_saliency_roi(pil)

        adj   = (adjustments[i] if adjustments and i < len(adjustments)
                 else {"zoom": 100, "panX": 0, "panY": 0})
        zoom  = max(1.0, adj.get("zoom",  100) / 100.0)
        pan_x = adj.get("panX", 0) / 100.0
        pan_y = adj.get("panY", 0) / 100.0

        sx = int(xf * CW) + gap // 2
        sy = int(yf * CH) + gap // 2
        sw = max(4, int(wf * CW) - gap)
        sh = max(4, int(hf * CH) - gap)

        if is_pol:
            paste_polaroid(canvas, pil, sx, sy, sw, sh,
                           angle=POLAROID_ANGLES[i % len(POLAROID_ANGLES)])
        elif i == circ_sl:
            size    = min(sw, sh)
            # Circle slot always uses fit so the full face is visible
            placed  = _place_fit(pil, size, size, cx, cy, zoom, pan_x, pan_y)
            ox, oy  = sx + (sw - size)//2, sy + (sh - size)//2
            paste_circle(canvas, placed, ox, oy, size, size)
        else:
            # ── Main path: routed by CROP_MODE ────────────────────────────
            placed = place_photo(pil, sw, sh, cx, cy, zoom, pan_x, pan_y)
            paste_slot(canvas, placed, sx, sy, sw, sh, corner)

        roi_list.append({
            "slot":    i + 1,
            "cx":      round(cx * 100, 1),
            "cy":      round(cy * 100, 1),
            "time_ms": round((time.perf_counter() - t0) * 1000, 1),
        })

    return canvas, roi_list, round(time.perf_counter() - t_start, 3), assignment


def collage_to_b64(canvas: Image.Image, quality: int = 92) -> str:
    buf = io.BytesIO()
    canvas.save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ════════════════════════════════════════════════════════════════════════════
#  DEBUG ROUTE
# ════════════════════════════════════════════════════════════════════════════

@app.route("/api/debug_roi", methods=["POST"])
def api_debug_roi():
    data = request.get_json(force=True)
    try:
        pil            = decode_image(data.get("image", ""))
        boxes          = _run_blazeface(pil)
        cx, cy         = get_saliency_roi(pil)
        quality        = get_photo_quality(pil)
        orig_w, orig_h = pil.size
        face_height_ratio = 0.0
        if boxes:
            avg_y1 = sum(b[1] for b in boxes) / len(boxes)
            avg_y2 = sum(b[3] for b in boxes) / len(boxes)
            face_height_ratio = (avg_y2 - avg_y1) / orig_h
        return jsonify({
            "faces_detected":    len(boxes),
            "boxes":             boxes,
            "cx":                round(cx, 4),
            "cy":                round(cy, 4),
            "quality":           round(quality, 4),
            "crop_mode":         CROP_MODE,
            "path":              "face-heavy" if boxes else "saliency-heavy",
            "photo_type":        ("close-up" if face_height_ratio > 0.35
                                  else "half-body" if face_height_ratio > 0.15
                                  else "full-body") if boxes else "no-face",
            "face_height_ratio": round(face_height_ratio, 3),
            "cy_pins_to_top":    cy < 0.20,
        })
    except Exception as e:
        return jsonify({"error": str(e)})


# ════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/templates")
def api_templates():
    return jsonify({
        k: {"name": v["name"], "slots": v["slots"], "preview_img": v["preview_img"],
            "circle_slot": v.get("circle_slot", -1)}
        for k, v in COLLAGE_TEMPLATES.items()
    })


@app.route("/api/slot_defs/<template_key>")
def api_slot_defs(template_key):
    if template_key not in COLLAGE_TEMPLATES:
        return jsonify({"error": "Unknown template"}), 400
    return jsonify([
        {"xf": s[0], "yf": s[1], "wf": s[2], "hf": s[3]}
        for s in COLLAGE_TEMPLATES[template_key]["slots_def"]
    ])


@app.route("/api/saliency", methods=["POST"])
def api_saliency():
    data = request.get_json(force=True)
    try:
        pil    = decode_image(data.get("image", ""))
        cx, cy = get_saliency_roi(pil)
        return jsonify({"cx": round(cx, 4), "cy": round(cy, 4)})
    except Exception as e:
        return jsonify({"cx": 0.5, "cy": 0.45, "error": str(e)})


@app.route("/api/saliency_batch", methods=["POST"])
def api_saliency_batch():
    from concurrent.futures import ThreadPoolExecutor
    data   = request.get_json(force=True)
    images = data.get("images", [])
    if not images:
        return jsonify({"results": []}), 200

    def process_one(b64):
        try:
            pil    = decode_image(b64)
            cx, cy = get_saliency_roi(pil)
            return {"cx": round(cx, 4), "cy": round(cy, 4)}
        except Exception:
            return {"cx": 0.5, "cy": 0.45}

    with ThreadPoolExecutor(max_workers=min(len(images), 4)) as ex:
        results = list(ex.map(process_one, images))

    return jsonify({"results": results}), 200


def _generate_response(data: dict) -> tuple:
    images_b64   = data.get("images", [])
    template_key = data.get("template", "modern_blocks")
    quality      = int(data.get("quality", 92))
    adjustments  = data.get("adjustments", None)

    if template_key not in COLLAGE_TEMPLATES:
        return {"error": f"Unknown template: {template_key}"}, 400

    needed = COLLAGE_TEMPLATES[template_key]["slots"]
    if len(images_b64) < needed:
        return {"error": f"Need {needed} images, got {len(images_b64)}"}, 400

    assignment = data.get("assignment", None)

    try:
        canvas, rois, elapsed, used_assignment = build_collage(
            images_b64[:needed], template_key, adjustments, assignment
        )
    except Exception as e:
        return {"error": str(e)}, 500

    CW, CH = COLLAGE_TEMPLATES[template_key]["canvas_ratio"]
    return {
        "collage":    collage_to_b64(canvas, quality),
        "rois":       rois,
        "time_s":     elapsed,
        "canvas":     f"{CW}×{CH}",
        "template":   COLLAGE_TEMPLATES[template_key]["name"],
        "slots":      needed,
        "assignment": used_assignment,
        "crop_mode":  CROP_MODE,
    }, 200


@app.route("/api/generate", methods=["POST"])
def api_generate():
    result, status = _generate_response(request.get_json(force=True))
    return jsonify(result), status


@app.route("/api/generate_adjusted", methods=["POST"])
def api_generate_adjusted():
    result, status = _generate_response(request.get_json(force=True))
    return jsonify(result), status


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  JIOPICS — AI Collage Studio")
    print(f"  Templates   : {len(COLLAGE_TEMPLATES)}")
    print(f"  Crop mode   : {CROP_MODE}")
    print(f"  Assignment  : Hungarian algorithm (globally optimal)")
    print(f"  URL         : http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=True, port=5000, host="0.0.0.0", use_reloader=False)