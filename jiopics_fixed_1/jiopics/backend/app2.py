from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import base64, hashlib, io, os, time, traceback
from PIL import Image, ImageFilter
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
ROOT         = os.path.dirname(BASE_DIR)
FRONTEND_DIR = os.path.join(ROOT, "frontend")
ASSETS_DIR   = os.path.join(ROOT, "assets", "collage_templates")

app = Flask(__name__,
            template_folder=os.path.join(FRONTEND_DIR, "templates"),
            static_folder=os.path.join(FRONTEND_DIR, "static"))
CORS(app)

# ══════════════════════════════════════════════════════════════════════════════
#  FAST_MODE = True  → skips U²-Net + MTCNN (too slow on CPU)
#                      uses heuristic saliency (~5ms per image)
#  FAST_MODE = False → full AI (slow but best quality)
# ══════════════════════════════════════════════════════════════════════════════
FAST_MODE = True
print(f"[JIOPICS] Starting — FAST_MODE={FAST_MODE}")

# ── Caches ─────────────────────────────────────────────────────────────────────
_saliency_cache = {}
_quality_cache  = {}
_face_cache     = {}
_sal_map_cache  = {}

@app.route("/assets/collage_templates/<path:filename>")
def serve_asset(filename):
    return send_from_directory(ASSETS_DIR, filename)


# ════════════════════════════════════════════════════════════════════════════════
#  TEMPLATE LOADER
# ════════════════════════════════════════════════════════════════════════════════

def _load_templates():
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

def _img_hash(pil_img):
    buf = io.BytesIO()
    pil_img.resize((32, 32), Image.NEAREST).save(buf, "JPEG", quality=30)
    return hashlib.md5(buf.getvalue()).hexdigest()

def decode_image(b64):
    data = base64.b64decode(b64.split(",")[-1])
    return Image.open(io.BytesIO(data)).convert("RGB")


# ════════════════════════════════════════════════════════════════════════════════
#  HEURISTIC SALIENCY — fast, no GPU/model needed
# ════════════════════════════════════════════════════════════════════════════════

def _heuristic_saliency_map(pil_img):
    SZ   = 160
    ow, oh = pil_img.size
    arr  = np.asarray(
        pil_img.convert("RGB").resize((SZ, SZ), Image.BILINEAR),
        dtype=np.float32
    )
    H, W = arr.shape[:2]
    Y, X = np.mgrid[0:H, 0:W]
    gauss = np.exp(-((X-W/2)**2/(2*(W*.40)**2) + (Y-H/2)**2/(2*(H*.40)**2)))
    gray  = arr.mean(2)
    edge  = np.hypot(
        np.abs(np.diff(gray, axis=0, prepend=gray[:1])),
        np.abs(np.diff(gray, axis=1, prepend=gray[:,:1]))
    )
    edge /= edge.max() + 1e-6
    cmax = np.maximum(np.maximum(arr[:,:,0], arr[:,:,1]), arr[:,:,2])
    cmin = np.minimum(np.minimum(arr[:,:,0], arr[:,:,1]), arr[:,:,2])
    sat  = (cmax - cmin) / (cmax + 1e-6) / 255.0
    sal  = gauss*.40 + edge*.35 + sat*.25
    sal /= sal.max() + 1e-6
    result = np.array(
        Image.fromarray((sal*255).astype(np.uint8)).resize((ow,oh), Image.BILINEAR)
    ).astype(np.float32) / 255.0
    return result

def _saliency_centroid(sal_map):
    H, W  = sal_map.shape
    Y, X  = np.mgrid[0:H, 0:W]
    tot   = sal_map.sum() + 1e-8
    cx    = float(np.clip((X * sal_map).sum() / tot / W, 0.05, 0.95))
    cy    = float(np.clip((Y * sal_map).sum() / tot / H, 0.05, 0.95))
    return cx, cy


# ════════════════════════════════════════════════════════════════════════════════
#  SUBJECT ANCHOR — where is the interesting part of the photo?
# ════════════════════════════════════════════════════════════════════════════════

def get_saliency_roi(pil_img, slot_w=0, slot_h=0):
    """Returns (cx, cy) — subject position as fractions 0.0–1.0"""
    h = _img_hash(pil_img)
    if h in _saliency_cache:
        return _saliency_cache[h]

    try:
        sal_map    = _heuristic_saliency_map(pil_img)
        cx, cy     = _saliency_centroid(sal_map)
        cy         = float(np.clip(cy - 0.04, 0.08, 0.90))
    except Exception:
        cx, cy = 0.5, 0.45

    _saliency_cache[h] = (cx, cy)
    return cx, cy


# ════════════════════════════════════════════════════════════════════════════════
#  PHOTO QUALITY SCORE — used for slot assignment
# ════════════════════════════════════════════════════════════════════════════════

def get_photo_quality(pil_img):
    h = _img_hash(pil_img)
    if h in _quality_cache:
        return _quality_cache[h]
    try:
        small = np.array(pil_img.convert("L").resize((100,100)), dtype=np.float32)
        gy    = np.abs(np.diff(small, axis=0))
        gx    = np.abs(np.diff(small, axis=1))
        score = min(1.0, (gy.var() + gx.var()) * 5 / 800.0)
    except Exception:
        score = 0.5
    _quality_cache[h] = score
    return score


# ════════════════════════════════════════════════════════════════════════════════
#  SLOT ASSIGNMENT — which photo goes in which slot?
# ════════════════════════════════════════════════════════════════════════════════

def match_images_to_slots(pil_images, slots_def, canvas_w, canvas_h):
    n = min(len(slots_def), len(pil_images))
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
    qualities        = [get_photo_quality(img) for img in pil_images[:n]]

    cost = np.zeros((n, n), dtype=np.float64)
    for ii in range(n):
        for jj in range(n):
            rd = abs(img_ratios[ii] - slot_ratios[jj])
            cost[ii][jj] = rd**2 * 2.0 - qualities[ii] * slot_importances[jj] * 2.0

    try:
        from scipy.optimize import linear_sum_assignment
        pi, si = linear_sum_assignment(cost)
        order  = [0] * n
        for p, s in zip(pi, si):
            order[int(s)] = int(p)
        print(f"[Assignment] Hungarian cost: {cost[pi,si].sum():.3f}")
        return order
    except ImportError:
        # Simple greedy fallback — no scipy needed
        print("[Assignment] scipy not available — using greedy fallback")
        used  = set()
        order = [0] * n
        for si in sorted(range(n), key=lambda i: -slot_areas[i]):
            best = -1
            best_cost = float('inf')
            for ii in range(n):
                if ii not in used and cost[ii][si] < best_cost:
                    best_cost = cost[ii][si]
                    best = ii
            if best == -1:
                best = next(i for i in range(n) if i not in used)
            order[si] = best
            used.add(best)
        return order


# ════════════════════════════════════════════════════════════════════════════════
#  PLACEMENT — smart crop centred on subject
# ════════════════════════════════════════════════════════════════════════════════

def place_photo(pil_img, target_w, target_h, cx, cy,
                zoom=1.0, pan_x=0.0, pan_y=0.0):
    """
    Crops and resizes pil_img to exactly (target_w, target_h).
    Subject at (cx,cy) is centred in the output.
    Handles zoom and pan for user adjustments.
    """
    if target_w <= 0 or target_h <= 0:
        target_w = max(1, target_w)
        target_h = max(1, target_h)

    src_w, src_h = pil_img.size

    # ── Find crop rectangle matching slot aspect ratio ──────────────────────
    slot_ratio = target_w / target_h
    src_ratio  = src_w / src_h

    if abs(src_ratio - slot_ratio) < 0.01:
        # Already the right ratio — just resize
        return pil_img.resize((target_w, target_h), Image.BILINEAR)

    if src_ratio > slot_ratio:
        # Image wider than slot — crop width
        crop_h = src_h
        crop_w = int(src_h * slot_ratio)
    else:
        # Image taller than slot — crop height
        crop_w = src_w
        crop_h = int(src_w / slot_ratio)

    crop_w = max(1, crop_w)
    crop_h = max(1, crop_h)

    # ── Centre crop on subject (cx, cy) ────────────────────────────────────
    ideal_left = cx * src_w - crop_w / 2
    ideal_top  = cy * src_h - crop_h / 2

    left = int(max(0, min(ideal_left, src_w - crop_w)))
    top  = int(max(0, min(ideal_top,  src_h - crop_h)))

    cropped = pil_img.crop((left, top, left + crop_w, top + crop_h))

    # ── Apply user zoom + pan ───────────────────────────────────────────────
    if zoom > 1.0 or pan_x != 0.0 or pan_y != 0.0:
        cw, ch   = cropped.size
        zoomed_w = max(target_w, int(cw * zoom))
        zoomed_h = max(target_h, int(ch * zoom))
        zoomed   = cropped.resize((zoomed_w, zoomed_h), Image.BILINEAR)
        base_x   = (zoomed_w - target_w) // 2
        base_y   = (zoomed_h - target_h) // 2
        off_x    = int(pan_x * target_w)
        off_y    = int(pan_y * target_h)
        win_x    = max(0, min(base_x + off_x, zoomed_w - target_w))
        win_y    = max(0, min(base_y + off_y, zoomed_h - target_h))
        cropped  = zoomed.crop((win_x, win_y,
                                win_x + target_w, win_y + target_h))

    # ── Resize to exact slot size ───────────────────────────────────────────
    if cropped.size != (target_w, target_h):
        cropped = cropped.resize((target_w, target_h), Image.BILINEAR)

    return cropped


# ════════════════════════════════════════════════════════════════════════════════
#  PASTE HELPERS
# ════════════════════════════════════════════════════════════════════════════════

def _mask_rounded(w, h, r):
    from PIL import ImageDraw
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, w-1, h-1], radius=r, fill=255)
    return m

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
    border = int(w * .06)
    bot    = int(h * .18)
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


# ════════════════════════════════════════════════════════════════════════════════
#  COLLAGE BUILDER
# ════════════════════════════════════════════════════════════════════════════════

def build_collage(images_b64, template_key, adjustments=None, assignment=None):
    print(f"[Build] Starting — template={template_key}, images={len(images_b64)}")
    t0   = time.perf_counter()
    tmpl = COLLAGE_TEMPLATES[template_key]
    CW, CH  = tmpl["canvas_ratio"]
    gap     = tmpl.get("gap", 6)
    bg_col  = tmpl.get("bg", (255, 255, 255))
    corner  = tmpl.get("corner", 0)
    is_pol  = tmpl.get("polaroid", False)
    circ_sl = tmpl.get("circle_slot", -1)
    slots   = tmpl["slots_def"]

    print(f"[Build] Canvas={CW}x{CH}, slots={len(slots)}, gap={gap}")

    # Decode images
    pil_imgs = []
    for idx, b in enumerate(images_b64):
        try:
            img = decode_image(b)
            pil_imgs.append(img)
            print(f"[Build] Decoded image {idx+1}: {img.size}")
        except Exception as e:
            print(f"[Build] ERROR decoding image {idx+1}: {e}")
            raise

    # Slot assignment
    if assignment is None:
        print("[Build] Running slot assignment...")
        assignment = match_images_to_slots(pil_imgs, slots, CW, CH)
        print(f"[Build] Assignment: {assignment}")

    canvas   = Image.new("RGB", (CW, CH), bg_col)
    roi_list = []

    for i, (xf, yf, wf, hf) in enumerate(slots):
        if i >= len(assignment):
            break
        idx = assignment[i]
        if idx >= len(pil_imgs):
            break

        pil = pil_imgs[idx]
        ts  = time.perf_counter()

        sx = int(xf * CW) + gap // 2
        sy = int(yf * CH) + gap // 2
        sw = max(4, int(wf * CW) - gap)
        sh = max(4, int(hf * CH) - gap)

        print(f"[Build] Slot {i+1}: img={idx}, pos=({sx},{sy}), size=({sw},{sh})")

        cx, cy = get_saliency_roi(pil, sw, sh)

        adj   = adjustments[i] if adjustments and i < len(adjustments) \
                else {"zoom": 100, "panX": 0, "panY": 0}
        zoom  = max(1.0, adj.get("zoom",  100) / 100.0)
        pan_x = adj.get("panX", 0) / 100.0
        pan_y = adj.get("panY", 0) / 100.0

        try:
            if is_pol:
                paste_polaroid(canvas, pil, sx, sy, sw, sh,
                               angle=POLAROID_ANGLES[i % len(POLAROID_ANGLES)])
            elif i == circ_sl:
                size   = min(sw, sh)
                placed = place_photo(pil, size, size, cx, cy, zoom, pan_x, pan_y)
                paste_circle(canvas, placed,
                             sx + (sw-size)//2, sy + (sh-size)//2, size, size)
            else:
                placed = place_photo(pil, sw, sh, cx, cy, zoom, pan_x, pan_y)
                paste_slot(canvas, placed, sx, sy, sw, sh, corner)
            print(f"[Build] Slot {i+1} done in {(time.perf_counter()-ts)*1000:.0f}ms")
        except Exception as e:
            print(f"[Build] ERROR placing slot {i+1}: {e}")
            traceback.print_exc()
            raise

        roi_list.append({
            "slot":    i + 1,
            "cx":      round(cx * 100, 1),
            "cy":      round(cy * 100, 1),
            "time_ms": round((time.perf_counter() - ts) * 1000, 1),
        })

    elapsed = round(time.perf_counter() - t0, 3)
    print(f"[Build] Complete in {elapsed}s")
    return canvas, roi_list, elapsed, assignment


def collage_to_b64(canvas, quality=92):
    buf = io.BytesIO()
    canvas.save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ════════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ════════════════════════════════════════════════════════════════════════════════

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
            "circle_slot": v.get("circle_slot", -1)
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
    from concurrent.futures import ThreadPoolExecutor
    imgs = request.get_json(force=True).get("images", [])
    if not imgs:
        return jsonify({"results": []}), 200
    def one(b64):
        try:
            p = decode_image(b64)
            cx, cy = get_saliency_roi(p)
            return {"cx": round(cx, 4), "cy": round(cy, 4)}
        except Exception:
            return {"cx": 0.5, "cy": 0.45}
    with ThreadPoolExecutor(max_workers=min(len(imgs), 4)) as ex:
        results = list(ex.map(one, imgs))
    return jsonify({"results": results}), 200

def _gen(data):
    imgs  = data.get("images", [])
    tk    = data.get("template", "modern_blocks")
    qual  = int(data.get("quality", 92))
    adjs  = data.get("adjustments", None)
    asgn  = data.get("assignment", None)

    print(f"\n[Generate] template={tk}, images={len(imgs)}, quality={qual}")

    if tk not in COLLAGE_TEMPLATES:
        print(f"[Generate] ERROR: Unknown template '{tk}'")
        return {"error": f"Unknown template: {tk}"}, 400

    needed = COLLAGE_TEMPLATES[tk]["slots"]
    if len(imgs) < needed:
        print(f"[Generate] ERROR: Need {needed} images, got {len(imgs)}")
        return {"error": f"Need {needed} images, got {len(imgs)}"}, 400

    try:
        canvas, rois, elapsed, used = build_collage(
            imgs[:needed], tk, adjs, asgn
        )
    except Exception as e:
        print(f"[Generate] FATAL ERROR: {e}")
        traceback.print_exc()
        return {"error": str(e)}, 500

    CW, CH = COLLAGE_TEMPLATES[tk]["canvas_ratio"]
    print(f"[Generate] Success — {elapsed}s")
    return {
        "collage":    collage_to_b64(canvas, qual),
        "rois":       rois,
        "time_s":     elapsed,
        "canvas":     f"{CW}×{CH}",
        "template":   COLLAGE_TEMPLATES[tk]["name"],
        "slots":      needed,
        "assignment": used,
        "placement":  "smart_crop",
    }, 200

@app.route("/api/generate", methods=["POST"])
def api_generate():
    r, s = _gen(request.get_json(force=True))
    return jsonify(r), s

@app.route("/api/generate_adjusted", methods=["POST"])
def api_generate_adjusted():
    r, s = _gen(request.get_json(force=True))
    return jsonify(r), s

@app.route("/api/debug_roi", methods=["POST"])
def api_debug_roi():
    d = request.get_json(force=True)
    try:
        pil    = decode_image(d.get("image", ""))
        cx, cy = get_saliency_roi(pil)
        q      = get_photo_quality(pil)
        return jsonify({
            "cx":      round(cx, 4),
            "cy":      round(cy, 4),
            "quality": round(q, 4),
            "fast_mode": FAST_MODE,
        })
    except Exception as e:
        return jsonify({"error": str(e)})


# ════════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  JIOPICS  —  AI Collage Studio")
    print(f"  Templates  : {len(COLLAGE_TEMPLATES)}")
    print(f"  Fast mode  : {FAST_MODE}")
    print(f"  URL        : http://127.0.0.1:5000")
    print("=" * 60)
    app.run(
        debug=False,
        port=5000,
        host="0.0.0.0",
        use_reloader=False,
        threaded=True
    )
