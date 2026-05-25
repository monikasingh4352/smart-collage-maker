# JIOPICS — AI Collage Studio

> Auto-positioning photo collage app powered by U²-Net saliency detection.  
> Built for Jio's engagement platform — targets users 35+.

---

## Project Structure

```
jiopics_v2/
├── backend/
│   └── app.py                  ← Flask server (all API routes + collage engine)
├── frontend/
│   ├── templates/
│   │   └── index.html          ← Main HTML (served by Flask)
│   └── static/
│       ├── css/
│       │   └── style.css       ← All styles
│       └── js/
│           └── app.js          ← Frontend logic
├── assets/
│   └── collage_templates/      ← Template preview images (12 templates)
│       ├── artspace.png
│       ├── black-blocks.jpg
│       ├── black-plus.jpg
│       ├── circle-center.jpg
│       ├── floating-boxes.jpg
│       ├── grey-photo-grid.jpg
│       ├── heart_collage.jpg
│       ├── simple-split.jpg
│       ├── storyart-film.jpg
│       ├── uneven-grid.jpg
│       ├── vintage-polaroid.jpg
│       └── 9_parts_style_3.png
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the server

```bash
cd backend
python app.py
```

### 3. Open in browser

```
http://127.0.0.1:5000
```

---

## Templates Available (12)

| Key               | Name              | Slots |
|-------------------|-------------------|-------|
| `artspace`        | Artspace Minimal  | 9     |
| `modern_blocks`   | Modern Blocks     | 5     |
| `plus_grid`       | Plus Grid         | 5     |
| `circle_focus`    | Circle Focus      | 6     |
| `floating_canvas` | Floating Canvas   | 3     |
| `dense_grid`      | Dense Grid        | 14    |
| `heart_collage`   | Heart Collage     | 12    |
| `simple_split`    | Simple Split      | 3     |
| `storyart_film`   | StoryArt Film     | 8     |
| `uneven_grid`     | Uneven Grid       | 6     |
| `vintage_polaroid`| Vintage Polaroid  | 5     |
| `nine_parts`      | 9 Parts Grid      | 9     |

---

## How AI Positioning Works

Each uploaded photo goes through:

1. **U²-Net Pocket Saliency** (if `u2netp.pth` present)  
   — detects the foreground subject (person, object) and returns a saliency map  
   — falls back to heuristic if model not available

2. **Heuristic Saliency** (always-available fallback)  
   — Gaussian center-bias (40%) + edge detection (35%) + color saturation (25%)  
   — Finds weighted centroid = ROI center (cx, cy)

3. **Smart Crop**  
   — Resizes image to fill the slot  
   — Anchors crop window around (cx, cy) so the subject is never cut off

4. **Render to Canvas**  
   — Pastes each cropped photo into its slot with correct dimensions  
   — Special handling for: polaroid frames, circular masks, gap/corner radius

---

## Upgrade to U²-Net AI (Optional)

For more accurate saliency on complex images:

```bash
# Download the ~4MB model
wget https://github.com/xuebinqin/U-2-Net/releases/download/u2netp/u2netp.pth

# Place it in the project root (next to backend/)
# Also place u2net_model.py in the root

pip install torch torchvision
```

The app auto-detects the model at startup.

---

## API Endpoints

| Method | Route               | Description                        |
|--------|---------------------|------------------------------------|
| GET    | `/`                 | Main web app                       |
| GET    | `/api/templates`    | List all templates + metadata      |
| POST   | `/api/generate`     | Build collage, returns base64 JPEG |
| POST   | `/api/saliency`     | Get ROI (cx, cy) for one image     |

### POST `/api/generate`

**Request:**
```json
{
  "template": "modern_blocks",
  "images": ["data:image/jpeg;base64,...", "..."],
  "quality": 92
}
```

**Response:**
```json
{
  "collage": "data:image/jpeg;base64,...",
  "rois": [
    { "slot": 1, "cx": 48.3, "cy": 52.1, "time_ms": 82.4 },
    ...
  ],
  "time_s": 0.643,
  "canvas": "1080×1440",
  "template": "Modern Blocks",
  "slots": 5
}
```

---

## Tech Stack

- **Backend:** Python 3.10+, Flask, Pillow, NumPy
- **AI:** U²-Net Pocket (optional), heuristic fallback (always-on)
- **Frontend:** Vanilla JS (ES6+), CSS custom properties, no framework
- **Fonts:** Sora + Space Mono (Google Fonts)

---

*Built as internship project at Jio — JIOPICS v2.0*
