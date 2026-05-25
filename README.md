# JioPics — AI-Powered Automatic Collage Maker

> Built during internship at **Reliance Retail Limited**

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-black?style=flat-square&logo=flask)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red?style=flat-square&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## What is JioPics?

JioPics is an AI-powered collage maker that automatically selects, crops, and arranges your photos into a beautiful collage — without any manual effort.

Upload your photos, pick a layout, and the AI handles everything: detecting faces, finding the interesting part of each image, scoring photo quality, and placing every photo in its most aesthetically fitting slot.

---

## Demo

| Step | Action |
|---|---|
| 1 | Choose a collage template |
| 2 | Upload your photos |
| 3 | Click **Generate Collage** |
| 4 | Fine-tune with drag, zoom, and swap |
| 5 | Download your collage |

---

## How the AI Works

```
Upload Photos
      ↓
Face Detection (MTCNN)
      ↓
Saliency Analysis (U²-Net)
      ↓
Quality Scoring
      ↓
Hungarian Algorithm — Slot Assignment
      ↓
Crop-First Placement (Rule of Thirds)
      ↓
Final Collage
```

- **Face Detection (MTCNN)** — Detects all faces and ensures subjects are never cropped out, with automatic headroom
- **Saliency Analysis (U²-Net)** — Generates a heatmap of the most visually interesting region in non-portrait photos
- **Quality Scoring** — Ranks each photo by face prominence, visual saliency, and sharpness
- **Hungarian Algorithm** — Mathematically assigns the best photo to the most prominent slot
- **Crop-First Placement** — Applies Rule of Thirds composition for professional-looking results

---

## Project Structure

```
jiopics/
├── backend/
│   ├── app.py              # Flask backend — all AI logic lives here
│   ├── templates.json      # Collage template definitions
│   └── u2netp.pth          # U²-Net weights (auto-downloaded on first run)
├── frontend/
│   ├── templates/
│   │   └── index.html      # Single-page frontend UI
│   └── static/
│       └── js/             # JavaScript (embedded in index.html)
├── assets/
│   └── collage_templates/  # Template preview images
├── requirements.txt
└── run.sh
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/jiopics.git
cd jiopics
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install PyTorch (CPU)

**Windows / Linux:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Mac (Apple Silicon M1/M2/M3):**
```bash
pip install torch torchvision
```

### 4. Run the app

```bash
cd backend
python app.py
```

Open your browser at: **http://127.0.0.1:5000**

---

## Requirements

```
flask
flask-cors
Pillow
numpy
torch
torchvision
facenet-pytorch
scipy
werkzeug
```

> U²-Net weights (~4 MB) are downloaded automatically on the first run.

---

## Features

- 14 collage templates — grids, polaroids, circles, and more
- Smart face-aware cropping — heads are never cut off
- Automatic slot assignment — best photo always gets the biggest slot
- User adjustments — drag to pan, scroll to zoom, click to swap slots
- Download collage as high-quality JPEG
- Fast mode — heuristic saliency for instant results on CPU

---

## Fast Mode vs Full AI Mode

| Mode | Speed | Quality | Use Case |
|---|---|---|---|
| `FAST_MODE = True` | 2–5 seconds | Good | Demo, low-end hardware |
| `FAST_MODE = False` | 30–60 seconds | Best | Production, GPU available |

To switch modes, open `backend/app.py` and change line 22:

```python
FAST_MODE = True   # ← change to False for full AI
```

---

## Future Work

An intelligent **photo selection pipeline** is planned as the next phase:

1. **CLIP Embeddings** — semantic understanding of each photo
2. **Hard Deduplication** — removes near-identical shots automatically
3. **Event Clustering** — groups photos by EXIF timestamp or visual similarity
4. **MMR Selection** — Maximal Marginal Relevance picks the most diverse, high-quality set from up to 50 uploaded photos

---

## Built With

| Technology | Purpose |
|---|---|
| Flask | Backend API server |
| Pillow | Image processing |
| PyTorch | Neural network inference |
| MTCNN (facenet-pytorch) | Face detection |
| U²-Net | Saliency detection |
| SciPy | Hungarian algorithm |
| NumPy | Matrix operations |
| HTML / CSS / JavaScript | Frontend UI |

---

## Internship Context

This project was built as part of an internship at **Reliance Retail Limited** over 3 weeks as a problem statement assignment. The previous problem statement had been completed before this was assigned.

---

## License

MIT License — free to use, modify, and distribute.

---

*Made with ❤️ at Reliance Retail Limited*
