# 🛰️ Drone Image Change Detection

Detect land-use and land-cover (LULC) changes between two drone captures of the same area using deep learning-based semantic segmentation.

---

## 📌 Overview

This pipeline takes two georeferenced drone images captured at different dates and produces a change map highlighting regions where land cover has changed. It combines classical GIS preprocessing with a deep learning segmentation model (ResUNet) to generate per-pixel class predictions, which are then differenced to identify change.

---

## 🗂️ Project Structure

```
LULC_change_detection_DL/
├── datasets/
│   └── annotations/
│       ├── generate_patch_dataset.ipynb   # Patch extraction from raw GeoTIFFs
│       └── prepare_dataset.ipynb          # Dataset preparation & splits
│   └── manifest.json                      # Dataset manifest
├── final_model/
│   └── resunet_scripted.pt                # Exported TorchScript model
├── dataset.py                             # Dataset class & data loading
├── losses.py                              # Loss functions
├── metrics.py                             # IoU, confusion matrix metrics
├── model.py                               # ResUNet architecture definition
├── utils.py                               # Helper utilities
├── prepare_model_for_inferencing.py       # Export trained model to TorchScript
├── inference.py                           # Tile-based inference on GeoTIFF
├── train.ipynb                            # Training notebook
├── requirements.base.txt                  # Base dependencies
├── requirements.cpu.txt                   # CPU-only install
├── requirements.gpu.txt                   # GPU (CUDA) install
└── README.md
```

---

## ✅ Completed Steps

### 1. Image Preprocessing & Alignment
- **Image alignment & coregistration** — spatially aligns both date images so pixels correspond to the same ground location
- **Pixel resolution standardization** — resamples both images to a common GSD (ground sampling distance)
- **CRS correction** — reprojects all inputs to a unified coordinate reference system
- **Bounding box clipping** — clips both images to the same spatial extent, ensuring identical shape and pixel grid

### 2. Model Building
- Semantic segmentation model based on **ResUNet** architecture with attention gates
- Trained to classify land-cover classes per pixel (e.g. vegetation, bare soil, water, built-up)
- Exported as a **TorchScript** (`.pt`) model for portable, dependency-light inference

### 3. Inference Pipeline
- Tile-based inference with **edge padding** to handle images of arbitrary size
- Normalizes input using ImageNet statistics
- Outputs a single-band GeoTIFF prediction mask with class IDs
- Preserves original CRS and spatial metadata in the output raster

---

## ⚠️ Not Completed

| Task | Status |
|---|---|
| Proper model training (full dataset, hyperparameter tuning) | ❌ Not completed |
| Change detection (diff of two prediction masks) | ❌ Not completed |

> The current pipeline produces segmentation masks for individual images. Change detection by differencing two masks is the planned next step.

---

## 🚀 Usage

### 1. Clone the repository

```bash
git clone https://github.com/your-username/lulc-change-detection.git
cd lulc-change-detection
```

### 2. Install dependencies

```bash
# CPU only
pip install -r requirements.base.txt -r requirements.cpu.txt

# GPU (CUDA)
pip install -r requirements.base.txt -r requirements.gpu.txt
``` in `inference.py`

Open `inference.py` and set your input GeoTIFF path:

```python
INPUT_TIF  = r"path/to/your/image.tif"
OUTPUT_TIF = "prediction.tif"
```

### 4. Run inference

```bash
python inference.py
```

Output will be saved as `prediction.tif` — a single-band GeoTIFF with per-pixel class labels.

---

## 🔧 Requirements

```
torch, rasterio, numpy, tqdm, + others in requirements.base.txt
```

Install via:

```bash
# CPU only
pip install -r requirements.base.txt -r requirements.cpu.txt

# GPU (CUDA)
pip install -r requirements.base.txt -r requirements.gpu.txt
```

> CUDA-capable GPU recommended for large images. Falls back to CPU automatically.

---

## 📎 Notes

- Input image must be a **3-band RGB GeoTIFF**
- Model expects inputs normalized with ImageNet mean/std — preprocessing is handled inside `inference.py`
- Patch size is set to `512×512` by default; edge tiles are reflection-padded automatically