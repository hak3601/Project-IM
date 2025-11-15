# Facial Region Cropper

This project extracts **forehead** and **cheek** regions from acne images using MediaPipe FaceMesh (468 facial landmarks).
It supports two image types:
- One-sided face images (left or right profile)
- Full-face images (frontal view)

Two separate scripts are provided to handle these cases.

---

## 1. Installation

Python version required: **Python 3.8 – 3.10**
(MediaPipe does NOT support 3.11 or 3.12)

Install dependencies:

```bash
pip install mediapipe==0.10.14 opencv-python==4.9.0.80 numpy==1.24.4
 #or
pip install opencv-python mediapipe numpy
```

## 2. Folder Structure

Organize your project directory like this:

```bash
hak/
│
├── crop_regions_full.py         # for frontal full-face images
├── crop_regions_onesided.py     # for one-sided (profile) images
│
├── images_in/                   # place raw images here
│     ├── img1.jpg
│     ├── img2.jpg
│     └── ...
│
├── crops_out/                   # output crops will be saved here
│
└── readme.md

```

- images_in/ → Input folder with face images
- crops_out/ → Output folder for cropped regions

## 3. How It Works
### Full-face mode (crop_regions_full.py)
- Designed for frontal images where both cheeks and the forehead are visible.
- Script crops and saves:
    ```bash
    *_forehead.png
    *_left_cheek.png
    *_right_cheek.png
    ```

### One-sided mode (crop_regions_onesided.py)
- Designed for profile images where only one cheek is visible.
- Script computes both cheeks but selects the dominant (larger) cheek automatically.
- Saves:
    ```bash
    *_forehead.png
    *_cheek_visible_left.png (or) *_cheek_visible_right.png
    ```

Both scripts internally use MediaPipe FaceMesh for robust landmark-based region extraction.

## 4. Usage
### A) Process ALL images in images_in/
Full-face mode:
    ```
    python crop_regions_full.py
    ```

One-sided mode:
    ```
    python crop_regions_onesided.py
    ```

### B) Process a SINGLE image
Full-face example: ```python crop_regions_full.py images_in/acne_front.jpg```
    
One-sided example: ```python crop_regions_onesided.py images_in/acne_left.jpg```


## 5. Customization
Inside crop_regions.py, you can modify:

| Variable           | Meaning                                          |
| ------------------ | ------------------------------------------------ |
| `INPUT_DIR`        | Where input images are read from                 |
| `OUTPUT_DIR`       | Where cropped images will be saved               |
| `PADDING`          | Extra padding around each region (default: 0.06) |
| `LEFT_CHEEK_IDXS`  | Landmark indices for subject’s left cheek        |
| `RIGHT_CHEEK_IDXS` | Landmark indices for subject’s right cheek       |
| `FOREHEAD_IDXS`    | Landmark indices for forehead region             |

## 6. Troubleshooting
### No face detected
- Ensure the image is clear, bright, and shows enough facial area.
- Make sure the face is not too far or too close.

### Crops look too small or too large
Adjust padding:

```bash
PADDING = 0.06  # increase for larger regions, decrease for tighter crops
```

### Want both cheeks saved always
Remove the cheek-selection logic in crop_regions.py:

```bash
# Remove area comparison and save both left_cheek and right_cheek
```

## 7. Notes
- MediaPipe FaceMesh automatically adapts to head pose and works reliably on both frontal and profile acne images.
- Badly lit, extremely occluded, or extreme-angle images may fail landmark detection.