# Facial Region Cropper

This tool automatically extracts the **forehead** and the **visible cheek** (left or right) from facial images using **MediaPipe FaceMesh**.  
It is designed for dermatology and acne analysis where accurate region localization is required.

---

## 1. Installation

Install required Python packages:

```bash
pip install mediapipe==0.10.14 opencv-python==4.9.0.80 numpy==1.24.4
 #or
pip install opencv-python mediapipe numpy
```

## 2. Folder Structure

Organize your project directory like this:

```bash
project/
│
├── crop_regions.py        # the Python script
├── images_in/             # put your raw face images here
│     ├── img1.jpg
│     ├── img2.jpg
│     └── ...
│
└── crops_out/             # auto-created output folder
```

- images_in/ → Input folder with face images
- crops_out/ → Output folder for cropped regions

## 3. Input Image Requirements
- Images must show one side of the face (left-face or right-face).
- Higher-resolution images work best.
- Supported formats: .jpg, .jpeg, .png.

## 4. Running the Script
Run the script from your terminal:

```bash
python crop_regions.py
```

After processing, the results will appear in:
```bash
crops_out/
│
├── img1_forehead.png
├── img1_cheek_left.png
├── img2_forehead.png
├── img2_cheek_right.png
└── ...
```

The script automatically:
- Detects 468 FaceMesh landmarks
- Crops the forehead
- Crops left cheek and right cheek
- Chooses the cheek that is more visible (larger region)
- Saves the crops

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