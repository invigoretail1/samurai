# SAMURAI Video Annotation Workflow

## Overview

SAMURAI (Segment Anything Model for Unified Real-time Image Annotation) is a video object segmentation tool used to generate training data for YOLO models. This workflow takes raw product videos, tracks objects across frames, and outputs YOLO-formatted annotations for model training.

**Repository Location:** `/home/bel/MLP/Payment/samurai`

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SAMURAI ANNOTATION PIPELINE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────┐                                                   │
│   │   Video Capture     │  Record product videos (object in first frame)   │
│   │   (25-30 fps)       │                                                   │
│   └──────────┬──────────┘                                                   │
│              ▼                                                              │
│   ┌─────────────────────┐                                                   │
│   │  convert_to_folder  │  Extract frames from videos                       │
│   │        .py          │                                                   │
│   └──────────┬──────────┘                                                   │
│              ▼                                                              │
│   ┌─────────────────────┐                                                   │
│   │   LabelMe           │  Annotate FIRST FRAME ONLY (bounding box)        │
│   │   Annotation        │                                                   │
│   └──────────┬──────────┘                                                   │
│              ▼                                                              │
│   ┌─────────────────────┐                                                   │
│   │  prep_for_samurai   │  Convert to LaSOT format                          │
│   │        .py          │                                                   │
│   └──────────┬──────────┘                                                   │
│              ▼                                                              │
│   ┌─────────────────────┐                                                   │
│   │  SAMURAI Inference  │  Propagate annotations across all frames          │
│   │  main_inference.py  │                                                   │
│   └──────────┬──────────┘                                                   │
│              ▼                                                              │
│   ┌─────────────────────┐                                                   │
│   │  Manual Cleaning    │  Review & remove bad frames                       │
│   └──────────┬──────────┘                                                   │
│              ▼                                                              │
│   ┌─────────────────────┐                                                   │
│   │   sync_folders.py   │  Sync cleaned frames with originals               │
│   └──────────┬──────────┘                                                   │
│              ▼                                                              │
│   ┌─────────────────────┐                                                   │
│   │ yolo_convert_extend │  Convert to YOLO training format                  │
│   │        .py          │                                                   │
│   └──────────┬──────────┘                                                   │
│              ▼                                                              │
│   ┌─────────────────────┐                                                   │
│   │   YOLO Dataset      │  Ready for training                               │
│   │   (images + labels) │                                                   │
│   └─────────────────────┘                                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Video Capture

Record videos of products for annotation. Quality at this stage directly impacts annotation accuracy.

### Requirements

| Requirement | Specification |
|-------------|---------------|
| Frame Rate | 25-30 fps |
| Duration | ~1 minute per video |
| First Frame | Object of interest MUST be visible |
| Occlusions | Avoid if possible (see note below) |

### Naming Convention

Use descriptive names that identify the product and camera:
```
<ProductName>_<CameraID>.mp4
```

**Examples:**
- `Robitussin_cam0.mp4`
- `Tylenol_cam1.mp4`
- `Advil_Cold_cam0.mp4`

### Best Practices

- Ensure consistent lighting throughout the video
- Keep the product fully visible in the first frame (this frame will be annotated)
- Minimize motion blur during object movement
- If occlusions are unavoidable, note the frame numbers for later (SAMURAI supports `full_occlusion.txt`)

---

## Step 2: Frame Extraction

Convert videos into individual frames for annotation.

**Script:** `samurai_prep_scripts/convert_to_folder.py`

```bash
python /home/bel/MLP/Payment/samurai/samurai_prep_scripts/convert_to_folder.py \
  --input_dir /path/to/videos \
  --output_dir /path/to/frames
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `--input_dir` | Directory containing input video files |
| `--output_dir` | Directory where frame folders will be created |

**Output Structure:**
```
output_dir/
├── Robitussin_cam0/
│   ├── 00001.jpg
│   ├── 00002.jpg
│   └── ...
├── Tylenol_cam1/
│   ├── 00001.jpg
│   └── ...
└── ...
```

---

## Step 3: First Frame Annotation

Annotate only the first frame of each video using LabelMe. SAMURAI will propagate these annotations to subsequent frames.

### Launch LabelMe

```bash
labelme
```

### Annotation Process

1. Open the first frame (`00001.jpg`) from each product folder
2. Draw a bounding box around the object of interest
3. Label with the product name (must match folder naming)
4. Save the annotation (creates a `.json` file alongside the image)

### Tips

- Draw tight bounding boxes that closely fit the object
- Be consistent with label names across all annotations
- Double-check that annotations are saved before closing

---

## Step 4: Prepare for SAMURAI

Convert the annotated frames into the LaSOT format required by SAMURAI.

**Script:** `samurai_prep_scripts/prep_for_samurai.py`

```bash
python /home/bel/MLP/Payment/samurai/samurai_prep_scripts/prep_for_samurai.py \
  --frames_dir /path/to/annotated/frames \
  --videos_dir /path/to/original/videos
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `--frames_dir` | Directory containing extracted frames with LabelMe annotations |
| `--videos_dir` | Directory containing original video files |

**Output Structure (LaSOT Format):**
```
data/LaSOT/
├── product_category/
│   └── product_name-1/
│       ├── img/
│       │   ├── 00001.jpg
│       │   ├── 00002.jpg
│       │   └── ...
│       ├── groundtruth.txt      # Initial bounding box coordinates
│       ├── full_occlusion.txt   # Frame numbers with full occlusion
│       ├── out_of_view.txt      # Frame numbers where object leaves view
│       └── nlp.txt              # Natural language description
├── training_set.txt
└── testing_set.txt
```

---

## Step 5: Generate Dataset Split Files

Create the training and testing set configuration files. You can copy the test_train script into the LaSOT folder and run it there to generate the files

**Script:** `samurai_prep_scripts/test_train.py`

```bash
cd /home/bel/MLP/Payment/samurai/data/LaSOT
python ../../samurai_prep_scripts/test_train.py
```

This generates:
- `training_set.txt` — List of sequences for training
- `testing_set.txt` — List of sequences for testing/inference

---

## Step 6: Run SAMURAI Inference

Execute SAMURAI to propagate annotations across all video frames.

### Activate Environment

```bash
mamba activate CanSAM-env
```

### Run Inference

```bash
cd /home/bel/MLP/Payment/samurai
python scripts/main_inference.py
```

### Output Locations

| Output | Location | Description |
|--------|----------|-------------|
| Annotated Videos | `visualization/` | Videos with bounding box overlays |
| Annotations | `results/` | Per-frame bounding box coordinates |

---

## Step 7: Quality Review and Cleaning

Review SAMURAI output and remove frames with poor tracking.

### Convert Output Videos to Frames

```bash
python /home/bel/MLP/Payment/samurai/samurai_prep_scripts/convert_to_folder.py \
  --input_dir /path/to/visualization \
  --output_dir /path/to/annotated_frames
```

### Manual Cleaning Criteria

Delete frames where:
- The object is not recognizable by human eye
- Tracking has drifted to wrong object
- Bounding box is significantly misaligned
- Object is completely occluded
- Severe motion blur obscures the object

### Cleaning Process

1. Open each product folder in an image viewer
2. Review frames sequentially
3. Delete any frames meeting the above criteria
4. Keep notes of which products had significant issues

---

## Step 8: Synchronize Frames

Sync the cleaned annotated frames with the original (non-annotated) frames.

**Script:** `train_scripts/sync_folders.py`

```bash
python /home/bel/MLP/Payment/train_scripts/sync_folders.py \
  --raw_root /path/to/original/frames \
  --clean_root /path/to/cleaned/annotated/frames \
  --output_root /path/to/synced/output
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `--raw_root` | Directory with original extracted frames (pre-SAMURAI) |
| `--clean_root` | Directory with cleaned frames (post-SAMURAI review) |
| `--output_root` | Directory for synchronized output |

This ensures that only frames which passed quality review are included, matched with their original (non-annotated) versions.

---

## Step 9: Convert to YOLO Format

Transform the synchronized data into YOLO training format.

**Script:** `train_scripts/yolo_convert_extend.py`

```bash
python /home/bel/MLP/Payment/train_scripts/yolo_convert_extend.py \
  --txt-dir /path/to/samurai/results \
  --frames-root /path/to/synced/frames \
  --out /path/to/yolo/dataset \
  --class-mode smart
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `--txt-dir` | Directory containing SAMURAI annotation outputs |
| `--frames-root` | Directory with synchronized frames |
| `--out` | Output directory for YOLO dataset |
| `--class-mode` | Class mapping mode (`smart` recommended) |

### Merging Multiple Datasets

The same script can merge multiple YOLO datasets:

```bash
python yolo_convert_extend.py \
  --txt-dir /path/to/new/annotations \
  --frames-root /path/to/new/frames \
  --out /path/to/existing/yolo/dataset \
  --class-mode smart
```

### YOLO Dataset Structure

```
yolo_dataset/
├── images/
│   ├── train/
│   │   ├── Robitussin_cam0_00001.jpg
│   │   └── ...
│   └── val/
│       └── ...
├── labels/
│   ├── train/
│   │   ├── Robitussin_cam0_00001.txt
│   │   └── ...
│   └── val/
│       └── ...
└── data.yaml
```

---

## Quick Reference

### Complete Workflow Commands

```bash
# 1. Extract frames from videos
python samurai_prep_scripts/convert_to_folder.py \
  --input_dir ./videos --output_dir ./frames

# 2. Annotate first frames with LabelMe
labelme

# 3. Prepare for SAMURAI
python samurai_prep_scripts/prep_for_samurai.py \
  --frames_dir ./frames --videos_dir ./videos

# 4. Generate train/test splits
cd data/LaSOT && python ../../samurai_prep_scripts/test_train.py

# 5. Run SAMURAI inference
mamba activate CanSAM-env
python scripts/main_inference.py

# 6. Convert output videos to frames for review
python samurai_prep_scripts/convert_to_folder.py \
  --input_dir ./visualization --output_dir ./annotated_frames

# 7. [MANUAL] Clean frames in annotated_frames/

# 8. Sync cleaned frames with originals
python train_scripts/sync_folders.py \
  --raw_root ./frames --clean_root ./annotated_frames --output_root ./synced

# 9. Convert to YOLO format
python train_scripts/yolo_convert_extend.py \
  --txt-dir ./results --frames-root ./synced --out ./yolo_dataset --class-mode smart
```

---

## Troubleshooting

### SAMURAI Tracking Drift

**Problem:** Bounding box drifts to wrong object mid-video.

**Solutions:**
- Ensure first-frame annotation is tight and accurate
- Record shorter videos with less object movement
- Add the problematic frames to `full_occlusion.txt`

### Missing Annotations After Sync

**Problem:** Some frames missing after synchronization.

**Solutions:**
- Verify folder names match between raw and cleaned directories
- Check that frame numbering is consistent (e.g., `00001.jpg` format)

### Class Mismatch in YOLO Output

**Problem:** Classes not mapping correctly.

**Solutions:**
- Use `--class-mode smart` for automatic handling
- Verify LabelMe labels match expected class names
- Check `data.yaml` in output for class mapping

---

## File Reference

| Script | Location | Purpose |
|--------|----------|---------|
| `convert_to_folder.py` | `samurai_prep_scripts/` | Extract frames from videos |
| `prep_for_samurai.py` | `samurai_prep_scripts/` | Convert to LaSOT format |
| `test_train.py` | `samurai_prep_scripts/` | Generate dataset splits |
| `main_inference.py` | `scripts/` | Run SAMURAI tracking |
| `sync_folders.py` | `train_scripts/` | Sync cleaned frames |
| `yolo_convert_extend.py` | `train_scripts/` | Convert to YOLO format |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | January 2026 | Initial documentation |
