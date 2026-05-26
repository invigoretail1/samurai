#!/usr/bin/env python3
"""
YOLO CONVERT AND MERGE TOOL
---------------------------
Author: Mohan
Description:
    1. Reads raw LabelMe/Tracker text files and corresponding video frames.
    2. matches them via Frame Index (frame_005.jpg -> Line 5 of txt).
    3. Converts coordinates to YOLO (normalized xywh).
    4. Smartly normalizes class names to merge different camera angles/runs into one class.
    5. Optionally MERGES with an existing YOLO dataset to create a combined master set.

Usage:
    # Step 1: Convert first batch
    python script.py --txt-dir path/to/txt1 --frames-root path/to/frames1 --out ./dataset_part1

    # Step 2: Convert second batch AND merge Part 1
    python script.py --txt-dir path/to/txt2 --frames-root path/to/frames2 --out ./dataset_combined \
                     --extend ./dataset_part1 --merge-extend
"""

import argparse, sys, shutil, glob, re, random, ast, os
from pathlib import Path
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ---------------------- Index & Box Helpers ----------------------

def extract_frame_index(filename):
    """Robustly extracts index from 'frame_00105.jpg' -> 105"""
    digits = re.findall(r'\d+', filename)
    if not digits: return None
    return int(digits[-1])

def xywh_to_xyxy(x, y, w, h):
    return x, y, x + w, y + h

def normalize_box_xyxy(x1, y1, x2, y2, W, H):
    x1 = max(0, min(x1, W - 1)); y1 = max(0, min(y1, H - 1))
    x2 = max(0, min(x2, W - 1)); y2 = max(0, min(y2, H - 1))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    bw = max(0.0, x2 - x1); bh = max(0.0, y2 - y1)
    if bw <= 0 or bh <= 0: return None
    cx = (x1 + x2) * 0.5; cy = (y1 + y2) * 0.5
    return (cx / W, cy / H, bw / W, bh / H)

def read_all_boxes(txt_path: Path):
    """Reads all lines to allow index-based matching."""
    boxes = []
    with txt_path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip().strip(",")
            if not ln:
                boxes.append(None)
                continue
            parts = [p.strip() for p in ln.replace(" ", "").split(",")]
            if len(parts) < 4:
                boxes.append(None)
                continue
            x1, y1, x2, y2 = map(float, parts[:4])
            boxes.append((x1, y1, x2, y2))
    return boxes

def sorted_images_in(pathlike):
    if isinstance(pathlike, Path) and pathlike.is_dir():
        imgs = [p for p in pathlike.iterdir() if p.suffix.lower() in IMG_EXTS]
    else:
        imgs = [Path(p) for p in glob.glob(str(pathlike))]
    return sorted(imgs)

# ---------------------- Name Normalization ----------------------

def smart_cleanup_name(name: str):
    """
    Turns: 'Hershey_Kisses_Cereal_1_cam0' -> 'Hershey_Kisses_Cereal'
    Turns: 'Robitussin_CC_cam1'          -> 'Robitussin_CC'
    """
    # 1. Remove _camX
    name = re.sub(r'_cam\d+$', '', name)
    # 2. Remove trailing _1, _2 if it exists (for the second run of videos)
    name = re.sub(r'_\d+$', '', name)
    return name

# ---------------------- YAML & Merge Helpers ----------------------

def _extract_names_from_yaml_text(txt: str):
    m = re.search(r'^\s*names\s*:\s*(\[[^\]]*\])', txt, re.MULTILINE)
    if m:
        try: return list(ast.literal_eval(m.group(1)))
        except: pass
    return [] # Simplified for brevity, usually inline list is used

def load_existing_class_names(extend_path: Path):
    if extend_path.is_dir(): yaml_path = extend_path / "data.yaml"
    else: yaml_path = extend_path
    if not yaml_path.exists(): return []
    try:
        return _extract_names_from_yaml_text(yaml_path.read_text(encoding="utf-8"))
    except: return []

def _link_or_copy(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists(): return
    if mode == "copy": shutil.copy2(src, dst)
    elif mode == "hard": os.link(src, dst)
    else: os.symlink(src, dst)

def _rewrite_label_ids(src_lbl: Path, dst_lbl: Path, id_map: dict):
    if not src_lbl.exists(): return
    lines_out = []
    with src_lbl.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln: continue
            parts = ln.split()
            try: old_id = int(parts[0])
            except: continue
            new_id = id_map.get(old_id, old_id)
            parts[0] = str(new_id)
            lines_out.append(" ".join(parts))
    dst_lbl.parent.mkdir(parents=True, exist_ok=True)
    with dst_lbl.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines_out) + "\n")

def merge_extend_dataset(extend_root: Path, out_root: Path, id_map: dict, link_mode: str):
    for split in ("train", "val"):
        img_dir = extend_root / "images" / split
        lbl_dir = extend_root / "labels" / split
        if not img_dir.exists(): continue
        
        for img_path in img_dir.rglob("*"):
            if img_path.suffix.lower() not in IMG_EXTS: continue
            rel = img_path.relative_to(img_dir)
            dst_img = out_root / "images" / split / rel
            
            # Avoid overwriting files with same name
            if dst_img.exists():
                print(f"[WARN] Collision during merge: {rel} exists. Skipping.")
                continue
                
            _link_or_copy(img_path, dst_img, link_mode)
            
            # Copy and remap label
            src_lbl = (lbl_dir / rel).with_suffix(".txt")
            dst_lbl = out_root / "labels" / split / rel.with_suffix(".txt")
            if src_lbl.exists():
                _rewrite_label_ids(src_lbl, dst_lbl, id_map)

# ---------------------- Main ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--txt-dir", required=True, help="Folder of *.txt")
    ap.add_argument("--frames-root", required=True, help="Root containing per-video frame folders")
    ap.add_argument("--out", required=True, help="Output YOLO dataset dir")
    ap.add_argument("--val-ratio", type=float, default=0.1)

    # Class Naming
    ap.add_argument("--class-mode", choices=["single","smart"], default="smart",
                    help="'smart' cleans names (Hershey_1_cam0 -> Hershey). 'single' makes everything 1 class.")
    ap.add_argument("--name", default="product", help="Class name for single-class mode")

    # Merge Options
    ap.add_argument("--extend", type=str, default=None,
                    help="Path to previous YOLO dataset to merge.")
    ap.add_argument("--merge-extend", action="store_true",
                    help="Enable the merge process.")
    
    args = ap.parse_args()

    txt_dir = Path(args.txt_dir).resolve()
    frames_root = Path(args.frames_root).resolve()
    out_root = Path(args.out).resolve()

    if out_root.exists(): shutil.rmtree(out_root)
    for sub in ["images/train","images/val","labels/train","labels/val"]:
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    # 1. Load Existing Classes (if merging)
    class_map = {}     # name -> id
    class_names = []   # index -> name

    if args.extend:
        existing_names = load_existing_class_names(Path(args.extend))
        for n in existing_names:
            class_map[n] = len(class_names)
            class_names.append(n)
        print(f"[INFO] Loaded {len(existing_names)} existing classes.")

    def get_class_id(raw_stem):
        if args.class_mode == "single":
            cname = args.name
        else:
            # Apply Smart Cleanup
            cname = smart_cleanup_name(raw_stem)
            
        if cname not in class_map:
            class_map[cname] = len(class_names)
            class_names.append(cname)
        return class_map[cname]

    # 2. Process New Data
    txt_files = sorted([p for p in txt_dir.iterdir() if p.suffix.lower()==".txt"])
    all_pairs = []

    for t in txt_files:
        stem = t.stem
        folder = frames_root / stem
        
        # Folder finding logic
        candidates = []
        if folder.exists() and folder.is_dir():
            candidates = sorted_images_in(folder)
        else:
            candidates = sorted_images_in(frames_root.glob(stem + "*"))

        if not candidates:
            print(f"[WARN] No frames for {stem}")
            continue

        boxes_list = read_all_boxes(t) # Uses index logic
        cid = get_class_id(stem)
        
        for img_path in candidates:
            # INDEX MATCHING
            idx = extract_frame_index(img_path.name)
            if idx is None or idx >= len(boxes_list) or boxes_list[idx] is None:
                continue

            # Box conversion
            x1, y1, x2, y2 = boxes_list[idx]
            with Image.open(img_path) as im:
                W, H = im.size
            
            # Assume input is xywh pixel
            x, y, w, h = x1, y1, x2, y2
            x1, y1, x2, y2 = xywh_to_xyxy(x, y, w, h)
            yolo = normalize_box_xyxy(x1, y1, x2, y2, W, H)
            
            if yolo:
                cx, cy, bw, bh = yolo
                yline = f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
                all_pairs.append((img_path, yline))

    # 3. Write New Data
    random.shuffle(all_pairs)
    val_n = int(len(all_pairs) * args.val_ratio)
    val_set = set(all_pairs[:val_n])

    for img_path, yline in all_pairs:
        split = "val" if (img_path, yline) in val_set else "train"
        # Prefix filename with parent folder to avoid collisions if merging datasets with same frame names
        new_name = f"{img_path.parent.name}_{img_path.name}"
        
        dst_img = out_root / "images" / split / new_name
        dst_lbl = out_root / "labels" / split / (Path(new_name).stem + ".txt")
        shutil.copy2(img_path, dst_img)
        with dst_lbl.open("w", encoding="utf-8") as f:
            f.write(yline + "\n")

    # 4. MERGE Old Dataset
    if args.extend and args.merge_extend:
        print("[INFO] Merging previous dataset...")
        # Create map from old ID to new ID
        # (In this script, IDs are preserved if names match, appended if new)
        old_names = load_existing_class_names(Path(args.extend))
        id_remap = {}
        for old_id, name in enumerate(old_names):
            if name in class_map:
                id_remap[old_id] = class_map[name]
            else:
                # Should match, but if something weird happens
                id_remap[old_id] = get_class_id(name)
        
        merge_extend_dataset(Path(args.extend), out_root, id_remap, "copy")

    # 5. Write Data YAML
    names_formatted = ", ".join([f"'{n}'" for n in class_names])
    yaml_content = f"path: {out_root.as_posix()}\ntrain: images/train\nval: images/val\nnames: [{names_formatted}]"
    (out_root / "data.yaml").write_text(yaml_content, encoding="utf-8")

    print(f"[SUCCESS] Final Dataset at: {out_root}")
    print(f"Classes: {class_names}")

if __name__ == "__main__":
    main()