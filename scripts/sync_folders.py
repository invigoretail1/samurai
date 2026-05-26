#!/usr/bin/env python3
import argparse
import sys
import shutil
import re
from pathlib import Path

RAW_IMG_DIRNAME = "img"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_imgs(p: Path):
    if not p.is_dir():
        return []
    return [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXTS]


def to_index(path: Path):
    digits = re.sub(r"\D", "", path.stem)
    return int(digits) if digits else None


def filter_predictions(pred_txt: Path, allowed_idx: set, out_path: Path) -> tuple[int, int]:
    lines = pred_txt.read_text().splitlines()
    kept, dropped = 0, 0
    out_lines = []
    for i, line in enumerate(lines, start=1):
        if i in allowed_idx:
            out_lines.append(line)   # keep line in place
            kept += 1
        else:
            out_lines.append("")     # blank placeholder, preserves index alignment
            dropped += 1
    out_path.write_text("\n".join(out_lines) + "\n")
    return kept, dropped


def main():
    parser = argparse.ArgumentParser(
        description="Sync cleaned frames + filtered SAMURAI predictions into images/ and labels/ folders."
    )
    parser.add_argument("--raw_root",     required=True, type=Path,
                        help="LaSOT root with seq folders (each has img/ subfolder)")
    parser.add_argument("--clean_root",   required=True, type=Path,
                        help="Folder with seq subfolders containing only the kept frames")
    parser.add_argument("--results_root", required=True, type=Path,
                        help="SAMURAI results folder, e.g. results/samurai/base_plus_2025-05-20/")
    parser.add_argument("--output_root",  required=True, type=Path,
                        help="Destination root. Will contain images/ and labels/ subfolders.")
    args = parser.parse_args()

    raw_root     = args.raw_root
    clean_root   = args.clean_root
    results_root = args.results_root
    output_root  = args.output_root

    for p, label in [(raw_root, "raw_root"), (clean_root, "clean_root"), (results_root, "results_root")]:
        if not p.is_dir():
            print(f"[ERR] {label} not found: {p}")
            sys.exit(1)

    images_root = output_root / "images"
    labels_root = output_root / "labels"
    images_root.mkdir(parents=True, exist_ok=True)
    labels_root.mkdir(parents=True, exist_ok=True)

    print(f"Raw root     : {raw_root}")
    print(f"Clean root   : {clean_root}")
    print(f"Results root : {results_root}")
    print(f"Output       : {output_root}")
    print(f"  images/    : {images_root}")
    print(f"  labels/    : {labels_root}")
    print("-" * 60)

    total_frames = 0
    total_labels = 0
    missing_preds = []

    clean_seq_dirs = sorted(d for d in clean_root.iterdir() if d.is_dir())
    if not clean_seq_dirs:
        print("[WARN] No sequence directories found in clean_root.")

    for seq_dir in clean_seq_dirs:
        seq_name = seq_dir.name
        raw_seq  = raw_root / seq_name

        # locate raw frames dir
        raw_imgs_dir = (raw_seq / RAW_IMG_DIRNAME) if (raw_seq / RAW_IMG_DIRNAME).is_dir() else raw_seq
        if not raw_imgs_dir.is_dir():
            print(f"[SKIP] No raw folder for '{seq_name}'")
            continue

        # allowed indices from clean set
        allowed_idx = {to_index(f) for f in list_imgs(seq_dir) if to_index(f) is not None}

        raw_files = list_imgs(raw_imgs_dir)
        keep = [f for f in raw_files if to_index(f) in allowed_idx]

        # ── images → output_root/images/seq_name/ ─────────────────
        out_imgs = images_root / seq_name
        out_imgs.mkdir(parents=True, exist_ok=True)
        for f in keep:
            shutil.copy2(f, out_imgs / f.name)
        total_frames += len(keep)

        # ── labels → output_root/labels/seq_name.txt ──────────────
        pred_txt = results_root / f"{seq_name}.txt"
        if pred_txt.is_file():
            out_label = labels_root / f"{seq_name}.txt"
            kept_lines, dropped_lines = filter_predictions(pred_txt, allowed_idx, out_label)
            total_labels += 1
            label_status = f"kept={kept_lines} dropped={dropped_lines}"
        else:
            missing_preds.append(seq_name)
            label_status = "MISSING"

        print(f"[{seq_name}] frames={len(keep)} | labels={label_status}")

    print("\n" + "=" * 40)
    print("Summary:")
    print(f"  Frames copied  : {total_frames}")
    print(f"  Label files    : {total_labels}")
    if missing_preds:
        print(f"  Missing preds  : {', '.join(missing_preds)}")
    print(f"  Output         : {output_root}")
    print("=" * 40)


if __name__ == "__main__":
    main()