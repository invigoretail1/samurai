##########################################################

# #!/usr/bin/env python3
# # main_inference.py

# import os
# import os.path as osp
# import re
# import glob
# import gc
# import cv2
# import numpy as np
# import torch

# from sam2.build_sam import build_sam2_video_predictor


# # ----------------------------- helpers -----------------------------

# def load_lasot_gt(gt_path):
#     """Read LaSOT-style GT (x,y,w,h per line) and convert to xyxy for frame-0 seed."""
#     with open(gt_path, 'r') as f:
#         lines = [ln.strip() for ln in f if ln.strip()]
#     prompts = {}
#     for fid, line in enumerate(lines):
#         parts = line.split(',')[:4]
#         if len(parts) < 4:
#             prompts[fid] = ((0, 0, 0, 0), 0)
#             continue
#         x, y, w, h = map(float, parts)
#         x1 = int(round(x))
#         y1 = int(round(y))
#         x2 = int(round(x + w))
#         y2 = int(round(y + h))
#         prompts[fid] = ((x1, y1, x2, y2), 0)
#     return prompts


# def detect_frame_pattern(frame_folder):
#     """
#     Detect (digit width, extension, start index) for numeric frame names.
#     Returns: frame_path(i), num_frames
#       - frame_path(i): callable mapping 0-based frame index -> absolute path
#       - num_frames: number of numeric frames found
#     Supports jpg/jpeg/png/bmp and any zero padding.
#     """
#     exts = ("jpg","jpeg","png","bmp","JPG","JPEG","PNG","BMP")
#     files = []
#     for ext in exts:
#         files.extend(glob.glob(osp.join(frame_folder, f"*.{ext}")))
#     if not files:
#         return None, 0

#     numeric = []
#     for f in files:
#         stem = osp.splitext(osp.basename(f))[0]
#         if re.fullmatch(r"\d+", stem):
#             try:
#                 numeric.append((int(stem), stem, f))
#             except ValueError:
#                 pass

#     if not numeric:
#         # no numeric stems -> fallback: sort by name, treat i -> files[i]
#         files.sort()
#         def frame_path(i):
#             if i < 0 or i >= len(files):
#                 return ""
#             return files[i]
#         return frame_path, len(files)

#     # infer digit width & start index from the smallest numeric name
#     numeric.sort(key=lambda t: t[0])
#     smallest_num, smallest_stem, sample = numeric[0]
#     digits = len(smallest_stem)
#     ext = osp.splitext(sample)[1].lstrip(".")
#     # build a fast set to count frames of same pattern
#     numeric_all = [(n, s, f) for (n, s, f) in numeric if len(s) == digits and f.lower().endswith(ext.lower())]
#     numeric_all.sort(key=lambda t: t[0])
#     start_index = numeric_all[0][0]
#     count = len(numeric_all)

#     def frame_path(i):
#         file_index = start_index + i
#         return osp.join(frame_folder, f"{file_index:0{digits}d}.{ext}")

#     return frame_path, count


# def read_split_list(txt_path):
#     """Lines can be flat ('Seq_0') or hierarchical ('airplane/airplane-1')."""
#     with open(txt_path, 'r') as f:
#         return [ln.strip().rstrip('/') for ln in f if ln.strip()]


# # ----------------------------- config ------------------------------

# TESTING_SET = "data/LaSOT/testing_set.txt"
# VIDEO_ROOT  = "data/LaSOT"

# EXP_NAME   = "samurai"
# MODEL_NAME = "base_plus"   # base_plus | small | base | large (depends on your checkpoints/configs)

# CHECKPOINT = f"sam2/checkpoints/sam2.1_hiera_{MODEL_NAME}.pt"
# if MODEL_NAME == "base_plus":
#     MODEL_CFG = "configs/samurai/sam2.1_hiera_b+.yaml"
# else:
#     MODEL_CFG = f"configs/samurai/sam2.1_hiera_{MODEL_NAME[0]}.yaml"

# PRED_FOLDER = f"results/{EXP_NAME}/{EXP_NAME}_{MODEL_NAME}"
# SAVE_TO_VIDEO = True
# FPS = 30
# CODEC = "mp4v"  # "avc1" or "XVID" if needed

# COLOR = [(255, 0, 0)]  # BGR for predictions
# # -------------------------------------------------------------------


# def main():
#     os.makedirs(PRED_FOLDER, exist_ok=True)
#     if SAVE_TO_VIDEO:
#         vis_folder = f"visualization/{EXP_NAME}/{MODEL_NAME}"
#         os.makedirs(vis_folder, exist_ok=True)

#     test_videos = read_split_list(TESTING_SET)
#     test_videos = sorted(test_videos)

#     predictor = build_sam2_video_predictor(MODEL_CFG, CHECKPOINT, device="cuda:0")

#     for vid, seq_rel in enumerate(test_videos, 1):
#         # accept either "Cat/Seq" or "Seq"
#         frame_folder = osp.join(VIDEO_ROOT, seq_rel, "img")
#         gt_path      = osp.join(VIDEO_ROOT, seq_rel, "groundtruth.txt")
#         safe_name    = seq_rel.replace('/', '__')

#         # existence checks
#         if not osp.isdir(frame_folder):
#             print(f"\033[93m[SKIP] Missing frames folder: {frame_folder}\033[0m")
#             continue
#         if not osp.isfile(gt_path) or os.path.getsize(gt_path) == 0:
#             print(f"\033[93m[SKIP] Missing/empty GT: {gt_path}\033[0m")
#             continue

#         # detect frame naming
#         frame_path, num_frames = detect_frame_pattern(frame_folder)
#         if frame_path is None or num_frames == 0:
#             print(f"\033[93m[SKIP] No readable frames in: {frame_folder}\033[0m")
#             continue

#         # read first frame
#         img0 = cv2.imread(frame_path(0))
#         if img0 is None:
#             print(f"\033[93m[SKIP] Cannot read first frame in: {frame_folder}\033[0m")
#             continue
#         H, W = img0.shape[:2]

#         print(f"\033[91mRunning video [{vid}/{len(test_videos)}]: {seq_rel} with {num_frames} frames\033[0m")
#         print("SAMURAI mode: True")

#         # set up video writer
#         out = None
#         frames_written = 0
#         if SAVE_TO_VIDEO:
#             fourcc = cv2.VideoWriter_fourcc(*CODEC)
#             out_path = osp.join(vis_folder, f"{safe_name}.mp4")
#             out = cv2.VideoWriter(out_path, fourcc, FPS, (W, H))
#             if not out.isOpened():
#                 print(f"\033[93m[WARN] Could not open VideoWriter for: {out_path}\033[0m")
#                 out = None

#         predictions = []

#         try:
#             with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
#                 state = predictor.init_state(
#                     frame_folder,
#                     offload_video_to_cpu=True,
#                     offload_state_to_cpu=True,
#                     async_loading_frames=True
#                 )

#                 prompts = load_lasot_gt(gt_path)

#                 # seed with frame-0 box
#                 if 0 not in prompts:
#                     print(f"\033[93m[SKIP] No frame-0 GT in: {gt_path}\033[0m")
#                     continue
#                 seed_bbox, _ = prompts[0]
#                 predictor.add_new_points_or_box(state, box=seed_bbox, frame_idx=0, obj_id=0)

#                 for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
#                     # Expect exactly one object; handle gracefully otherwise
#                     bbox_to_vis = {}
#                     for obj_id, mask in zip(object_ids, masks):
#                         mask = mask[0].float().cpu().numpy() > 0.0  # [H,W] bool
#                         nz = np.argwhere(mask)
#                         if nz.size == 0:
#                             bbox = [0, 0, 0, 0]
#                         else:
#                             y_min, x_min = nz.min(axis=0).tolist()
#                             y_max, x_max = nz.max(axis=0).tolist()
#                             bbox = [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]
#                         bbox_to_vis[int(obj_id)] = bbox

#                     predictions.append(bbox_to_vis)

#                     if SAVE_TO_VIDEO and out is not None:
#                         img = cv2.imread(frame_path(frame_idx))
#                         if img is None:
#                             print(f"\033[93m[WARN] Missing frame idx {frame_idx}: {frame_path(frame_idx)}\033[0m")
#                             break

#                         # draw predicted bbox (red)
#                         for obj_id, (x, y, w, h) in bbox_to_vis.items():
#                             cv2.rectangle(img, (x, y), (x + w, y + h), COLOR[obj_id % len(COLOR)], 2)

#                         # draw GT only on frame 0 (green)
#                         if frame_idx == 0:
#                             x1, y1, x2, y2 = prompts[0][0]
#                             cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

#                         # write
#                         if img.shape[0] == H and img.shape[1] == W:
#                             out.write(img)
#                             frames_written += 1

#         finally:
#             # write predictions to txt (xywh per frame; obj 0 if present else zeros)
#             os.makedirs(PRED_FOLDER, exist_ok=True)
#             pred_txt = osp.join(PRED_FOLDER, f"{safe_name}.txt")
#             with open(pred_txt, 'w') as f:
#                 for pred in predictions:
#                     x, y, w, h = pred.get(0, [0, 0, 0, 0])
#                     f.write(f"{x},{y},{w},{h}\n")

#             # release video
#             if SAVE_TO_VIDEO and out is not None:
#                 out.release()
#                 if frames_written == 0:
#                     try:
#                         os.remove(out_path)
#                         print(f"\033[93m[INFO] Removed empty video: {out_path}\033[0m")
#                     except Exception:
#                         pass

#             # free
#             del img0
#             gc.collect()
#             torch.clear_autocast_cache()
#             torch.cuda.empty_cache()


# if __name__ == "__main__":
#     main()
###########################################################

#!/usr/bin/env python3
# main_inference.py

import os
import os.path as osp
import re
import glob
import gc
import cv2
import argparse
import numpy as np
import torch
from datetime import datetime

from sam2.build_sam import build_sam2_video_predictor


# ----------------------------- helpers -----------------------------

def load_lasot_gt(gt_path):
    with open(gt_path, 'r') as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    prompts = {}
    for fid, line in enumerate(lines):
        parts = line.split(',')[:4]
        if len(parts) < 4:
            prompts[fid] = ((0, 0, 0, 0), 0)
            continue
        x, y, w, h = map(float, parts)
        x1 = int(round(x));  y1 = int(round(y))
        x2 = int(round(x + w)); y2 = int(round(y + h))
        prompts[fid] = ((x1, y1, x2, y2), 0)
    return prompts


def detect_frame_pattern(frame_folder):
    exts = ("jpg","jpeg","png","bmp","JPG","JPEG","PNG","BMP")
    files = []
    for ext in exts:
        files.extend(glob.glob(osp.join(frame_folder, f"*.{ext}")))
    if not files:
        return None, 0

    numeric = []
    for f in files:
        stem = osp.splitext(osp.basename(f))[0]
        if re.fullmatch(r"\d+", stem):
            try:
                numeric.append((int(stem), stem, f))
            except ValueError:
                pass

    if not numeric:
        files.sort()
        def frame_path(i):
            return files[i] if 0 <= i < len(files) else ""
        return frame_path, len(files)

    numeric.sort(key=lambda t: t[0])
    smallest_num, smallest_stem, sample = numeric[0]
    digits = len(smallest_stem)
    ext = osp.splitext(sample)[1].lstrip(".")
    numeric_all = [(n, s, f) for (n, s, f) in numeric
                   if len(s) == digits and f.lower().endswith(ext.lower())]
    numeric_all.sort(key=lambda t: t[0])
    start_index = numeric_all[0][0]
    count = len(numeric_all)

    def frame_path(i):
        return osp.join(frame_folder, f"{start_index + i:0{digits}d}.{ext}")

    return frame_path, count


def read_split_list(txt_path):
    with open(txt_path, 'r') as f:
        return [ln.strip().rstrip('/') for ln in f if ln.strip()]


# ----------------------------- CLI --------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SAMURAI inference over a LaSOT-style dataset.")
    p.add_argument(
        "--dataset", type=str, default=None,
        help="Dataset tag appended to the LaSOT root folder name, "
             "e.g.  --dataset 2025-05-20  →  data/LaSOT_2025-05-20\n"
             "      --dataset session1    →  data/LaSOT_session1\n"
             "Omit to use  data/LaSOT  as-is."
    )
    p.add_argument(
        "--data-root", type=str, default=None,
        help="Full override for the data root path (ignores --dataset). "
             "e.g.  --data-root /mnt/nas/LaSOT_session1"
    )
    p.add_argument(
        "--run-id", type=str, default=None,
        help="Tag appended to result/visualization folder names. "
             "Defaults to current datetime: YYYY-MM-DD_HH-MM-SS"
    )
    p.add_argument(
        "--model", type=str, default="base_plus",
        choices=["base_plus", "small", "base", "large"],
        help="SAM-2 model variant (default: base_plus)"
    )
    p.add_argument(
        "--no-video", action="store_true",
        help="Disable visualization video output"
    )
    return p.parse_args()


# ----------------------------- main --------------------------------

def main():
    args = parse_args()

    # ── run identity ──────────────────────────────────────────────
    run_id     = args.run_id or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = args.model
    exp_name   = "samurai"

    # ── paths ─────────────────────────────────────────────────────
    if args.data_root:
        video_root = args.data_root                          # full manual override
    elif args.dataset:
        video_root = f"data/LaSOT_{args.dataset}"           # data/LaSOT_2025-05-20
    else:
        video_root = "data/LaSOT"                           # plain default
        
    testing_set  = osp.join(video_root, "testing_set.txt")

    checkpoint = f"sam2/checkpoints/sam2.1_hiera_{model_name}.pt"
    if model_name == "base_plus":
        model_cfg = "configs/samurai/sam2.1_hiera_b+.yaml"
    else:
        model_cfg = f"configs/samurai/sam2.1_hiera_{model_name[0]}.yaml"

    # e.g.  results/samurai/base_plus_2025-05-20_14-30-00/
    pred_folder = osp.join("results", exp_name, f"{model_name}_{run_id}")
    vis_folder  = osp.join("visualization", exp_name, f"{model_name}_{run_id}")

    save_to_video = not args.no_video
    fps   = 30
    codec = "mp4v"
    color = [(255, 0, 0)]

    print(f"Run ID     : {run_id}")
    print(f"Model      : {model_name}")
    print(f"Data root  : {video_root}")
    print(f"Results    : {pred_folder}")
    if save_to_video:
        print(f"Vis folder : {vis_folder}")
    print()

    # ── setup ─────────────────────────────────────────────────────
    os.makedirs(pred_folder, exist_ok=True)
    if save_to_video:
        os.makedirs(vis_folder, exist_ok=True)

    test_videos = sorted(read_split_list(testing_set))
    predictor   = build_sam2_video_predictor(model_cfg, checkpoint, device="cuda:0")

    for vid, seq_rel in enumerate(test_videos, 1):
        frame_folder = osp.join(video_root, seq_rel, "img")
        gt_path      = osp.join(video_root, seq_rel, "groundtruth.txt")
        safe_name    = seq_rel.replace('/', '__')

        if not osp.isdir(frame_folder):
            print(f"\033[93m[SKIP] Missing frames: {frame_folder}\033[0m")
            continue
        if not osp.isfile(gt_path) or os.path.getsize(gt_path) == 0:
            print(f"\033[93m[SKIP] Missing/empty GT: {gt_path}\033[0m")
            continue

        frame_path, num_frames = detect_frame_pattern(frame_folder)
        if frame_path is None or num_frames == 0:
            print(f"\033[93m[SKIP] No readable frames: {frame_folder}\033[0m")
            continue

        img0 = cv2.imread(frame_path(0))
        if img0 is None:
            print(f"\033[93m[SKIP] Cannot read first frame: {frame_folder}\033[0m")
            continue
        H, W = img0.shape[:2]

        print(f"\033[91m[{vid}/{len(test_videos)}] {seq_rel}  ({num_frames} frames)\033[0m")

        out = None
        frames_written = 0
        if save_to_video:
            fourcc   = cv2.VideoWriter_fourcc(*codec)
            out_path = osp.join(vis_folder, f"{safe_name}.mp4")
            out      = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
            if not out.isOpened():
                print(f"\033[93m[WARN] Cannot open VideoWriter: {out_path}\033[0m")
                out = None

        predictions = []

        try:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                state   = predictor.init_state(
                    frame_folder,
                    offload_video_to_cpu=True,
                    offload_state_to_cpu=True,
                    async_loading_frames=True,
                )
                prompts = load_lasot_gt(gt_path)

                if 0 not in prompts:
                    print(f"\033[93m[SKIP] No frame-0 GT: {gt_path}\033[0m")
                    continue

                seed_bbox, _ = prompts[0]
                predictor.add_new_points_or_box(state, box=seed_bbox, frame_idx=0, obj_id=0)

                for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
                    bbox_to_vis = {}
                    for obj_id, mask in zip(object_ids, masks):
                        mask = mask[0].float().cpu().numpy() > 0.0
                        nz   = np.argwhere(mask)
                        if nz.size == 0:
                            bbox = [0, 0, 0, 0]
                        else:
                            y_min, x_min = nz.min(axis=0).tolist()
                            y_max, x_max = nz.max(axis=0).tolist()
                            bbox = [int(x_min), int(y_min),
                                    int(x_max - x_min), int(y_max - y_min)]
                        bbox_to_vis[int(obj_id)] = bbox
                    predictions.append(bbox_to_vis)

                    if save_to_video and out is not None:
                        img = cv2.imread(frame_path(frame_idx))
                        if img is None:
                            print(f"\033[93m[WARN] Missing frame {frame_idx}\033[0m")
                            break
                        for obj_id, (x, y, w, h) in bbox_to_vis.items():
                            cv2.rectangle(img, (x, y), (x+w, y+h),
                                          color[obj_id % len(color)], 2)
                        if frame_idx == 0:
                            x1, y1, x2, y2 = prompts[0][0]
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        if img.shape[0] == H and img.shape[1] == W:
                            out.write(img)
                            frames_written += 1

        finally:
            os.makedirs(pred_folder, exist_ok=True)
            with open(osp.join(pred_folder, f"{safe_name}.txt"), 'w') as f:
                for pred in predictions:
                    x, y, w, h = pred.get(0, [0, 0, 0, 0])
                    f.write(f"{x},{y},{w},{h}\n")

            if save_to_video and out is not None:
                out.release()
                if frames_written == 0:
                    try:
                        os.remove(out_path)
                    except Exception:
                        pass

            del img0
            gc.collect()
            torch.clear_autocast_cache()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()