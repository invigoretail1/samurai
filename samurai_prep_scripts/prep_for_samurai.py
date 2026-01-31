import os
import json
import shutil
import glob
import argparse
import sys

def process_folders(root_dir, vid_source):
    # Verify input directory exists
    if not os.path.exists(root_dir):
        print(f"Error: Frames directory '{root_dir}' does not exist.")
        sys.exit(1)
        
    if not os.path.exists(vid_source):
        print(f"Warning: Video source directory '{vid_source}' does not exist. Video moving step will be skipped.")

    # Iterate over every folder in the frames root directory
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        
        # Skip if it's not a directory
        if not os.path.isdir(folder_path):
            continue
            
        print(f"Processing: {folder_name}...")

        # -----------------------------
        # 1. Create groundtruth.txt
        # -----------------------------
        json_files = glob.glob(os.path.join(folder_path, "*.json"))
        
        if json_files:
            # Take the first json found (usually frame_00000.json)
            json_file_path = json_files[0]
            
            with open(json_file_path, 'r') as f:
                data = json.load(f)
            
            try:
                # Extract first shape
                shape = data['shapes'][0]
                points = shape['points']
                
                # LabelMe stores as [[x1, y1], [x2, y2]]
                (x1, y1), (x2, y2) = points
                
                # Calculate x, y (top-left) and w, h
                x = min(x1, x2)
                y = min(y1, y2)
                w = abs(x2 - x1)
                h = abs(y2 - y1)
                
                # Write to groundtruth.txt
                gt_path = os.path.join(folder_path, "groundtruth.txt")
                with open(gt_path, 'w') as out_f:
                    # Writing format: x,y,w,h
                    out_f.write(f"{x},{y},{w},{h}")
                    
                print(f"  -> Generated groundtruth.txt: {x},{y},{w},{h}")
                
            except (IndexError, KeyError) as e:
                print(f"  -> Error parsing JSON in {folder_name}: {e}")
        else:
            print(f"  -> No JSON found in {folder_name}")

        # -----------------------------
        # 2. Reorganize Structure & RENAME FILES
        # -----------------------------
        # Create 'img' directory
        img_dir = os.path.join(folder_path, "img")
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
            
        # Get all .jpg files
        jpg_files = glob.glob(os.path.join(folder_path, "*.jpg"))
        
        for jpg in jpg_files:
            filename = os.path.basename(jpg)
            
            # --- RENAME LOGIC ---
            # If the file starts with "frame_", remove it.
            if filename.startswith("frame_"):
                new_filename = filename.replace("frame_", "")
            else:
                new_filename = filename
            
            # Move and Rename simultaneously
            destination = os.path.join(img_dir, new_filename)
            shutil.move(jpg, destination)
            
        print(f"  -> Moved & Renamed {len(jpg_files)} frames to /img")

        # -----------------------------
        # 3. Move Original Video (Optional)
        # -----------------------------
        if os.path.exists(vid_source):
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
            for ext in video_extensions:
                video_filename = folder_name + ext
                source_video_path = os.path.join(vid_source, video_filename)
                
                if os.path.exists(source_video_path):
                    shutil.move(source_video_path, os.path.join(folder_path, video_filename))
                    print(f"  -> Moved video {video_filename} into folder.")
                    break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process LabelMe frames for SAMURAI/LaSOT format.")
    
    # Required Arguments
    parser.add_argument("--frames_dir", required=True, help="The root folder containing your frame subfolders")
    parser.add_argument("--videos_dir", required=True, help="The folder containing the original video files")

    args = parser.parse_args()
    
    process_folders(args.frames_dir, args.videos_dir)