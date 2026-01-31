import os
import glob

# Path to your LaSOT dataset root
dataset_root = "data/LaSOT"

def rename_frames():
    # Find all 'img' directories recursively
    # This matches data/LaSOT/*/img/
    search_path = os.path.join(dataset_root, "*", "img")
    img_folders = glob.glob(search_path)
    
    if not img_folders:
        print(f"No 'img' folders found in {dataset_root}. Check your path.")
        return

    print(f"Found {len(img_folders)} folders to process.")

    for folder in img_folders:
        print(f"Processing: {folder}")
        
        # Get all jpg files in the folder
        files = glob.glob(os.path.join(folder, "frame_*.jpg"))
        
        count = 0
        for file_path in files:
            directory, filename = os.path.split(file_path)
            
            # Check if it starts with 'frame_'
            if filename.startswith("frame_"):
                # specific split to handle "frame_00001.jpg" -> "00001.jpg"
                new_filename = filename.replace("frame_", "")
                
                new_path = os.path.join(directory, new_filename)
                
                # Rename the file
                os.rename(file_path, new_path)
                count += 1
        
        print(f"  -> Renamed {count} images.")

if __name__ == "__main__":
    rename_frames()
