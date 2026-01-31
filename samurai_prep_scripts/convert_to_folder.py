import cv2
import os
import argparse
import sys

def extract_frames(source_folder, output_folder):
    # Supported video extensions
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv')
    
    # Check if source exists
    if not os.path.exists(source_folder):
        print(f"Error: Source folder '{source_folder}' not found.")
        sys.exit(1)

    # Create the main output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")

    # Walk through the directory to find video files
    video_found = False
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.lower().endswith(video_extensions):
                video_found = True
                video_path = os.path.join(root, file)
                video_name = os.path.splitext(file)[0]
                
                # Create a specific folder for this video's frames
                save_path = os.path.join(output_folder, video_name)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                
                print(f"Processing: {file}...")
                
                # Capture the video
                cap = cv2.VideoCapture(video_path)
                frame_count = 0
                
                while True:
                    success, frame = cap.read()
                    
                    if not success:
                        break # End of video
                    
                    # Save frame as JPG
                    # format: frame_00001.jpg
                    frame_filename = os.path.join(save_path, f"frame_{frame_count:05d}.jpg")
                    cv2.imwrite(frame_filename, frame)
                    
                    frame_count += 1
                
                cap.release()
                print(f"Finished {file}: Extracted {frame_count} frames.")
    
    if not video_found:
        print("No video files found in the source directory.")
    else:
        print("\nExtraction complete.")

if __name__ == "__main__":
    # Initialize Argument Parser
    parser = argparse.ArgumentParser(description="Extract frames from videos in a folder.")
    
    # Add arguments with '--' to make them named flags
    parser.add_argument("--input_dir", required=True, help="Path to the folder containing videos")
    parser.add_argument("--output_dir", required=True, help="Path to the folder where frames will be saved")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the function with CLI args
    extract_frames(args.input_dir, args.output_dir)