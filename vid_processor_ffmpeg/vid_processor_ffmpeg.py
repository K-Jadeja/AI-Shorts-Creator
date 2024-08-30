import cv2
import numpy as np
import subprocess
import os
import time

def smooth_tracking(previous, current, smoothing_factor=0.8):
    return previous * smoothing_factor + current * (1 - smoothing_factor)

def process_video(input_file, output_file):
    video = cv2.VideoCapture(input_file)
    
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} fps, {total_frames} frames")
    
    # Set output size for 9:16 aspect ratio (common for mobile)
    output_height = 1920
    output_width = 1080
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    temp_output_file = "temp_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output_file, fourcc, fps, (output_width, output_height))
    
    frame_count = 0
    start_time = time.time()
    
    # Initialize tracking variables
    prev_center_x, prev_center_y = width // 2, height // 2
    smooth_center_x, smooth_center_y = prev_center_x, prev_center_y
    
    # Calculate zoom factor (increased significantly)
    zoom_factor = 0.4  # This means we're only showing 40% of the original frame width
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 100 == 0:
            elapsed_time = time.time() - start_time
            frames_per_second = frame_count / elapsed_time
            print(f"Processed {frame_count}/{total_frames} frames. FPS: {frames_per_second:.2f}")
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # Find the largest face
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            (x, y, w, h) = largest_face
            
            center_x, center_y = x + w // 2, y + h // 2
            
            # Apply smoothing
            smooth_center_x = int(smooth_tracking(smooth_center_x, center_x))
            smooth_center_y = int(smooth_tracking(smooth_center_y, center_y))
            
            prev_center_x, prev_center_y = smooth_center_x, smooth_center_y
        else:
            # If no face, gradually move towards the center
            smooth_center_x = int(smooth_tracking(smooth_center_x, width // 2))
            smooth_center_y = int(smooth_tracking(smooth_center_y, height // 2))
        
        # Calculate crop area with zoom
        crop_width = int(width * zoom_factor)
        crop_height = int(height * zoom_factor)
        crop_x = max(0, min(smooth_center_x - crop_width // 2, width - crop_width))
        crop_y = max(0, min(smooth_center_y - crop_height // 2, height - crop_height))
        
        # Ensure crop dimensions are within frame boundaries
        crop_x_end = min(crop_x + crop_width, width)
        crop_y_end = min(crop_y + crop_height, height)
        cropped = frame[crop_y:crop_y_end, crop_x:crop_x_end]
        
        # Resize to output dimensions
        resized = cv2.resize(cropped, (output_width, output_height))
        
        out.write(resized)
    
    video.release()
    out.release()
    
    print(f"Video processing completed. Total frames processed: {frame_count}")

    print("Merging processed video with original audio...")
    ffmpeg_cmd = [
        'ffmpeg',
        '-i', temp_output_file,
        '-i', input_file,
        '-c:v', 'libx264',
        '-crf', '23',
        '-preset', 'medium',
        '-c:a', 'aac',
        '-b:a', '128k',
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-shortest',
        output_file
    ]
    subprocess.run(ffmpeg_cmd, check=True)
    
    # Clean up temporary file
    os.remove(temp_output_file)
    
    print("Video processing and audio merging completed.")

# Usage
input_file = 'trimmed_input.mp4'
output_file = './vid_processor_ffmpeg/vid_processor_ffmpeg.mp4'
process_video(input_file, output_file)