import cv2
import numpy as np
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
import time

def process_video(input_file, output_file):
    # Load the video
    video = cv2.VideoCapture(input_file)
    
    # Get video properties
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} fps, {total_frames} frames")
    
    # Define the output size (adjust as needed for phone screen)
    output_width, output_height = 720, 1280
    
    # Initialize face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (output_width, output_height))
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 100 == 0:
            elapsed_time = time.time() - start_time
            frames_per_second = frame_count / elapsed_time
            print(f"Processed {frame_count}/{total_frames} frames. "
                  f"FPS: {frames_per_second:.2f}")
        
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # Get the largest face
            (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
            
            # Calculate crop area
            center_x = x + w // 2
            center_y = y + h // 2
            crop_x = max(0, min(center_x - output_width // 2, width - output_width))
            crop_y = max(0, min(center_y - output_height // 2, height - output_height))
        else:
            # If no face, center crop
            crop_x = max(0, width // 2 - output_width // 2)
            crop_y = max(0, height // 2 - output_height // 2)
        
        # Ensure crop dimensions are within frame boundaries
        crop_x_end = min(crop_x + output_width, width)
        crop_y_end = min(crop_y + output_height, height)
        cropped = frame[crop_y:crop_y_end, crop_x:crop_x_end]
        
        # Resize if necessary
        if cropped.shape[:2] != (output_height, output_width):
            cropped = cv2.resize(cropped, (output_width, output_height))
        
        out.write(cropped)
    
    video.release()
    out.release()
    
    print(f"Video processing completed. Total frames processed: {frame_count}")

    # Add subtitles
    print("Adding subtitles...")
    video = VideoFileClip(output_file)
    # Here you would add logic to generate subtitles, possibly using a speech recognition library
    # For demonstration, we'll just add a sample subtitle
    txt_clip = TextClip("Sample Subtitle", fontsize=70, color='white')
    txt_clip = txt_clip.set_pos(('center', 'bottom')).set_duration(video.duration)
    
    final = CompositeVideoClip([video, txt_clip])
    final.write_videofile(output_file, codec='libx264')
    
    print("Video processing and subtitle addition completed.")

# Usage
input_file = 'short_input.mp4'
output_file = 'output.mp4'
process_video(input_file, output_file)