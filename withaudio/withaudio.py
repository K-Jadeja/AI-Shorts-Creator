import cv2
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip
import time
import os

def smooth_tracking(previous, current, smoothing_factor=0.8):
    return previous * smoothing_factor + current * (1 - smoothing_factor)

def process_video(input_file, output_file):
    # Extract audio from input video
    print("Extracting audio...")
    video_with_audio = VideoFileClip(input_file)
    audio = video_with_audio.audio
    audio_file = "temp_audio.mp3"
    audio.write_audiofile(audio_file)
    video_with_audio.close()

    video = cv2.VideoCapture(input_file)
    
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} fps, {total_frames} frames")
    
    output_width, output_height = 1080, 1920
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    temp_output_file = "temp_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output_file, fourcc, fps, (output_width, output_height))
    
    frame_count = 0
    start_time = time.time()
    
    # Initialize tracking variables
    prev_center_x, prev_center_y = width // 2, height // 2
    smooth_center_x, smooth_center_y = prev_center_x, prev_center_y
    
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
            # Find the face closest to the previous center
            closest_face = min(faces, key=lambda f: abs(f[0] + f[2]//2 - prev_center_x) + abs(f[1] + f[3]//2 - prev_center_y))
            (x, y, w, h) = closest_face
            
            center_x, center_y = x + w // 2, y + h // 2
            
            # Apply smoothing
            smooth_center_x = smooth_tracking(smooth_center_x, center_x)
            smooth_center_y = smooth_tracking(smooth_center_y, center_y)
            
            prev_center_x, prev_center_y = smooth_center_x, smooth_center_y
        else:
            # If no face, gradually move towards the center
            smooth_center_x = smooth_tracking(smooth_center_x, width // 2)
            smooth_center_y = smooth_tracking(smooth_center_y, height // 2)
        
        # Calculate crop area
        crop_x = int(max(0, min(smooth_center_x - output_width // 2, width - output_width)))
        crop_y = int(max(0, min(smooth_center_y - output_height // 2, height - output_height)))
        
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

    print("Adding audio and subtitles...")
    video = VideoFileClip(temp_output_file)
    audio = AudioFileClip(audio_file)
    video = video.set_audio(audio)
    txt_clip = TextClip("Sample Subtitle", fontsize=70, color='white')
    txt_clip = txt_clip.set_pos(('center', 'bottom')).set_duration(video.duration)
    
    final = CompositeVideoClip([video, txt_clip])
    final.write_videofile(output_file, codec='libx264', audio_codec='aac')
    
    # Clean up temporary files
    os.remove(temp_output_file)
    os.remove(audio_file)
    
    print("Video processing, audio addition, and subtitle addition completed.")

# Usage
input_file = 'trimmed_input.mp4'
output_file = './withaudio/withaudio.mp4'
process_video(input_file, output_file)