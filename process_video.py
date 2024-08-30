import cv2
import numpy as np
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip

def process_video(input_file, output_file):
    # Load the video
    video = cv2.VideoCapture(input_file)
    
    # Get video properties
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    
    # Define the output size (adjust as needed for phone screen)
    output_width, output_height = 720, 1280
    
    # Initialize face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (output_width, output_height))
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # Get the largest face
            (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
            
            # Calculate crop area
            center_x = x + w // 2
            center_y = y + h // 2
            crop_x = max(0, center_x - output_width // 2)
            crop_y = max(0, center_y - output_height // 2)
            
            # Crop and resize
            cropped = frame[crop_y:crop_y+output_height, crop_x:crop_x+output_width]
            if cropped.shape[0] != output_height or cropped.shape[1] != output_width:
                cropped = cv2.resize(cropped, (output_width, output_height))
        else:
            # If no face, center crop
            crop_x = max(0, width // 2 - output_width // 2)
            crop_y = max(0, height // 2 - output_height // 2)
            cropped = frame[crop_y:crop_y+output_height, crop_x:crop_x+output_width]
            
        out.write(cropped)
    
    video.release()
    out.release()

    # Add subtitles
    video = VideoFileClip(output_file)
    # Here you would add logic to generate subtitles, possibly using a speech recognition library
    # For demonstration, we'll just add a sample subtitle
    txt_clip = TextClip("Sample Subtitle", fontsize=70, color='white')
    txt_clip = txt_clip.set_pos(('center', 'bottom')).set_duration(video.duration)
    
    final = CompositeVideoClip([video, txt_clip])
    final.write_videofile(output_file, codec='libx264')

# Usage
process_video('input_video.mp4', 'output.mp4')