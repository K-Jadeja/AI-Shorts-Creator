import cv2

def extract_first_two_minutes(input_file, output_file, duration=120):
    # Load the video
    video = cv2.VideoCapture(input_file)
    
    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_in_seconds = frame_count / fps

    # Ensure we don't try to process more than the video duration
    duration = min(duration, duration_in_seconds)
    total_frames = int(duration * fps)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    frame_number = 0
    while frame_number < total_frames:
        ret, frame = video.read()
        if not ret:
            break
        
        # Write the frame to the output video
        out.write(frame)
        
        frame_number += 1
    
    # Release resources
    video.release()
    out.release()

# Usage
extract_first_two_minutes('input_video.mp4', 'extracted_output.mp4', duration=120)