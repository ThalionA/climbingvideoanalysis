import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def process_frame(pose, frame_rgb):
    results = pose.process(frame_rgb)
    return results

def process_video(video_file):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Temporary file for the output video
    out_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    out_file_name = out_file.name
    
    # Create video writer object
    out = cv2.VideoWriter(out_file_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    
    start_time = time.time()
    frames_processed = 0
    
    progress_bar = st.progress(0)
    progress_text = st.empty()
    eta_text = st.empty()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        with ThreadPoolExecutor(max_workers=4) as executor:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                futures = [executor.submit(process_frame, pose, frame_rgb)]
                results = [f.result() for f in futures]
                
                for result in results:
                    mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Convert RGB back to BGR before writing 
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame)
                
                frames_processed += 1
                elapsed_time = time.time() - start_time
                estimated_total_time = (elapsed_time / frames_processed) * frame_count
                remaining_time = estimated_total_time - elapsed_time
                
                progress_bar.progress(frames_processed / frame_count)
                progress_text.text(f"Processing video... {frames_processed}/{frame_count} frames processed.")
                eta_text.text(f"Estimated time remaining: {int(remaining_time)} seconds")
    
    cap.release()
    out.release()

    return out_file_name

def main():
    st.title("Climbing Video Analysis")
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])
    
    if uploaded_file is not None:
        st.video(uploaded_file)
        
        if st.button("Analyze Video"):
            with st.spinner('Processing video...'):
                processed_video_path = process_video(uploaded_file)
            
            st.success("Video processed successfully!")
            
            # Display the processed video
            st.video(processed_video_path)

    st.sidebar.header("Analysis Options")
    speed = st.sidebar.slider("Playback Speed", min_value=0.25, max_value=2.0, value=1.0, step=0.25)
    st.sidebar.checkbox("Enable Side-by-Side Comparison")

if __name__ == "__main__":
    main()
