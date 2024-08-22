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

def process_video(video_file, progress_container):
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
                    if result.pose_landmarks:
                        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Convert RGB back to BGR before writing 
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame)
                
                frames_processed += 1
                progress = frames_processed / frame_count
                progress_container.progress(progress)
    
    cap.release()
    out.release()

    return out_file_name

def main():
    st.title("Side-by-Side Climbing Video Analysis")
    uploaded_file1 = st.file_uploader("Choose the first video file", type=["mp4", "mov", "avi"], key="file1")
    uploaded_file2 = st.file_uploader("Choose the second video file", type=["mp4", "mov", "avi"], key="file2")
    
    if uploaded_file1 is not None and uploaded_file2 is not None:
        st.video(uploaded_file1)
        st.video(uploaded_file2)
        
        if st.button("Analyze Videos"):
            with st.spinner('Processing videos...'):
                col1, col2 = st.columns(2)
                with col1:
                    st.text("Processing first video")
                    progress_container1 = st.empty()
                    processed_video_path1 = process_video(uploaded_file1, progress_container1)
                    st.video(processed_video_path1)
                
                with col2:
                    st.text("Processing second video")
                    progress_container2 = st.empty()
                    processed_video_path2 = process_video(uploaded_file2, progress_container2)
                    st.video(processed_video_path2)

            st.success("Videos processed successfully!")

    st.sidebar.header("Analysis Options")
    speed = st.sidebar.slider("Playback Speed", min_value=0.25, max_value=2.0, value=1.0, step=0.25)
    st.sidebar.checkbox("Enable Side-by-Side Comparison", value=True)

if __name__ == "__main__":
    main()
