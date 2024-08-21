import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tempfile
import time

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def process_video(video_file):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer object
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    
    # Progress tracking
    start_time = time.time()
    frames_processed = 0
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the BGR image to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame with MediaPipe Pose
            results = pose.process(frame_rgb)

            # Draw pose landmarks on the frame
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Write the frame to the output video
            out.write(frame)
            
            # Update progress
            frames_processed += 1
            elapsed_time = time.time() - start_time
            estimated_total_time = (elapsed_time / frames_processed) * frame_count
            remaining_time = estimated_total_time - elapsed_time

            # Display progress with ETA
            st.progress(frames_processed / frame_count)
            st.write(f"Processing video... {frames_processed}/{frame_count} frames processed.")
            st.write(f"Estimated time remaining: {int(remaining_time)} seconds")

    cap.release()
    out.release()

def main():
    st.title("Climbing Video Analysis")

    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
        st.video(uploaded_file)

        if st.button("Analyze Video"):
            with st.spinner('Processing video...'):
                process_video(uploaded_file)
            
            st.success("Video processed successfully!")
            st.video('output.mp4')

    st.sidebar.header("Analysis Options")
    speed = st.sidebar.slider("Playback Speed", min_value=0.25, max_value=2.0, value=1.0, step=0.25)
    
    # Placeholder for side-by-side comparison
    st.sidebar.checkbox("Enable Side-by-Side Comparison")

if __name__ == "__main__":
    main()
