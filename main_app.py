import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tempfile
import os

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def process_video(video_file, output_name="output.mp4", playback_speed=1.0):
    with tempfile.NamedTemporaryFile(delete=False) as tfile:
        tfile.write(video_file.read())
        temp_video_path = tfile.name

    try:
        cap = cv2.VideoCapture(temp_video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer object
        out = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc(*'mp4v'), fps * playback_speed, (frame_width, frame_height))

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
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Write the frame to the output video
                out.write(frame)

        cap.release()
        out.release()
    finally:
        os.remove(temp_video_path)

def main():
    st.title("Climbing Video Analysis")

    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
        st.video(uploaded_file)

        if st.button("Analyze Video"):
            with st.spinner('Processing video...'):
                output_name = f"{os.path.splitext(uploaded_file.name)[0]}_processed.mp4"
                process_video(uploaded_file, output_name=output_name)
            
            st.success("Video processed successfully!")
            st.video(output_name)

    st.sidebar.header("Analysis Options")
    speed = st.sidebar.slider("Playback Speed", min_value=0.25, max_value=2.0, value=1.0, step=0.25)
    
    # Placeholder for side-by-side comparison
    st.sidebar.checkbox("Enable Side-by-Side Comparison")

if __name__ == "__main__":
    main()
