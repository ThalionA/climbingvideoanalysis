import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import tensorflow as tf

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def check_gpu():
    if tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None):
        return "GPU is available and will be used for processing."
    else:
        return "No GPU found. Processing will be done on CPU."

def process_frame(pose, frame_rgb):
    results = pose.process(frame_rgb)
    return results

def process_video(video_file, output_file_name, progress_queue):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Resize the frame for side-by-side display
    max_width = 640  # Set the maximum width for side-by-side display
    scale_factor = max_width / frame_width
    frame_width = max_width
    frame_height = int(frame_height * scale_factor)
    
    # Create video writer object
    out = cv2.VideoWriter(output_file_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    frames_processed = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize the frame
            frame = cv2.resize(frame, (frame_width, frame_height))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = process_frame(pose, frame_rgb)
            
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Convert RGB back to BGR before writing 
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)
            
            frames_processed += 1
            progress = frames_processed / frame_count
            progress_queue.put(progress)
    
    cap.release()
    out.release()
    progress_queue.put(1.0)  # Ensure the progress reaches 100%


def main():
    st.title("Dual Video Pose Analysis with GPU Acceleration")
    
    # Check for GPU availability
    gpu_message = check_gpu()
    st.write(gpu_message)
    
    # Upload two videos
    video_file1 = st.file_uploader("Choose the first video file", type=["mp4", "mov", "avi"], key="video1")
    video_file2 = st.file_uploader("Choose the second video file", type=["mp4", "mov", "avi"], key="video2")
    
    if video_file1 and video_file2:
        st.video(video_file1)
        st.video(video_file2)
        
        if st.button("Analyze Videos"):
            # Progress bars and queues for progress updates
            progress_queue1 = Queue()
            progress_queue2 = Queue()
            progress_bar1 = st.progress(0)
            progress_text1 = st.empty()
            progress_bar2 = st.progress(0)
            progress_text2 = st.empty()
            
            def update_progress(progress_queue, progress_bar, progress_text):
                while True:
                    progress = progress_queue.get()
                    if progress is None:
                        break
                    progress_bar.progress(progress)
                    progress_text.text(f"Processing... {int(progress * 100)}% completed.")
            
            with ThreadPoolExecutor(max_workers=2) as executor:
                future1 = executor.submit(process_video, video_file1, tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name, progress_queue1)
                future2 = executor.submit(process_video, video_file2, tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name, progress_queue2)
                
                executor.submit(update_progress, progress_queue1, progress_bar1, progress_text1)
                executor.submit(update_progress, progress_queue2, progress_bar2, progress_text2)
                
                future1.result()  # Wait for first video to finish
                future2.result()  # Wait for second video to finish

            # Close the progress queues
            progress_queue1.put(None)
            progress_queue2.put(None)
            
            st.success("Videos processed successfully!")
            
            # Display the processed videos side by side
            col1, col2 = st.columns(2)
            with col1:
                st.video(future1.result())
            with col2:
                st.video(future2.result())

    st.sidebar.header("Analysis Options")
    speed = st.sidebar.slider("Playback Speed", min_value=0.25, max_value=2.0, value=1.0, step=0.25)

if __name__ == "__main__":
    main()
