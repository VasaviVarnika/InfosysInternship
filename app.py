import streamlit as st
import tempfile
import os
from tracking import process_video

st.title("Object Tracking Application")

# Upload video file
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    # Define paths for input and output videos
    input_video_path = tfile.name
    output_video_path = "output/tracking_obj_video.avi"
    
    # Process the video
    process_video(input_video_path, output_video_path)
    
    # Display the processed video
    st.video(output_video_path)
