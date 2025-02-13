import streamlit as st
import tempfile
import os
import cv2
import sys
from scf2.tracking import process_video
from scf2.object_detection import ObjectDetection
from scf2.obj_det import detect_objects
from scf2.retina_net_w import detect_objects_retina

# Add the scf2 directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scf2'))

st.title("Object Tracking and Detection Application")

# Upload video file
uploaded_video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_video_file is not None:
    # Save the uploaded video file to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video_file.read())
    
    # Define paths for input and output videos
    input_video_path = tfile.name
    output_video_path = "output/tracking_obj_video.mp4"
    
    # Process the video
    process_video(input_video_path, output_video_path)
    
    # Display the processed video
    st.video(os.path.normpath(output_video_path))
    
    with open(output_video_path, "rb") as file:
        btn = st.download_button(
            label="Download Processed Video",
            data=file,
            file_name=uploaded_video_file.name,
            mime="video/mp4"
        )

# Upload image file
uploaded_image_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])

if uploaded_image_file is not None:
    # Save the uploaded image file to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    tfile.write(uploaded_image_file.read())
    
    # Define path for input image
    input_image_path = tfile.name
    
    # Select the model for object detection
    model_option = st.selectbox("Select the model for object detection", ("YOLOv3", "RetinaNet"))

    if model_option == "YOLOv3":
        # Perform object detection using obj_det.py
        image, class_ids, confidences, boxes = detect_objects(input_image_path)
    elif model_option == "RetinaNet":
        # Perform object detection using retina_net_w.py
        image, class_ids, confidences, boxes = detect_objects_retina(input_image_path)
    
    # Draw bounding boxes on the image
    for (class_id, confidence, box) in zip(class_ids, confidences, boxes):
        (x, y, w, h) = box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = f"{class_id}: {confidence:.2f}"
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Save the output image
    output_image_path = "output/detected_image.jpg"
    cv2.imwrite(output_image_path, image)
    
    # Display the processed image
    st.image(output_image_path)
    
    with open(output_image_path, "rb") as file:
        btn = st.download_button(
            label="Download Processed Image",
            data=file,
            file_name="detected_image.jpg",
            mime="image/jpeg"
        )
