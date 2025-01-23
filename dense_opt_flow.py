import cv2
import numpy as np

# Open the video file
cap = cv2.VideoCapture("videos/video.mp4")

# Check if the video file opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Read the first frame
ret, first_frame = cap.read()
if not ret:
    print("Error: Could not read the first frame.")
    exit()

# Convert the first frame to grayscale
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Create a mask image for drawing (same size as the frame, but with 3 channels)
mask = np.zeros_like(first_frame)
mask[..., 1] = 255  # Set saturation to maximum

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output/dense_opt_flow.avi', fourcc, 20.0, (first_frame.shape[1], first_frame.shape[0]))

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video file reached or error reading frame.")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate dense optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Compute the magnitude and angle of the flow
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Set the hue and value of the mask
    mask[..., 0] = angle * 180 / np.pi / 2
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

    # Write the frame to the output video
    out.write(rgb)

    # Update the previous frame
    prev_gray = gray

    # Display the frame (optional)
    cv2.imshow('Dense Optical Flow', rgb)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()