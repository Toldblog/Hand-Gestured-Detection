import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
from utils import draw_landmarks_on_image

# Path to the hand landmark model
model_path = '/Users/apple/Documents/AI/Projects/Advanced_Projects/HandGesture/hand_landmarker.task'

# Create a HandLandmarker object
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# Initialize webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from webcam.")
        break
    
    image = mp.Image(
    image_format=mp.ImageFormat.SRGB, data=np.asarray(frame))
    
    # Detect hand landmarks from the input image
    detection_result = detector.detect(image)
    
    # Process the classification result and visualize it
    frame = draw_landmarks_on_image(image.numpy_view(), detection_result)

    cv2.imshow('frame',frame)
    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
