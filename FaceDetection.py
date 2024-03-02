# Made by Jim Grysmpolakis.
# Please do not claim as yours.

# Face Detection Program

# The Face Detection with OpenCV script is a Python application that utilizes the OpenCV library for real-time
# face detection through a computer's webcam. The script captures video from the default camera, converts 
# frames to grayscale, and applies a Haar cascade classifier to identify faces. 

# Detected faces are highlighted with rectangles in the live video stream. With adjustable parameters for detection, 
# the script provides a simple and interactive way to explore basic face recognition techniques. 
# This lightweight and easy-to-understand tool serves as a starting point for understanding computer vision concepts
# and experimenting with facial detection in real-time scenarios.

# Import OpenCV library.
import cv2

# Initialize the camera (0 is the default camera index).
camera = cv2.VideoCapture(0)

# Load the Haar cascade classifier for face detection.
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Function to detect faces and draw bounding boxes.
def detect_bounding_box(vid):
    # Convert the video frame to grayscale.
    gray_image = cv2.cvtColor(vid, cv2.COLOR_RGB2GRAY)

    # Detect faces in the grayscale image.
    faces = face_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    # Draw rectangles around detected faces.
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (250, 250, 0), 4)

    return faces

# Main loop for real-time face detection.
while True:
    # Read a frame from the camera.
    result, video_frame = camera.read()

    # Detect faces and draw bounding boxes.
    detect_bounding_box(video_frame)

    # Display the video frame with detected faces.
    cv2.imshow("Face Recognition", video_frame)

    # Check for the 'Esc' key to exit the loop.
    k = cv2.waitKey(1)
    if k % 256 == 27:
        print("Closing...")
        break

# Release the camera and close OpenCV windows.
camera.release()
cv2.destroyAllWindows()
