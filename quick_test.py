import cv2
import sys

print("OpenCV version:", cv2.__version__)
print("Attempting to open webcam...")

cap = cv2.VideoCapture(0)
print("VideoCapture created")

if cap.isOpened():
    print("SUCCESS: Webcam is accessible!")
    ret, frame = cap.read()
    if ret:
        print(f"Frame captured successfully: {frame.shape}")
    else:
        print("ERROR: Could not read frame")
    cap.release()
else:
    print("ERROR: Could not open webcam (camera index 0)")
    print("This could mean:")
    print("  - No webcam is connected")
    print("  - Another program is using the webcam")
    print("  - Camera permissions are not granted")

print("Test complete")
