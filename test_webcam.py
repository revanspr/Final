"""
Simple webcam test to check if camera is accessible
"""
import cv2
import sys

print("Testing webcam access...")
print("=" * 50)

# Try to open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam")
    print("Possible issues:")
    print("  1. No webcam connected")
    print("  2. Webcam is being used by another application")
    print("  3. Permission denied")
    sys.exit(1)

print("SUCCESS: Webcam opened successfully!")

# Try to read a frame
ret, frame = cap.read()

if not ret:
    print("ERROR: Could not read frame from webcam")
    cap.release()
    sys.exit(1)

print(f"SUCCESS: Frame captured - Resolution: {frame.shape[1]}x{frame.shape[0]}")

# Test face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if face_cascade.empty():
    print("ERROR: Could not load face cascade classifier")
    cap.release()
    sys.exit(1)

print("SUCCESS: Face detector loaded")
print("=" * 50)
print("\nStarting live test...")
print("A window should appear showing your webcam feed")
print("Press 'q' to quit")
print("=" * 50)

frame_count = 0
face_detected = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Lost connection to webcam")
        break

    # Detect faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20))

    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        if not face_detected:
            print("FACE DETECTED!")
            face_detected = True

    # Add text to frame
    cv2.putText(frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Faces: {len(faces)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "Press 'q' to quit", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Show frame
    cv2.imshow('Webcam Test', frame)

    # Print status every 30 frames
    if frame_count % 30 == 0:
        print(f"Running... Frame {frame_count}, Faces detected: {len(faces)}")

    frame_count += 1

    # Check for 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\nQuitting...")
        break

cap.release()
cv2.destroyAllWindows()
print("Test complete!")
