import cv2
from ultralytics import YOLO
from playsound import playsound
import time

# Load the trained YOLOv8 model
model = YOLO("best.pt")  # Replace with your model path

# Load doorbell sound file
DOORBELL_PATH = "doorbell.mp3"  # Replace with your actual audio file

# Open webcam (0 = default cam)
cap = cv2.VideoCapture(1)

# Loop for continuous detection
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run detection
    results = model.predict(source=frame, conf=0.5, save=False, verbose=False)

    # Check if any person is detected
    for result in results:
        for cls in result.boxes.cls:
            if int(cls) == 0:  # COCO class 0 = person
                print("Person detected!")
                # Play doorbell sound
                playsound(DOORBELL_PATH)

                # Skip detection until sound is done
                print("Resuming detection...")
                break

    # Display the frame (optional)
    cv2.imshow("YOLOv8 Person Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
