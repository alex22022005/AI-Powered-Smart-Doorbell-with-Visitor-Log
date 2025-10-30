import cv2
from ultralytics import YOLO
from playsound import playsound
import threading
import time

# Load model and setup
model = YOLO("best.pt")
DOORBELL_PATH = "doorbell.mp3"

# Doorbell threading logic
doorbell_active = False  # Flag to avoid multiple triggers

def play_doorbell():
    global doorbell_active
    playsound(DOORBELL_PATH)
    doorbell_active = False  # Reset flag once done

# Open webcam
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_skip = 1
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame_count += 1
    annotated_frame = frame  # Default to plain frame

    # Process only every nth frame
    if frame_count % frame_skip == 0:
        results = model.predict(source=frame, conf=0.5, save=False, verbose=False)

        person_detected = False
        for result in results:
            annotated_frame = result.plot()  # Annotate frame
            for cls in result.boxes.cls:
                if int(cls) == 0:  # 'person' class
                    person_detected = True

        # Trigger doorbell if not already ringing
        if person_detected and not doorbell_active:
            print("Person detected! Triggering doorbell...")
            doorbell_active = True
            bell_thread = threading.Thread(target=play_doorbell)
            bell_thread.start()

    # Show the latest annotated frame (even if it wasn't updated this round)
    cv2.imshow("YOLOv8 Person Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
