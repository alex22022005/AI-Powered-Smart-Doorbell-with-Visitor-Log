import cv2
from ultralytics import YOLO
from playsound import playsound
import threading
import time
import requests
import os
from datetime import datetime
import json

# Configuration
MODEL_PATH = "best.pt"
DOORBELL_PATH = "doorbell.mp3"
UPLOAD_URL = "http://localhost:5000/upload"  # Change to your server URL
IMAGE_SAVE_DIR = "captured_images"
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

# Load model
model = YOLO(MODEL_PATH)

# Doorbell and image capture logic
doorbell_active = False  # Flag to avoid multiple triggers

def play_doorbell_and_capture(frame):
    global doorbell_active
    try:
        # Play doorbell sound
        playsound(DOORBELL_PATH)
        
        # Capture and save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(IMAGE_SAVE_DIR, f"visitor_{timestamp}.jpg")
        cv2.imwrite(image_path, frame)
        print(f"Image saved: {image_path}")
        
        # Send to web server
        with open(image_path, 'rb') as img_file:
            files = {'image': (f"visitor_{timestamp}.jpg", img_file, 'image/jpeg')}
            data = {'timestamp': timestamp}
            response = requests.post(UPLOAD_URL, files=files, data=data)
            print(f"Upload response: {response.text}")
            
    except Exception as e:
        print(f"Error in doorbell/capture: {e}")
    finally:
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
            print("Person detected! Triggering doorbell and capturing image...")
            doorbell_active = True
            # Make a copy of the frame for the thread to use
            frame_copy = frame.copy()
            bell_thread = threading.Thread(target=play_doorbell_and_capture, args=(frame_copy,))
            bell_thread.start()

    # Show the latest annotated frame (even if it wasn't updated this round)
    cv2.imshow("YOLOv8 Person Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()