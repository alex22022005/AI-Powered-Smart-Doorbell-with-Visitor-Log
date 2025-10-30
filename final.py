import cv2
from ultralytics import YOLO
from playsound import playsound
import threading
import time
import requests
import os
from datetime import datetime

# Configuration
MODEL_PATH = "best.pt"
DOORBELL_PATH = "doorbell.mp3"
UPLOAD_URL = "http://localhost:5000/upload"
IMAGE_SAVE_DIR = "captured_images"
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

# Load model
model = YOLO(MODEL_PATH)

# Detection state
processing_active = False

def doorbell_and_upload(frame):
    global processing_active
    try:
        # First capture the image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(IMAGE_SAVE_DIR, f"visitor_{timestamp}.jpg")
        cv2.imwrite(image_path, frame)
        print(f"Image captured: {image_path}")
        
        # Then upload to server
        with open(image_path, 'rb') as img_file:
            files = {'image': (f"visitor_{timestamp}.jpg", img_file, 'image/jpeg')}
            data = {
                'timestamp': timestamp,
                'detection_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            response = requests.post(UPLOAD_URL, files=files, data=data)
            print(f"Upload response: {response.text}")
            
        # Finally play doorbell sound
        playsound(DOORBELL_PATH)
        print("Doorbell sound played")
        
    except Exception as e:
        print(f"Error in processing: {e}")
    finally:
        processing_active = False

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
    annotated_frame = frame.copy()

    # Process only every nth frame when not already processing
    if frame_count % frame_skip == 0 and not processing_active:
        results = model.predict(source=frame, conf=0.5, save=False, verbose=False)

        person_detected = False
        for result in results:
            annotated_frame = result.plot()
            for cls in result.boxes.cls:
                if int(cls) == 0:  # 'person' class
                    person_detected = True

        if person_detected:
            print("Person detected! Starting capture and bell sequence...")
            processing_active = True
            # Start processing in separate thread
            process_thread = threading.Thread(target=doorbell_and_upload, args=(frame.copy(),))
            process_thread.start()

    # Show the latest annotated frame
    cv2.imshow("Smart Doorbell - Detection Running", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()