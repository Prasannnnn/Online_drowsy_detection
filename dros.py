import cv2
import dlib
from scipy.spatial import distance
import time
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime
import os
import numpy as np

# Load pre-trained classifiers for face and eyes
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Initialize dlib's face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Function to compute Eye Aspect Ratio (EAR) for drowsiness detection
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Yawning detection function
def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[10])
    B = distance.euclidean(mouth[4], mouth[8])
    C = distance.euclidean(mouth[0], mouth[6])
    mar = (A + B) / (2.0 * C)
    return mar

# Drowsiness detection threshold
EAR_THRESHOLD = 0.3
CONSEC_FRAMES = 48
MAR_THRESHOLD = 0.8

# Initialize counters
frame_counter = 0
drowsy_count = 0
yawn_count = 0
drowsy = False
yawning = False

# Initialize video capture
cap = cv2.VideoCapture(0)
start_time = time.time()

# To keep track of runtime and detections
timestamps = []
drowsiness_status = []
yawning_status = []

# Real-time time tracking
start_real_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
screen_on_time = None
screen_off_time = None
screen_on = False

# Directory to save image
image_dir = "detection_images"
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

# Variable to track if image has been captured
image_captured = False
captured_image_path = None

# Apply smoothing to EAR and MAR
def smooth(value, prev_value, alpha=0.5):
    return alpha * value + (1 - alpha) * prev_value

# Initialize previous values for smoothing
prev_ear = 0
prev_mar = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        
        # Get coordinates for eyes
        left_eye = []
        right_eye = []
        for i in range(36, 42):
            left_eye.append((landmarks.part(i).x, landmarks.part(i).y))
        for i in range(42, 48):
            right_eye.append((landmarks.part(i).x, landmarks.part(i).y))

        # Get coordinates for mouth
        mouth = []
        for i in range(48, 68):
            mouth.append((landmarks.part(i).x, landmarks.part(i).y))

        # Calculate EAR for both eyes (drowsiness detection)
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Calculate MAR for mouth (yawning detection)
        mar = mouth_aspect_ratio(mouth)

        # Apply smoothing to EAR and MAR
        smoothed_ear = smooth(ear, prev_ear)
        smoothed_mar = smooth(mar, prev_mar)
        prev_ear = smoothed_ear
        prev_mar = smoothed_mar

        # Check for drowsiness (eyes closed)
        if smoothed_ear < EAR_THRESHOLD:
            frame_counter += 1
            if frame_counter >= CONSEC_FRAMES:
                if not drowsy:
                    drowsy = True
                    drowsy_count += 1
                    cv2.putText(frame, "DROWSY", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    print("Warning! Drowsiness Detected.")
                    if not image_captured:
                        captured_image_path = os.path.join(image_dir, f"drowsy_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg")
                        cv2.imwrite(captured_image_path, frame)
                        image_captured = True
                    screen_on = True
                    screen_on_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        else:
            frame_counter = 0
            drowsy = False

        # Check for yawning (mouth wide open)
        if smoothed_mar > MAR_THRESHOLD:
            if not yawning:
                yawning = True
                yawn_count += 1
                cv2.putText(frame, "YAWNING", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                print("Yawning Detected.")
        else:
            yawning = False

        # Draw bounding box around face and display EAR/MAR
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Track runtime
    elapsed_time = time.time() - start_time
    timestamps.append(elapsed_time)
    drowsiness_status.append(drowsy_count)
    yawning_status.append(yawn_count)

    # Display the resulting frame with time
    current_real_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, f"Time: {current_real_time}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('Drowsiness and Yawning Detection', frame)

    # Press 'q' to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        if screen_on:
            screen_off_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        break

# Capture end time and total runtime
end_time = time.time()
total_runtime = end_time - start_time

# Create a report with the data
report_filename = f"Detection_Report_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pdf"
c = canvas.Canvas(report_filename, pagesize=letter)

# Add basic info to the report
c.drawString(100, 750, f"Detection Report ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
c.drawString(100, 730, f"Screen On Time: {screen_on_time}")
c.drawString(100, 710, f"Screen Off Time: {screen_off_time}")
c.drawString(100, 690, f"Total Runtime: {int(total_runtime)} seconds")
c.drawString(100, 670, f"Total Drowsiness Detected: {drowsy_count} times")
c.drawString(100, 650, f"Total Yawning Detected: {yawn_count} times")

# Save plot as a graph
plt.figure(figsize=(8, 6))
plt.plot(timestamps, drowsiness_status, label='Drowsiness Count', color='red')
plt.plot(timestamps, yawning_status, label='Yawning Count', color='green')
plt.xlabel('Time (seconds)')
plt.ylabel('Detection Count')
plt.title('Drowsiness and Yawning Detection Over Time')
plt.legend()
graph_filename = 'Detection_Plot.png'
plt.savefig(graph_filename)
plt.close()

# Insert the graph into the PDF below the text
c.drawImage(graph_filename, 100, 300, width=400, height=300)

# Insert the last captured image (passport size photo in the corner)
if captured_image_path:
    # Resize the captured image to 2x2 inches (passport size)
    img = cv2.imread(captured_image_path)
    img_resized = cv2.resize(img, (150, 150))  # 150x150 pixels (approx. 2x2 inches)
    resized_image_path = "resized_passport_image.jpg"
    cv2.imwrite(resized_image_path, img_resized)
    
    # Add the resized image to the PDF (top-right corner)
    c.drawImage(resized_image_path, 400, 500, width=150, height=150)

# Save the PDF report
c.save()

# Release resources
cap.release()
cv2.destroyAllWindows()

print(f"Report saved as {report_filename}")
