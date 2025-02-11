import cv2
import dlib
from scipy.spatial import distance
import time
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime
import os

# Function to compute Eye Aspect Ratio (EAR) for drowsiness detection
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to compute Mouth Aspect Ratio (MAR) for yawning detection
def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[10])
    B = distance.euclidean(mouth[4], mouth[8])
    C = distance.euclidean(mouth[0], mouth[6])
    mar = (A + B) / (2.0 * C)
    return mar

# Constants
EAR_THRESHOLD = 0.3
CONSEC_FRAMES = 48
MAR_THRESHOLD = 0.5

# Initialize counters
frame_counter = 0
drowsy_count = 0
yawning_count = 0
drowsy = False
yawning = False

# Video capture
cap = cv2.VideoCapture(0)
start_time = time.time()

# Variables for tracking
timestamps = []
drowsiness_status = []
yawning_status = []
image_captured = False
captured_image_path = None
total_screen_time = 0
screen_start_time = None
screen_on_time = None
screen_off_time = None

# Directory to save images
image_dir = "detection_images"
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        
        # Extract eye and mouth points
        left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
        right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
        mouth = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)]
        
        # EAR and MAR calculations
        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
        mar = mouth_aspect_ratio(mouth)

        # Drowsiness detection
        if ear < EAR_THRESHOLD:
            frame_counter += 1
            if frame_counter >= CONSEC_FRAMES and not drowsy:
                drowsy = True
                drowsy_count += 1
                if not image_captured:
                    captured_image_path = os.path.join(image_dir, f"drowsy_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg")
                    cv2.imwrite(captured_image_path, frame)
                    image_captured = True
        else:
            frame_counter = 0
            drowsy = False

        # Yawning detection
        if mar > MAR_THRESHOLD and not yawning:
            yawning = True
            yawning_count += 1
            if not image_captured:
                captured_image_path = os.path.join(image_dir, f"yawning_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg")
                cv2.imwrite(captured_image_path, frame)
                image_captured = True
        else:
            yawning = False

    # Tracking timestamps and counts
    elapsed_time = time.time() - start_time
    timestamps.append(elapsed_time)
    drowsiness_status.append(drowsy_count)
    yawning_status.append(yawning_count)

    # Display the webcam feed
    cv2.imshow('Webcam Feed', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

end_time = time.time()
total_runtime = end_time - start_time

# Generate Report
report_filename = f"Detection_Report_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pdf"
c = canvas.Canvas(report_filename, pagesize=letter)

# Title
c.setFont("Helvetica-Bold", 16)
c.drawString(50, 750, "Drowsiness and Yawning Detection Report")

# Report details
c.setFont("Helvetica", 12)
c.drawString(50, 720, f"Start Time: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}")
c.drawString(50, 700, f"End Time: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}")
c.drawString(50, 680, f"Total Runtime: {int(total_runtime)} seconds")
c.drawString(50, 660, f"Total Drowsiness Detected: {drowsy_count} times")
c.drawString(50, 640, f"Total Yawning Detected: {yawning_count} times")

# Add graph below text
plt.plot(timestamps, drowsiness_status, label='Drowsiness Count')
plt.plot(timestamps, yawning_status, label='Yawning Count')
plt.xlabel('Time (seconds)')
plt.ylabel('Count')
plt.title('Detection Over Time')
plt.legend()
graph_filename = 'Detection_Plot.png'
plt.savefig(graph_filename)
plt.close()
c.drawImage(graph_filename, 50, 350, width=500, height=250)

# Add passport-sized image
if captured_image_path:
    img = cv2.imread(captured_image_path)
    img_resized = cv2.resize(img, (150, 150))
    resized_image_path = os.path.join(image_dir, "resized_passport_image.jpg")
    cv2.imwrite(resized_image_path, img_resized)
    c.drawImage(resized_image_path, 450, 700, width=100, height=100)

# Save the PDF
c.save()

# Cleanup
cap.release()
cv2.destroyAllWindows()

print(f"Report saved as {report_filename}")
