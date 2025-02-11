import cv2
import dlib
import numpy as np
import time
import csv
import pickle
from scipy.spatial import distance as dist
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from matplotlib import pyplot as plt

# Load face recognition model
with open("models/face_encodings_model.pkl", "rb") as file:
    model_data = pickle.load(file)
    known_face_encodings = np.array(model_data["encodings"], dtype=np.float64)  # Ensure numeric type
    known_face_names = model_data["names"]

# Dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Constants for drowsiness and yawning detection
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 48  # Number of frames that the EAR needs to be below threshold to trigger alert
MAR_THRESH = 0.7
HEAD_POSE_YAW_THRESH = 15
HEAD_POSE_PITCH_THRESH = 10

# Initialize counters
COUNTER = 0
TOTAL_YAWNS = 0
TOTAL_DROWSINESS = 0
START_TIME = time.time()
EVENT_LOG = []
MATCHED = 0
UNMATCHED = 0
MATCHED_NAMES = []

# Functions for EAR and MAR
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[3], mouth[8])
    B = dist.euclidean(mouth[2], mouth[10])
    C = dist.euclidean(mouth[0], mouth[6])
    mar = (A + B) / (2.0 * C)
    return mar


face_rec_model = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")
# Initialize webcam
cap = cv2.VideoCapture(0)

# Start time of the session
session_start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # For face encoding
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # Detect eyes and mouth landmarks
        left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
        right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
        mouth = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)])

        # Calculate EAR and MAR
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        mar = mouth_aspect_ratio(mouth)

        # Drowsiness detection (EAR below threshold)
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                # Trigger drowsiness alert
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                TOTAL_DROWSINESS += 1
                EVENT_LOG.append(["Drowsy", time.time() - session_start_time])
        else:
            COUNTER = 0  # Reset counter if EAR is above threshold

        # Yawning detection
        if mar > MAR_THRESH:
            TOTAL_YAWNS += 1
            EVENT_LOG.append(["Yawning", time.time() - session_start_time])
            cv2.putText(frame, "YAWNING DETECTED", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Face recognition
        shape = predictor(gray, face)  # Detect facial landmarks
        face_descriptor = np.array(face_rec_model.compute_face_descriptor(frame, shape))

        # Calculate distances to known encodings
        distances = np.linalg.norm(known_face_encodings - face_descriptor, axis=1)
        min_distance = min(distances)

        if min_distance < 0.6:  # Threshold for match
            match_index = np.argmin(distances)
            name = known_face_names[match_index]
            MATCHED += 1
            MATCHED_NAMES.append(name)
            cv2.putText(frame, f"Matched: {name}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        else:
            UNMATCHED += 1
            cv2.putText(frame, "Unmatched", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow("Frame", frame)

    # Break loop on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Session End Time
session_end_time = time.time()
total_duration = session_end_time - session_start_time

# Save event log to CSV
with open("drowsiness_report.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Event", "Timestamp (s)"])
    writer.writerows(EVENT_LOG)

# Generate event graph
plt.bar(["Drowsy", "Yawning", "Normal"], [TOTAL_DROWSINESS, TOTAL_YAWNS, len(EVENT_LOG) - (TOTAL_DROWSINESS + TOTAL_YAWNS)], color=["red", "green", "blue"])
plt.title("Event Counts")
plt.savefig("event_graph.png")

# Generate recognition graph
matched_percentage = (MATCHED / (MATCHED + UNMATCHED)) * 100
unmatched_percentage = (UNMATCHED / (MATCHED + UNMATCHED)) * 100
plt.bar(["Matched", "Unmatched"], [matched_percentage, unmatched_percentage], color=["blue", "orange"])
plt.title("Face Recognition Match Percentage")
plt.savefig("recognition_graph.png")

# Generate PDF report
c = canvas.Canvas("drowsiness_report.pdf", pagesize=letter)
c.drawString(30, 750, f"Session Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(session_start_time))}")
c.drawString(30, 730, f"Session End Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(session_end_time))}")
c.drawString(30, 710, f"Total Session Duration: {total_duration:.2f} seconds")
c.drawString(30, 690, f"Total Drowsy Events: {TOTAL_DROWSINESS}")
c.drawString(30, 670, f"Total Yawning Events: {TOTAL_YAWNS}")
c.drawString(30, 650, f"Matched Faces: {MATCHED}")
c.drawString(30, 630, f"Unmatched Faces: {UNMATCHED}")

# Include the event graph and recognition graph in the report
c.drawImage("event_graph.png", 30, 400, width=550, height=250)
c.drawImage("recognition_graph.png", 30, 150, width=550, height=250)

# Save photo of the person (Capture a frame with a matched face)
if MATCHED > 0:
    cv2.imwrite("person_photo.jpg", frame)
    c.drawImage("person_photo.jpg", 300, 550, width=200, height=200)

c.save()

cap.release()
cv2.destroyAllWindows()
