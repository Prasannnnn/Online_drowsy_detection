import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import time
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Constants for thresholds
EYE_AR_THRESHOLD = 0.25
EYE_AR_CONSEC_FRAMES = 20
MOUTH_AR_THRESHOLD = 0.65

# Load Dlib's face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Eye and mouth landmarks indices
LEFT_EYE = list(range(42, 48))
RIGHT_EYE = list(range(36, 42))
MOUTH = list(range(48, 68))

# Calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Calculate Mouth Aspect Ratio (MAR)
def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[13], mouth[19])  # Vertical distance (outer lips)
    B = dist.euclidean(mouth[14], mouth[18])  # Vertical distance (inner lips)
    C = dist.euclidean(mouth[12], mouth[16])  # Horizontal distance
    mar = (A + B) / (2.0 * C)
    return mar

# Initialize variables
eye_blink_counter = 0
drowsy_alert = False
yawn_alert = False
start_time = time.time()
events = []  # To store event logs

# Open webcam
cap = cv2.VideoCapture(0)

print("Drowsiness detection started. Press 'q' to quit and save the report.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        points = np.array([[p.x, p.y] for p in landmarks.parts()])

        # Extract eyes and mouth landmarks
        left_eye = points[LEFT_EYE]
        right_eye = points[RIGHT_EYE]
        mouth = points[MOUTH]

        # Compute EAR and MAR
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        mar = mouth_aspect_ratio(mouth)

        # Check EAR for drowsiness
        if avg_ear < EYE_AR_THRESHOLD:
            eye_blink_counter += 1
            if eye_blink_counter >= EYE_AR_CONSEC_FRAMES:
                drowsy_alert = True
                events.append(
                    {
                        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Event": "Drowsiness Detected",
                    }
                )
        else:
            eye_blink_counter = 0
            drowsy_alert = False

        # Check MAR for yawning
        if mar > MOUTH_AR_THRESHOLD:
            yawn_alert = True
            events.append(
                {
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Event": "Yawning Detected",
                }
            )
        else:
            yawn_alert = False

        # Display drowsiness and yawning alerts
        if drowsy_alert:
            cv2.putText(frame, "Drowsiness Detected!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if yawn_alert:
            cv2.putText(frame, "Yawning Detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw landmarks and bounding box
        for point in points:
            cv2.circle(frame, tuple(point), 2, (255, 0, 0), -1)

    elapsed_time = time.time() - start_time
    cv2.putText(frame, f"Session Time: {int(elapsed_time)}s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Save session data
cap.release()
cv2.destroyAllWindows()

# Convert events to DataFrame and save to CSV
df = pd.DataFrame(events)
csv_filename = "drowsiness_report.csv"
df.to_csv(csv_filename, index=False)

# Plot EAR and MAR over time
df["Event"].value_counts().plot(kind="bar", color=["red", "green"], title="Event Summary")
plt.xlabel("Event Type")
plt.ylabel("Frequency")
plt.savefig("event_summary.png")
plt.show()

print(f"Session ended. Report saved to '{csv_filename}'. Visualization saved as 'event_summary.png'.")
