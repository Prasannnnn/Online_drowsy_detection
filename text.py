import cv2
import dlib
import numpy as np
import time
import csv
from scipy.spatial import distance as dist

# Initialize Dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models\shape_predictor_68_face_landmarks.dat")

# Constants
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 48
MAR_THRESH = 0.7
HEAD_POSE_YAW_THRESH = 15  # Degrees
HEAD_POSE_PITCH_THRESH = 10

# Initialize counters
COUNTER = 0
TOTAL_YAWNS = 0
DROWSINESS_EVENTS = 0
START_TIME = time.time()
EVENT_LOG = []

# Mouth aspect ratio calculation
def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[3], mouth[8])  # Vertical
    B = dist.euclidean(mouth[2], mouth[10])  # Vertical
    C = dist.euclidean(mouth[0], mouth[6])  # Horizontal
    mar = (A + B) / (2.0 * C)
    return mar

# Eye aspect ratio calculation
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])  # Vertical
    B = dist.euclidean(eye[2], eye[4])  # Vertical
    C = dist.euclidean(eye[0], eye[3])  # Horizontal
    ear = (A + B) / (2.0 * C)
    return ear

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        # Detect landmarks
        landmarks = predictor(gray, face)
        
        # Get coordinates for eyes and mouth
        left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
        right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
        mouth = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)])

        # Calculate EAR and MAR
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        mar = mouth_aspect_ratio(mouth)

        # Detect drowsiness
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                DROWSINESS_EVENTS += 1
                EVENT_LOG.append(["Drowsy", time.time() - START_TIME])
        else:
            COUNTER = 0

        # Detect yawning
        if mar > MAR_THRESH:
            TOTAL_YAWNS += 1
            EVENT_LOG.append(["Yawning", time.time() - START_TIME])
            cv2.putText(frame, "YAWNING DETECTED", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Head pose estimation
        try:
            image_points = np.array([
                (landmarks.part(30).x, landmarks.part(30).y),  # Nose tip
                (landmarks.part(8).x, landmarks.part(8).y),    # Chin
                (landmarks.part(36).x, landmarks.part(36).y),  # Left eye left corner
                (landmarks.part(45).x, landmarks.part(45).y),  # Right eye right corner
                (landmarks.part(48).x, landmarks.part(48).y),  # Left Mouth corner
                (landmarks.part(54).x, landmarks.part(54).y)   # Right Mouth corner
            ], dtype="double")

            model_points = np.array([
                (0.0, 0.0, 0.0),            # Nose tip
                (0.0, -330.0, -65.0),       # Chin
                (-225.0, 170.0, -135.0),    # Left eye left corner
                (225.0, 170.0, -135.0),     # Right eye right corner
                (-150.0, -150.0, -125.0),   # Left Mouth corner
                (150.0, -150.0, -125.0)     # Right Mouth corner
            ])

            focal_length = frame.shape[1]
            center = (frame.shape[1] / 2, frame.shape[0] / 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype="double")

            dist_coeffs = np.zeros((4, 1))

            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs
            )

            if success:
                rmat, _ = cv2.Rodrigues(rotation_vector)
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                yaw, pitch = angles[1], angles[0]

                if abs(yaw) > HEAD_POSE_YAW_THRESH or abs(pitch) > HEAD_POSE_PITCH_THRESH:
                    cv2.putText(frame, "WANDERING OFF", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    EVENT_LOG.append(["Wandering", time.time() - START_TIME])

        except IndexError:
            pass  # Skip head pose if landmarks are incomplete

    # Display the frame
    cv2.imshow("Frame", frame)

    # Break loop on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save event log to CSV
END_TIME = time.time()
RUNTIME = END_TIME - START_TIME
with open("drowsiness_report.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Event", "Timestamp (s)"])
    writer.writerows(EVENT_LOG)
    writer.writerow(["Total Runtime (s)", RUNTIME])

cap.release()
cv2.destroyAllWindows()
