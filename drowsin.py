import cv2
import dlib
import time
from scipy.spatial import distance
from fpdf import FPDF
import matplotlib.pyplot as plt
import numpy as np

# Load face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Thresholds
EYE_ASPECT_RATIO_THRESHOLD = 0.25
MOUTH_ASPECT_RATIO_THRESHOLD = 0.75
DROWSY_CONSEC_FRAMES = 1 * 30  # Assuming 30 FPS (15 seconds)
YAWN_CONSEC_FRAMES = 1  # Number of consecutive frames for yawning detection

# Variables
drowsy_count = 0
yawn_count = 0
sleep_duration = 0
is_drowsy = False
drowsy_start_time = None
drowsy_frame_count = 0
yawn_frame_count = 0
image_captured = False
passport_image = None
drowsy_data = []
yawn_data = []
sleep_data = []
time_data = []

# Calculate aspect ratio
def calculate_aspect_ratio(landmarks, points):
    A = distance.euclidean(landmarks[points[1]], landmarks[points[5]])
    B = distance.euclidean(landmarks[points[2]], landmarks[points[4]])
    C = distance.euclidean(landmarks[points[0]], landmarks[points[3]])
    return (A + B) / (2.0 * C)

# Process video stream
cap = cv2.VideoCapture(0)
start_time = time.strftime("%Y-%m-%d %H:%M:%S")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        landmarks_points = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)]

        # EAR calculation for drowsiness
        left_eye_points = [36, 37, 38, 39, 40, 41]
        right_eye_points = [42, 43, 44, 45, 46, 47]
        left_ear = calculate_aspect_ratio(landmarks_points, left_eye_points)
        right_ear = calculate_aspect_ratio(landmarks_points, right_eye_points)
        ear = (left_ear + right_ear) / 2.0

        # MAR calculation for yawning
        mouth_points = [60, 61, 62, 63, 64, 65, 66, 67]
        mar = calculate_aspect_ratio(landmarks_points, mouth_points)

        # Detect drowsiness
        if ear < EYE_ASPECT_RATIO_THRESHOLD:
            drowsy_frame_count += 1
            if drowsy_frame_count >= DROWSY_CONSEC_FRAMES:
                drowsy_count += 1
                if not is_drowsy:
                    drowsy_start_time = time.time()
                    is_drowsy = True
        else:
            if is_drowsy:
                sleep_duration += (time.time() - drowsy_start_time) / 60  # Convert to minutes
                is_drowsy = False
            drowsy_frame_count = 0  # Reset counter

        # Detect yawning
        if mar > MOUTH_ASPECT_RATIO_THRESHOLD:
            yawn_frame_count += 1
            if yawn_frame_count >= YAWN_CONSEC_FRAMES:
                yawn_count += 1
                yawn_frame_count = 0
        else:
            yawn_frame_count = 0

        # Capture passport-size image
        if not image_captured:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            passport_image = frame[y:y+h, x:x+w]
            passport_image = cv2.resize(passport_image, (150, 200))
            image_captured = True

        # Store data for graph plotting
        drowsy_data.append(drowsy_count)
        yawn_data.append(yawn_count)
        sleep_data.append(round(sleep_duration, 2))
        time_data.append(time.strftime("%H:%M:%S"))

        # Draw annotations
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
        cv2.putText(frame, f"Drowsy: {drowsy_count} | Yawns: {yawn_count} | Sleep: {round(sleep_duration, 2)} mins",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Drowsiness and Yawning Detection", frame)

    # Break loop on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Generate report
if passport_image is not None:
    cv2.imwrite("passport_image.jpg", passport_image)

# Plot graph
plt.figure(figsize=(10, 6))
plt.plot(time_data, drowsy_data, label='Drowsy Count', color='blue', marker='o')
plt.plot(time_data, yawn_data, label='Yawns Count', color='green', marker='o')
plt.plot(time_data, sleep_data, label='Sleep Duration (mins)', color='red', marker='o')
plt.xlabel('Time')
plt.ylabel('Count / Duration')
plt.title('Real-Time Drowsiness, Yawning and Sleep Duration')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

# Save the graph as an image
graph_image_path = "drowsiness_graph.png"
plt.savefig(graph_image_path)
plt.close()

# Create PDF report
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.cell(200, 10, txt="Drowsiness and Yawning Detection Report", ln=True, align='C')
pdf.ln(10)
pdf.cell(200, 10, txt=f"Start Time: {start_time}", ln=True)
pdf.cell(200, 10, txt=f"Drowsiness Count: {drowsy_count}", ln=True)
pdf.cell(200, 10, txt=f"Yawning Count: {yawn_count}", ln=True)
pdf.cell(200, 10, txt=f"Total Sleep Duration: {round(sleep_duration, 2)} minutes", ln=True)
if passport_image is not None:
    pdf.image("passport_image.jpg", x=10, y=80, w=50, h=70)
# Add the graph to the PDF
pdf.ln(10)
pdf.image(graph_image_path, x=10, y=160, w=180, h=120)

# Save the PDF
pdf.output("drowsiness_report.pdf")
print("Report saved as 'drowsiness_report.pdf'")
