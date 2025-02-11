import cv2
import face_recognition
import pickle
import numpy as np
import time
import os
from datetime import datetime
import matplotlib.pyplot as plt

# Load the face encodings model
with open("models/face_encodings_model.pkl", "rb") as f:
    data = pickle.load(f)
known_face_encodings = data["encodings"]
known_face_names = data["names"]

# Initialize Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + r'C:\Users\ADMIN\Desktop\AIML-Projects\Drwosiness-Detection\haarcascade_frontalface_default.xml')
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + r'C:\Users\ADMIN\Desktop\AIML-Projects\Drwosiness-Detection\haarcascade_mcs_mouth.xml')
if mouth_cascade.empty() or face_cascade.empty():
    raise ValueError("Cascade files not loaded. Check file paths.")

# Initialize variables
yawning_count = 0
drowsy_count = 0
normal_count = 0
start_time = None
end_time = None
detection_data = []  # For report generation

# Real-time video capture
cap = cv2.VideoCapture(0)
start_time = datetime.now()
print("Program started at", start_time)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (x, y, w, h), face_encoding in zip(faces, face_encodings):
        # Match faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        name = "Unknown"

        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Detect yawning (mouth detection)
        roi_gray = gray[y:y + h, x:x + w]
        mouth_rects = mouth_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=11, minSize=(30, 30))

        if len(mouth_rects) > 0:
            yawning_count += 1
            cv2.putText(frame, "Yawning Detected", (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            normal_count += 1

        # Drowsiness detection (based on duration or other criteria)
        if len(face_locations) > 0:
            drowsy_count += 1
            cv2.putText(frame, "Drowsiness Detected", (x, y + h + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Save detection data
        detection_data.append({
            "name": name,
            "time": datetime.now(),
            "yawning": len(mouth_rects) > 0,
            "drowsy": len(face_locations) > 0
        })

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

end_time = datetime.now()
print("Program ended at", end_time)
cap.release()
cv2.destroyAllWindows()

# Generate Report
report_filename = "drowsiness_report.pdf"
from fpdf import FPDF

pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.set_font("Arial", size=12)

pdf.cell(200, 10, txt="Drowsiness Detection Report", ln=True, align="C")
pdf.cell(200, 10, txt=f"Start Time: {start_time}", ln=True)
pdf.cell(200, 10, txt=f"End Time: {end_time}", ln=True)

# Total Duration
duration = end_time - start_time
pdf.cell(200, 10, txt=f"Total Duration: {duration}", ln=True)

# Summary
pdf.cell(200, 10, txt=f"Total Yawns: {yawning_count}", ln=True)
pdf.cell(200, 10, txt=f"Total Drowsiness Events: {drowsy_count}", ln=True)

# Graphs
plt.figure(figsize=(10, 5))
plt.bar(["Yawning", "Drowsy", "Normal"], [yawning_count, drowsy_count, normal_count], color=['red', 'yellow', 'green'])
plt.title("Event Summary")
plt.savefig("event_summary.png")
plt.close()

pdf.add_page()
pdf.cell(200, 10, txt="Event Summary Graph", ln=True)
pdf.image("event_summary.png", x=10, y=30, w=180)

# Save PDF
pdf.output(report_filename)
print(f"Report saved as {report_filename}")
