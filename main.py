import streamlit as st
import cv2
import dlib
import numpy as np
from scipy.spatial import distance
from matplotlib import pyplot as plt
from datetime import datetime
import time
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Load face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to calculate Mouth Aspect Ratio (MAR)
def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[10])
    B = distance.euclidean(mouth[4], mouth[8])
    C = distance.euclidean(mouth[0], mouth[6])
    mar = (A + B) / (2.0 * C)
    return mar

# Initialize variables
drowsy_count = 0
yawn_count = 0
sleep_duration = 0
is_sleeping = False
start_time = time.time()
screen_on_time = 0
recording_active = True

# Start video capture (using a mock for Streamlit demo purposes)
cap = cv2.VideoCapture(0)

# Streamlit Dashboard
st.title("Drowsiness Detection Dashboard")

# Create placeholders for real-time data
start_time_placeholder = st.empty()
drowsy_count_placeholder = st.empty()
yawn_count_placeholder = st.empty()
sleep_duration_placeholder = st.empty()
screen_on_time_placeholder = st.empty()
graph_placeholder = st.empty()
passport_image_placeholder = st.empty()

# Start the live stream (simulated for this demo)
while recording_active:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = detector(gray)
    
    for face in faces:
        landmarks = predictor(gray, face)
        
        # Get the points for the eyes and mouth
        left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
        right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
        mouth = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)]
        
        # Calculate EAR for both eyes
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        
        # Detect drowsiness (if EAR < 0.2 for 2 consecutive frames)
        if ear < 0.2:
            drowsy_count += 1

        # Detect yawning (if MAR > 0.5)
        mar = mouth_aspect_ratio(mouth)
        if mar > 0.5:
            yawn_count += 1
        
        # Sleeping Detection (based on eye closure duration or posture)
        if ear < 0.2:
            is_sleeping = True
            sleep_duration = (time.time() - start_time) / 60  # in minutes
        else:
            is_sleeping = False

        # Update Streamlit placeholders with real-time data
        start_time_placeholder.text(f"Start Time: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}")
        drowsy_count_placeholder.text(f"Drowsy Count: {drowsy_count}")
        yawn_count_placeholder.text(f"Yawning Count: {yawn_count}")
        sleep_duration_placeholder.text(f"Total Sleep Duration: {sleep_duration:.2f} minutes")
        screen_on_time_placeholder.text(f"Screen On Time: {screen_on_time / 60:.2f} minutes")
    
    # Capture the passport-size image
    cv2.imwrite("passport_image.jpg", frame)
    passport_image_placeholder.image("passport_image.jpg", caption="Passport Image", use_container_width=True)

    # Plot the graph of drowsiness, yawning, and sleep duration
    fig, ax = plt.subplots()
    ax.bar(["Drowsy", "Yawning", "Sleeping"], [drowsy_count, yawn_count, sleep_duration])
    plt.xlabel('Detection Type')
    plt.ylabel('Count')
    plt.title('Real-time Detection Summary')
    graph_path = "detection_graph.png"
    plt.savefig(graph_path)
    plt.close(fig)
    graph_placeholder.image(graph_path, caption="Detection Summary Graph", use_container_width=True)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Function to generate the PDF report
def generate_report(drowsy_count, yawn_count, sleep_duration, screen_on_time, start_time, frame):
    c = canvas.Canvas("Drowsiness_Report.pdf", pagesize=letter)
    width, height = letter
    
    # Passport size image (capturing current frame as passport-sized image)
    cv2.imwrite("passport_image.jpg", frame)
    c.drawImage("passport_image.jpg", 50, height - 150, width=100, height=100)

    # Title
    c.setFont("Helvetica", 16)
    c.drawString(200, height - 100, f"Drowsiness Detection Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Add space for better formatting
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 180, f"Start Time: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawString(50, height - 200, f"Total Duration: {screen_on_time / 60:.2f} minutes")
    
    # Drowsy, Yawn, and Sleep Statistics
    c.drawString(50, height - 240, f"Drowsy Count: {drowsy_count}")
    c.drawString(50, height - 260, f"Yawning Count: {yawn_count}")
    c.drawString(50, height - 280, f"Total Sleep Duration: {sleep_duration:.2f} minutes")
    c.drawString(50, height - 300, f"Screen On Time: {screen_on_time / 60:.2f} minutes")

    # Add a bit more space before the graph
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 320, "Detection Summary (Graph):")

    # Graph of counts
    fig, ax = plt.subplots()
    ax.bar(["Drowsy", "Yawning", "Sleeping"], [drowsy_count, yawn_count, sleep_duration])
    plt.xlabel('Detection Type')
    plt.ylabel('Count')
    plt.title('Real-time Detection Summary')
    graph_path = "detection_graph.png"
    plt.savefig(graph_path)
    plt.close(fig)
    c.drawImage(graph_path, 50, height - 450, width=500, height=200)
    
    # Footer message
    c.setFont("Helvetica", 10)
    c.drawString(50, 50, "Report generated automatically. Data is based on real-time monitoring.")

    # Save the PDF
    c.save()

# Streamlit button to stop recording and download the report
if st.button('Stop Recording and Download Report'):
    recording_active = False  # Stop the recording
    
    # Generate and download the report
    generate_report(drowsy_count, yawn_count, sleep_duration, screen_on_time, start_time, frame)
    
    with open("Drowsiness_Report.pdf", "rb") as pdf_file:
        st.download_button("Click to Download Report", pdf_file, file_name="Drowsiness_Report.pdf")

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
