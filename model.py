import face_recognition
import os
import pickle

# Save model function
def save_model(known_encodings, known_names):
    with open(r'models\face_encodings_model.pkl', 'wb') as file:
        pickle.dump({'encodings': known_encodings, 'names': known_names}, file)
    print("Model saved successfully!")

# Path to the directory with known images
known_faces_dir = r"images"

# Load and encode known faces
known_encodings = []
known_names = []

# Iterate through all files in the known faces directory
for file_name in os.listdir(known_faces_dir):
    if file_name.endswith(('.png', '.jpg', '.jpeg')):  # Process only image files
        # Load the image
        image_path = os.path.join(known_faces_dir, file_name)
        image = face_recognition.load_image_file(image_path)

        # Encode the face
        encodings = face_recognition.face_encodings(image)
        if encodings:  # Ensure at least one face is detected
            known_encodings.append(encodings[0])
            # Extract the name from the filename (e.g., "Alice.png" -> "Alice")
            known_names.append(os.path.splitext(file_name)[0])

# Save the model after encoding the faces
save_model(known_encodings, known_names)
