import face_recognition
import numpy as np
import cv2 
import os
import gradio as gr


image_folder = 'C:\\Users\\face_recognition\\faces\\'

def extract_face_embeddings(image_folder):
    face_embeddings = {}
    for filename in os.listdir(image_folder1):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            image_path = os.path.join(image_folder1, filename)
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image)
            if len(face_locations) > 0:
                face_encoding = face_recognition.face_encodings(image, face_locations)[0]
                face_embeddings[filename] = face_encoding
    return face_embeddings

# Provide the path to the folder containing the images
image_folder2 = 'C:\\Users\\face_recognition\\faces\\faces2\\'

# Extract face embeddings from the images in the folder
celebrity_embeddings = extract_face_embeddings(image_folder)

# Save the face embeddings as .npy files
for celebrity, embedding in celebrity_embeddings.items():
    np.save(os.path.join(image_folder, f"{celebrity}.npy"), embedding)
    
    
# Replace this with the appropriate model and loading code
face_model = None

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the dataset of known celebrities
known_celebrities = {
    'Scarlet Johansson': r'C:\Users\hamna\Downloads\COMPUTER VISION\face_recognition\faces\faces2\scarlet.jpg.npy',
    'Tom Cruise': r'C:\Users\hamna\Downloads\COMPUTER VISION\face_recognition\faces\faces2\tom.jpg.npy',
    'Amber Heard': r'C:\Users\hamna\Downloads\COMPUTER VISION\face_recognition\faces\faces2\amber.jpg.npy',
    'Miley Cyrus': r'C:\Users\hamna\Downloads\COMPUTER VISION\face_recognition\faces\faces2\miley.jpg.npy',
    'Jennifer Lopez': r'C:\Users\hamna\Downloads\COMPUTER VISION\face_recognition\faces\faces2\jlo.jpg.npy'
    # Add more celebrities and their respective embeddings files
}

celeb_embeddings = {}
for celeb, emb_file in known_celebrities.items():
    celeb_embeddings[celeb] = np.load(emb_file)
    
    
    
    
    
    
def recognize_faces(image):
    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract the face region
        face_roi = image[y:y+h, x:x+w]

        # Preprocess the face region (e.g., resizing, normalization) if needed

        # Compute the face embedding using the face recognition model
        face_embedding = face_model.compute_face_embedding(face_roi)

        # Find the closest match among known celebrities
        min_distance = float('inf')
        recognized_celebrity = 'Unknown'

        for celeb, celeb_embedding in celeb_embeddings.items():
            # Compute the distance between the face embedding and the celebrity's embedding
            distance = np.linalg.norm(face_embedding - celeb_embedding)

            # Update the recognized celebrity if a closer match is found
            if distance < min_distance:
                min_distance = distance
                recognized_celebrity = celeb

        # Draw the bounding box and label the recognized celebrity
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(image, recognized_celebrity, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return image

 def process_folder(image_folder1):
    # Iterate over each image in the folder
    for filename in os.listdir(image_folder1):
        # Check if the file is an image (you can add more image extensions if needed)
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # Read the image
            image_path = os.path.join(image_folder1, filename)
            image = cv2.imread(image_path)

            # Perform face recognition
            processed_image = recognize_faces(image)

            # Display the processed image
            cv2.imshow("Face Recognition", processed_image)
            cv2.waitKey(0)

    cv2.destroyAllWindows()

    
def process_image(input_image):
    # Read the input image
    image = cv2.imread(input_image)

    # Perform face recognition
    processed_image = recognize_faces(image)

    return processed_image


# Create a Gradio interface
iface = gr.Interface(fn=process_image, inputs="image", outputs="image")

# Specify the path to the folder containing the images
# Process the images in the folder
image_folder2 = 'C:\\Users\\hamna\\Downloads\\COMPUTER VISION\\face_recognition\\faces\\faces2\\'
processed_images = process_folder(image_folder)

# Launch the Gradio interface to visualize the processed images
iface.launch(processed_images)

    
