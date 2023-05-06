#import necessary libraries
import cv2 
import os
import gradio as gr


#add folder path containing jpg or png images to read and display them

image_folder = 'C:\\Users\\YOUR_PATH_OF_IMAGES'

for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png") :
        image_path = os.path.join(image_folder, filename)
        
        # Read the image
        image = cv2.imread(image_path)
        
        # Display the image
        cv2.imshow('Image', image)
        cv2.waitKey(0)  # Wait for key press to display the next image
        cv2.destroyAllWindows()
        
        

# Load the pre-trained face recognition model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def recognize_faces(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Check if the image was read successfully
    if image is None or image.size == 0:
        print(f"Failed to read or empty image: {image_path}")
        return
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)
    
    # Display the image with faces detected
    cv2.imshow('Face Recognition', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
    
    
    
# Path to the folder containing the images
def process_input_folder(upload):
    folder_path = "./temp_folder"
    os.makedirs(folder_path, exist_ok=True)
    
    # Iterate over the uploaded files and save them in the temporary folder
    for file in upload:
        file_path = os.path.join(folder_path, file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())
        recognize_faces(file_path)   

    
    # Remove the temporary folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        os.remove(file_path)
    os.rmdir(folder_path)

    
 #interface designed through gradio   
    
iface = gr.Interface(fn=process_input_folder, inputs="files", outputs=None, title="Face Recognition",
                     description="Upload a folder containing images, and the system will detect faces and draw rectangles around them in each image.")

iface.launch() 
    
    
 
