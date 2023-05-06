# Face_recognition_gradio
I have designed face recognition system and this app's interface is done through gradio.


The code provided demonstrates a basic implementation of face recognition using OpenCV and a pre-trained face recognition model. Here's an explanation of the different parts:

1. **Loading the models and datasets:**
   - The code starts by initializing the face recognition model (`face_model`) and the face detection model (`face_cascade`).
   - The dataset of known celebrities (`known_celebrities`) is defined as a dictionary where the keys are the names of the celebrities and the values are the file paths to their corresponding face embeddings files.

2. **Recognizing Faces:**
   - The `recognize_faces` function takes an input image as a parameter.
   - It converts the image to grayscale using `cv2.cvtColor` since face detection works best on grayscale images.
   - The `face_cascade.detectMultiScale` function is used to detect faces in the grayscale image. It returns a list of rectangles representing the bounding boxes of the detected faces.
   - For each detected face, the face region is extracted from the original image using the coordinates of the bounding box.
   - The face embedding is computed using the `face_model.compute_face_embedding` function. This step involves preprocessing the face region, such as resizing or normalization, if necessary.
   - The code then compares the computed face embedding with the known celebrity embeddings in `celeb_embeddings` to find the closest match. It computes the Euclidean distance between the embeddings using `np.linalg.norm`.
   - The recognized celebrity with the minimum distance is updated.
   - Finally, the bounding box and the recognized celebrity's name are drawn on the original image using `cv2.rectangle` and `cv2.putText`, respectively.

3. **Main Function:**
   - The main function reads an input image using `cv2.imread`.
   - It calls the `recognize_faces` function to perform face recognition on the image.
   - The processed image is displayed using `cv2.imshow`, and `cv2.waitKey` is used to wait for a key press to close the window.

It's important to note that in the code you provided, the face recognition model (`face_model`) and the loading code for the embeddings files are missing. You would need to replace the placeholder code with the appropriate face recognition model and loading mechanism. Additionally, the file paths for the input image and the embeddings files need to be provided.
