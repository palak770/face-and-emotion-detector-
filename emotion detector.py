import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the face detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load your custom-trained emotion model
emotion_model = load_model('my_custom_emotion_model.h5')

# --- THIS IS THE LINE TO FIX ---
# This list must contain all 7 emotion folder names, in alphabetical order.
# Please verify these names against your 'train' folder.
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each face found
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Extract the face region (Region of Interest - ROI)
        roi_gray = gray[y:y+h, x:x+w]
        
        # Prepare the ROI for your custom model (96x96 pixel color image)
        roi_color = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)
        roi_color = cv2.resize(roi_color, (96, 96), interpolation=cv2.INTER_AREA)
        img_pixels = roi_color.astype('float') / 255.0
        img_pixels = np.expand_dims(img_pixels, axis=0)

        # Make a prediction using your custom model
        prediction = emotion_model.predict(img_pixels)[0]
        
        # Get the emotion with the highest confidence
        label = emotion_labels[prediction.argmax()]
        label_position = (x, y - 10)

        # Display the predicted emotion label
        cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the final result
    cv2.imshow('Custom Emotion Detector', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
