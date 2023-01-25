from keras.utils.image_utils import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np

# saving paths in a variable
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised",
 "neutral"]

camera = cv2.VideoCapture(0) # get video capture objects and store it in the variable camera
while True:
    frame = camera.read()[1] # read the frame
    frame = imutils.resize(frame,width=300) # resize the image (frame) to width 300 while maintaining proportion. 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert the color of the image (frame) to gray for easier detection of the edges of the faces
    faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE) # Detect faces,
    
    frameClone = frame.copy()
    if len(faces) > 0: # if at least one face is detected 
        faces = sorted(faces, reverse=True,
        key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
                    # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
            # the ROI for classification via the CNN
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        
        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]
    else: continue # if no faces detected, start the loop over 

 
    for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)): 
                # construct the label text
                cv2.putText(frameClone, label, (fX, fY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                              (0, 0, 255), 2)

# Resize the video to make it bigger on the laptop screen
    width = int(frameClone.shape[1] * 500 / 100) # 50% increase in width
    height = int(frameClone.shape[0] * 500 / 100) # 50% increase in height
    dimensions = (width, height) # dimensions with both the width and height
    Show = cv2.resize(frameClone, dimensions, interpolation=cv2.INTER_AREA) # resize the frameClone video streaming with dimensions above, store it in Show
    cv2.imshow('Video', Show) # show the video streaming in a window named "video"
    if cv2.waitKey(1) & 0xFF == ord('q'): # press q to get out of the loop
        break

camera.release()
cv2.destroyAllWindows()
