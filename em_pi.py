from keras.utils.image_utils import img_to_array
import cv2, sys, time, os
from pantilthat import *
from gpiozero import AngularServo
from gpiozero.pins.pigpio import PiGPIOFactory
from keras.models import load_model
import numpy as np

# parameters for loading data and images
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised",
 "neutral"]


FRAME_W = 320
FRAME_H = 200

# Default Pan/Tilt for the camera in degrees. I have set it up to roughly point at my face location when it starts the code.
# Camera range is from 0 to 180. Alter the values below to determine the starting point for your pan and tilt.
cam_pan = 0
cam_tilt = 0

factory = PiGPIOFactory()

pan = AngularServo(12, min_pulse_width=0.8/1000, max_pulse_width=2.5/1000, pin_factory=factory)
tilt = AngularServo(13, min_pulse_width=0.8/1000, max_pulse_width=2.5/1000, pin_factory=factory)

# getting access to the camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  320);
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 200);
time.sleep(2)


while True:
    frame = cap.read()[1]
    #reading the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    
    if len(faces) > 0:
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

    for (fX, fY, fW, fH) in faces:
        # Draw a green rectangle around the face (There is a lot of control to be had here, for example If you want a bigger border change 4 to 8)
        cv2.rectangle(frame, (fX, fY), (fX+fW, fY+fH), (0, 255, 0), 4)

        # Track face with the square around it
        
        # Get the centre of the face
        fX = fX + (fW/2)
        fY = fY + (fH/2)

        # Correct relative to centre of image
        turn_x  = float(fX - (FRAME_W/2))
        turn_y  = float(fY - (FRAME_H/2))

        # Convert to percentage offset
        turn_x  /= float(FRAME_W/2)
        turn_y  /= float(FRAME_H/2)

        # Scale offset to degrees (PID)
        turn_x   *= 4.5 # VFOV
        turn_y   *= 4.5 # HFOV
        cam_pan  += -turn_x
        cam_tilt += -turn_y

        # Clamp Pan/Tilt to 0 to 180 degrees
        cam_pan = max(-90,min(90,cam_pan))
        cam_tilt = max(-90,min(90,cam_tilt))

        print(f'pan: {cam_pan} - tilt: {cam_tilt}')
        
        # Update the servos
        pan.angle = int(cam_pan)
        tilt.angle = int(cam_tilt)

        break

    #If no faces found, just run the loop again
    else: continue

    for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                cv2.putText(frame, label, (fX, fY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH),
                              (0, 0, 255), 2)

    cv2.imshow('your_face', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
