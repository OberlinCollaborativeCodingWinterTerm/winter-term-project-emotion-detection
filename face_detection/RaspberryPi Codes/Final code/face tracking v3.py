"""
This code tracks a face using OpenCV and shows the output on the LCD display using a Raspberry Pi 3B.

Authors:
Jeongho Lee
Otavio Paz Nascimento

References:
Engr Fahad (LCD) - https://www.electroniclinic.com/raspberry-pi-16x2-lcd-i2c-interfacing-and-python-programming/
OpenCV + Raspberry Pi - https://www.tutorialspoint.com/how-to-install-opencv-in-python
Servos control - https://www.digikey.com/en/maker/blogs/2021/how-to-control-servo-motors-with-a-raspberry-pi
Movement Tracking - https://www.youtube.com/watch?v=T_892SKVNf4&ab_channel=CoreElectronics
"""
import cv2 as cv
import time
from pantilthat import *
from gpiozero import AngularServo
from gpiozero.pins.pigpio import PiGPIOFactory
from rpi_lcd import LCD

#Initialize LCD
lcd = LCD()

# Frame Size: Smaller is faster, but less accurate
FRAME_W = 320 # width
FRAME_H = 200 # height

# Pan/Tilt servos in degrees
cam_pan = 0
cam_tilt = 0

# Initilize the servo factory settings 
factory = PiGPIOFactory()

# Both servos are 9g 180 degrees
# min_pulse_width and max_puls_width values were tested until we got the full range of motion
pan = AngularServo(12, min_pulse_width=0.8/1000, max_pulse_width=2.5/1000, pin_factory=factory)
tilt = AngularServo(13, min_pulse_width=0.8/1000, max_pulse_width=2.5/1000, pin_factory=factory)

# Cascade Classifier for face tracking
# This is using the Haar Cascade face recognition method with LBP (Local Binary Patterns)
# Based on our tests, it's better than Haar Cascade without LBP
faceCascade = cv.CascadeClassifier('lbpcascade_frontalface.xml')

# Sets up new sizes for the video capture (must be similar to the ones previously stated)
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH,  320);
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 200);
time.sleep(1)

# Turn the camera to the start position
# pan.angle and tilt.angle variables expect to see are any numbers between -90 to 90 degrees).

pan.angle = 0
tilt.angle = 0

# Infinite loop, the system will run forever or until we manually tell it to stop 
# "esc" button will close the app
while True:

    # Captures frame-by-frame
    _, img = cap.read()

    # Flips the camera because it's monted upside down
    img = cv.flip(img, -1)

    # Convert to greyscale for easier, faster, and accurate face detection
    gray = cv.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv.equalizeHist(gray)

    # Do face detection to search for faces from these captures frames
    faces = faceCascade.detectMultiScale(img, 1.1, 3, 0, (10, 10))
    
    # Shows on the lcd
    lcd.text("Faces detected", 1)
    lcd.text(f"{len(faces)}", 2)
    
    # For every face detected it will draw a rectangle on top of it
    # Then it's going to calculate the center of the face 
    for (x, y, w, h) in faces:
        
        # Draw a green rectangle around the face 
        cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Get the center of the face
        x = x + (w/2)
        y = y + (h/2)

        # Correct relative to center of image
        turn_x  = float(x - (FRAME_W/2))
        turn_y  = float(y - (FRAME_H/2))

        # Convert to percentage offset
        turn_x  /= float(FRAME_W/2)
        turn_y  /= float(FRAME_H/2)

        # Scale offset to degrees (that 2.5 value is a proportional variable, as in PID)
        # We tested and 4.0 seems to have good results
        # Low current also makes the servos slower
        turn_x   *= 4.0 # VFOV
        turn_y   *= 4.0 # HFOV
        cam_pan  += -turn_x
        cam_tilt += -turn_y

        # Clamp Pan/Tilt to -90 and 90 degrees
        cam_pan = max(-90,min(90,cam_pan))
        cam_tilt = max(-90,min(90,cam_tilt))

        # Shows the position of each servo
        print(f'pan: {cam_pan} - tilt: {cam_tilt}')
        
        # Update the servos positions
        pan.angle = int(cam_pan)
        tilt.angle = int(cam_tilt)

        break

   
    # display the video captured, with rectangles overlayed
    cv.imshow('Video', img)

    # type esc at any point this will end the loop
    if cv.waitKey(1) & 0xFF == ord('esc'):
        break
    
# When everything is done, release the ccapture, clear the lcd and destroy all windows
cap.release()
lcd.clear()
cv.destroyAllWindows()
