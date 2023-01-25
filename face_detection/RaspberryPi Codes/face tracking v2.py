import cv2 as cv
from gpiozero import AngularServo
from time import sleep

from gpiozero.pins.pigpio import PiGPIOFactory

# saves the classifier
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# Get the video from Pi Camera.
# Legacy Camera function HAS TO BE ENABLED
cap = cv.VideoCapture(0)

FRAME_WIDTH = 320
FRAME_HEIGHT = 200

LINES_WIDTH = FRAME_WIDTH // 3 # x value
LINE_HEIGHT = FRAME_HEIGHT // 2 # y value

# Resizes the camera so it won't lag that much
cap.set(cv.CAP_PROP_FRAME_WIDTH,  320);
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 200);

# Servos configs

factory = PiGPIOFactory()

pan = AngularServo(12, min_pulse_width=0.8/1000, max_pulse_width=2.5/1000, pin_factory=factory)
tilt = AngularServo(13, min_pulse_width=0.7/1000, max_pulse_width=2.0/1000, pin_factory=factory)
pan_cx = 0
pan.angle = pan_cx
tilt.angle = 45
movement = 5




while True:
    # boolean and pixels of each frame
    ret, img = cap.read()

    # # if the booleans return false, it means it was not possible to retrieve a image from the pi camera
    # if ret == False:

    #   print("Error getting image.")
    #   continue

    # draws imaginary lines - WIDTH
    cv.line(img, (LINES_WIDTH, 0), (LINES_WIDTH, FRAME_HEIGHT), (0, 255, 0), 2) # left vertical line
    cv.line(img, (2*LINES_WIDTH, 0), (2*LINES_WIDTH, FRAME_HEIGHT), (0, 255, 0), 2) # right vertical line

    # draw imaginary line - HEIGHT
    cv.line(img, (0, LINE_HEIGHT), (FRAME_WIDTH, LINE_HEIGHT), (0, 255, 0), 2) # horizontal line

    # The camera is upside down, so we flipped 180 degress
    img = cv.flip(img, -1)

    # Change to grayscale so the classifiers are more accurate
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.equalizeHist(gray)

    # detects the face
    faces = face_cascade.detectMultiScale(img, 1.1, 5)

    
    # for (x, y, w, h) in faces:
    #     cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
 
    # prioritize the first detected face
    first_face = faces[:1]

    for (x, y, w, h) in first_face:
        # draws a rectangle on the detected face
        cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # calculates the mid point
        mid_point_x = x + (w // 2)
        mid_point_y = y

        # print(f'{x} {y} {w} {h} - midpoint x: {mid_point_x}')
        # mid point crosses vertical left line
        
        # Servo movements

        if mid_point_x < LINES_WIDTH:
            print('move left')
            pan_cx += movement

            if pan_cx < -90:
                pan_cx = -90
                pan.angle = pan_cx
                
            elif pan_cx > 90:
                pan_cx = 90
                pan.angle = pan_cx
                
            else:
                pan.angle = pan_cx
           
        if mid_point_x > LINES_WIDTH and mid_point_x < 2*LINES_WIDTH:
            print('centralized')
            pan.angle = pan_cx
    
        # mid point crosses vertical right line
        if mid_point_x > 2*LINES_WIDTH:
            print('move right')
            pan_cx -= movement
            if pan_cx < -90:
                pan_cx = -90
                pan.angle = -90
                
            elif pan_cx > 90:
                pan_cx = 90
                pan.angle = pan_cx
                
            else:
                pan.angle = pan_cx
        
        # mid point crosses the horizontal line
        if mid_point_y > LINE_HEIGHT:
            print('move down')

        # mid point crosses the imaginary top horizontal line 
        if mid_point_y < 15:
            print('move up')

    # shows the video
    cv.imshow('Pi Camera', img)
    
    print(pan_cx)
    # if the user types 'esc' it will close the program
    esc = cv.waitKey(30) & 0xff

    if esc == 27:
        break

# close all the processess and windows
cap.release()
cv.destroyAllWindows()
