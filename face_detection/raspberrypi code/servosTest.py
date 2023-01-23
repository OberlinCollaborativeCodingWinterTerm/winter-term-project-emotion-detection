import cv2 as cv
from gpiozero import AngularServo
from time import sleep

from gpiozero.pins.pigpio import PiGPIOFactory

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH,  320);
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 200);

factory = PiGPIOFactory()

servo = AngularServo(12, min_pulse_width=0.8/1000, max_pulse_width=2.5/1000, pin_factory=factory)
servo.angle = 0

while (True):
    ret, img = cap.read()
    
    img = cv.flip(img, -1)
        
    try:    
        servo.angle = servo.angle + 5 # it throws an error if the value is greater than 90
        sleep(1)
        
    except:
        servo.angle = 80
        servo.angle = 70
        servo.angle = 60
        servo.angle = 50
        servo.angle = 40
        servo.angle = 30
        servo.angle = 20
        servo.angle = 10
        sleep(5)
        
    cv.imshow('Webcam', img)
    
    esc = cv.waitKey(30) & 0xff
    if esc == 27:
        break

cap.release()
cv.destroyAllWindows()

