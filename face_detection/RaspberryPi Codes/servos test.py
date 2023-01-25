from gpiozero import AngularServo
from time import sleep

from gpiozero.pins.pigpio import PiGPIOFactory

factory = PiGPIOFactory()

servo = AngularServo(12, min_pulse_width=0.8/1000, max_pulse_width=2.5/1000, pin_factory=factory)

while (True):
    
    servo.angle = 90
    sleep(5)
    servo.angle = 0
    sleep(5)
    servo.angle = -90
    sleep(5)

