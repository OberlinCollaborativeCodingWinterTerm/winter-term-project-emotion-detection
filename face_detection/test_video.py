import cv2 as cv

# it saves the face cascade
face_cascade = cv.CascadeClassifier('face detection\haarcascade_frontalface_default.xml')

# it captures the webcam video
cap = cv.VideoCapture(0)

while True:
    _, img = cap.read() # splits into a boolean variable (which is not used) and a frame

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # applies grayscale filter

    # we mighe have to tune this value
    faces = face_cascade.detectMultiScale(gray, 1.1, 4) # detects faces

    # for each list of coords, create a rectangle
    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # shows the original image with the rectangles placed on where the algorithm found faces
    cv.imshow('Webcam', img)

    # it closes the apps when the 'esc' button is pressed
    esc = cv.waitKey(30) & 0xff
    if esc == 27:
        break

cap.release()