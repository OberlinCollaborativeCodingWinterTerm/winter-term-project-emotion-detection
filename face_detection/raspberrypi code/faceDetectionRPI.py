import cv2 as cv

# saves the classifier
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# Get the video from Pi Camera.
# Legacy Camera function HAS TO BE ENABLED
cap = cv.VideoCapture(0)

# Resizes the camera so it won't lag that much
cap.set(cv.CAP_PROP_FRAME_WIDTH,  320);
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 200);

while True:
    # boolean and pixels of each frame
    ret, img = cap.read()

    # if the booleans return false, it means it was not possible to retrieve a image from the pi camera
    if ret == False:

      print("Error getting image.")
      continue

    # The camera is upside down, so we flipped 180 degress
    img = cv.flip(img, -1)

    # Change to grayscale so the classifiers are more accurate
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.equalizeHist(gray)

    # detects the face
    faces = face_cascade.detectMultiScale(img, 1.1, 5)

    # draws a rectangle on the detected face
    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # shows the video
    cv.imshow('Pi Camera', img)

    # if the user types 'esc' it will close the program
    esc = cv.waitKey(30) & 0xff

    if esc == 27:
        break

# close all the processess and windows
cap.release()
cv.destroyAllWindows()

