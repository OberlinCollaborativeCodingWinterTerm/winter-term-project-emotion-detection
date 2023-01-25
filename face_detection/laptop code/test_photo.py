import cv2 as cv

# saves the image (as a matrix)
image = cv.imread('face detection/otavio.jpg')
cv.imshow("Otavio", image) # shows the image

# the classifier uses the edges to detect a face, so the color it doesn't matter

# apply grayscale filter
grayscale =cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow("Gray Otavio", grayscale)

# using the haar face cascade to detect faces
haar_cascade = cv.CascadeClassifier('face detection/haarcascade_frontalface_default.xml')

# faces_rect will have the lists of coordinates of faces detected
faces_rect = haar_cascade.detectMultiScale(grayscale, scaleFactor=1.1, minNeighbors=6)

# prints out how many faces were detected,
print(f'Number of faces detected: {len(faces_rect)}')

# for every list of coords, create a rectangle using the coordinates
for (x, y, w, h) in faces_rect:
    cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# show the original pictures with the rectangles
cv.imshow('Detected Faces', image)

# it waits forever so it doesn't close when the code is done
cv.waitKey(0)