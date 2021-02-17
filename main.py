import cv2 as cv
import numpy as np

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv.VideoCapture(0)

while cap.isOpened():
    _ , frame = cap.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w,h) in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

        cv.imshow('face detection', frame)

        if cv.waitKey(110) == 'q':
            break

cap.release()
cv.destroyAllWindows()
