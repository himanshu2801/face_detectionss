import dlib
import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

hog_face_detector = dlib.get_frontal_face_detector()

face_landmark = dlib.shape_predictor('shape_predictor_68_face_landmark.dat')

while True:
    _, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = hog_face_detector(gray)

    for face in faces:
        face_landmarks = face_landmark(gray, face)

        for i in range(0, 68):
            x = face_landmarks.part(i).x
            y = face_landmarks.part(i).y
            cv.circle(frame, (x, y), 1, (0, 255, 0), 1)

        cv.imshow('frame', frame)

        if cv.waitKey(0) == 27:
            break

cap.release()
cv.destroyedAllwindows()