import cv2 as cv
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mphands = mp.solutions.hands

cap = cv.VideoCapture(0)
hands = mphands.Hands()
while True:
    data,image = cap.read()
    image = cv.cvtColor(cv.flip(image,1),cv.COLOR_BGR2RGB)
    results = hands.process(image)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,mphands.HAND_CONNECTIONS)
    cv.imshow('Handtracker',image)
    cv.waitKey(1)