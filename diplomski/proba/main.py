import mediapipe as mp
import cv2
import numpy as np
import uuid
import os

import tensorflow as tf
from tensorflow.keras.models import load_model

cap = cv2.VideoCapture(0)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Load the gesture recognizer model
model = load_model('mp_hand_gesture')

print(model.summary())

# Load class names
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands: 

    estimate_pose = False

    while cap.isOpened():
        ret, frame = cap.read()
        x, y, c = frame.shape
        
        # BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Flip on horizontal
        image = cv2.flip(image, 1)
        
        # Set flag
        image.flags.writeable = False
        
        # Detections
        results = hands.process(image)
        
        # Set flag to true
        image.flags.writeable = True
        
        # RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        className = ""
        
        # Rendering results
        label_draw_info = []
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                         )
                if estimate_pose:
                    landmarks = []
                    label_draw_info.append([hand.landmark[0].x * x, hand.landmark[0].y * y])
                    for id, lm in enumerate(hand.landmark):
                        # print(id, lm)
                        lmx = int(lm.x * x)
                        lmy = int(lm.y * y)
                        landmarks.append([lmx, lmy])
                    prediction = model.predict([landmarks])
                    # print(prediction)
                    classID = np.argmax(prediction)
                    className = classNames[classID]
                    # print(className)
                    label_draw_info[len(label_draw_info) - 1].append(className)
        
        if estimate_pose:
            for coordinates in label_draw_info:
                cv2.putText(image, coordinates[2], (int(coordinates[0]), int(coordinates[1])), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0,0,255), 2, cv2.LINE_AA)
        cv2.imshow('Hand Tracking', image)

        key = cv2.waitKey(1)
        if key == 32: # SPACE
            estimate_pose = not estimate_pose
        # if results.multi_hand_landmarks is not None and key == 115:
        #     for i in range(21):
        #         print("X{}:".format(i), results.multi_hand_landmarks[0].landmark[i].x, "\nY{}:".format(i), results.multi_hand_landmarks[0].landmark[i].y, "\n")

        
        if key == 27: # ESCAPE
            break

cap.release()
cv2.destroyAllWindows()