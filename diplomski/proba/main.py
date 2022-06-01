from copyreg import pickle
import mediapipe as mp
import cv2
import numpy as np
import pickle
from prometheus_client import Enum
import time

import tensorflow as tf
from tensorflow.keras.models import load_model

class ModelType(Enum):
    SVM = "SVM"
    NN = "NN"
    RANDOM_FOREST = "RF"

# Load the gesture recognizer model
def load_nn_model():
    model = load_model('my_hand_gesture')
    print(model.summary())
    return model

def load_svm_model():
    with open("svm.pickle", "rb") as file:
        model = pickle.load(file)
    return model

def load_random_forest_model():
    with open("random_forest.pickle", "rb") as file:
        model = pickle.load(file)
    return model

# Load class names
def load_class_names():
    f = open('gesture.names', 'r')
    class_names = f.read().split('\n')
    f.close()
    print(class_names)
    return class_names

def run_real_time_demo(cap, model_type, class_names, count_fps=False):
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    if model_type == ModelType.SVM:
        model = load_svm_model()
    elif model_type == ModelType.NN:
        model = load_nn_model()
    elif model_type == ModelType.RANDOM_FOREST:
        model = load_random_forest_model()

    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands: 

        estimate_pose = False

        previous_frame_time = 0        
        current_frame_time = 0

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
                        if model_type != ModelType.NN:
                            landmarks = []
                            label_draw_info.append([hand.landmark[0].x * x, hand.landmark[0].y * y])
                            for id, lm in enumerate(hand.landmark):
                                lmx = float(lm.x)
                                lmy = float(lm.y)
                                landmarks.append(lmx)
                                landmarks.append(lmy)
                            # print(landmarks)
                            prediction = model.predict([landmarks])
                            classID = prediction[0]
                            className = class_names[classID]
                            label_draw_info[len(label_draw_info) - 1].append(className)
                        else:
                            landmarks = []
                            label_draw_info.append([hand.landmark[0].x * x, hand.landmark[0].y * y])
                            for id, lm in enumerate(hand.landmark):
                                # print(id, lm)
                                lmx = int(lm.x * x)
                                lmy = int(lm.y * y)
                                landmarks.append([lmx, lmy])
                            prediction = model.predict([landmarks])
                            classID = np.argmax(prediction)
                            className = class_names[classID]
                            # print(className)
                            label_draw_info[len(label_draw_info) - 1].append(className)
            
            # FPS counter
            if count_fps:
                current_frame_time = time.time()
                fps = 1/(current_frame_time - previous_frame_time)
                previous_frame_time = current_frame_time
                fps = int(fps)
                fps = str(fps)
                cv2.putText(image, fps, (7, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2, cv2.LINE_AA)

            if estimate_pose:
                for coordinates in label_draw_info:
                    cv2.putText(image, coordinates[2], (int(coordinates[0]), int(coordinates[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0,0,255), 2, cv2.LINE_AA)
            cv2.imshow('Hand Tracking And Gesture Recognition', image)

            key = cv2.waitKey(1)
            if key == 32: # SPACE
                estimate_pose = not estimate_pose
            if key == 27: # ESCAPE
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    run_real_time_demo(cap, ModelType.NN, load_class_names(), count_fps=True)
