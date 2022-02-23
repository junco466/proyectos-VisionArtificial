import mediapipe as mp
import cv2
import numpy as np
import uuid
import os
import pickle

#     Import Model
filename = 'model_01.sav'
print('**********************Loading model***************************')
loaded_model = pickle.load(open(filename, 'rb'))
print('******************model succesfully loaded********************')

CATEGORIES = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

def test(vector, classifier):
    result = classifier.predict(vector) 
    print('Letra', CATEGORIES[int(result)])
    letra = CATEGORIES[int(result)]
    cv2.putText(image,letra,(10,420), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 6, cv2.LINE_AA)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands: 
    while cap.isOpened():
        ret, frame = cap.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #image = cv2.flip(image, 1)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        landmark_coordinates_list=[]
        if results.multi_hand_landmarks != None:
            for handLandmarks in results.multi_hand_landmarks:
                for point in mp_hands.HandLandmark:
                                landmark_coordinates = [handLandmarks.landmark[point].x,handLandmarks.landmark[point].y,handLandmarks.landmark[point].z]
                                landmark_coordinates_list = landmark_coordinates_list + landmark_coordinates
            if len(landmark_coordinates_list)==63:
                caracteristicas=np.array(landmark_coordinates_list)
                caracteristicas = caracteristicas.reshape(1, -1)
                test(caracteristicas, loaded_model)
                
                

        # Rendering results
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                         )
            
        
        cv2.imshow('Hand Tracking', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()