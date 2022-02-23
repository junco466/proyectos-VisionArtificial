from typing import Iterable
from matplotlib.pyplot import cla
import mediapipe as mp
import numpy as np
import csv
import os
import cv2
import time
import pathlib

path = str(pathlib.Path().absolute())

from numpy import empty, random

start_time = time.time()

#La base de datos se puede descargar del siguiente link
# https://www.kaggle.com/grassknoted/asl-alphabet 
DATADIR = f"{path}\\entrenar"
DATADIR_TEST = f"{path}\\prueba"

CATEGORIES = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']



training_data = []
training_data_bin = []
features = []

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def create_taining_data(datadir):

    chart=[]
    for category in CATEGORIES:
        path = os.path.join(datadir,category)
        class_num = CATEGORIES.index(category)

        '''for i in range (10):
            image = random.choice([
                x for x in os.listdir(path)
                if os.path.isfile(os.path.join(path, x))
            ])'''
        for image in os.listdir(path):

            with mp_hands.Hands(min_detection_confidence=0.8) as hands:
                img = cv2.imread(os.path.join(path,image),cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img.flags.writeable = False
                result = hands.process(img)
                img.flags.writeable = True
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                landmark_coordinates_list=[]
                if result.multi_hand_landmarks != None:
                    for handLandmarks in result.multi_hand_landmarks:
                        if handLandmarks != None:
                            for point in mp_hands.HandLandmark:
                                landmark_coordinates = [handLandmarks.landmark[point].x,handLandmarks.landmark[point].y,handLandmarks.landmark[point].z]
                                landmark_coordinates_list = landmark_coordinates_list + landmark_coordinates
                            print(image)

                            if (datadir is DATADIR_TEST):
                                chart = landmark_coordinates_list
                            else:
                                chart = landmark_coordinates_list
                                chart.append(class_num)

            features.append(chart)

'''
create_taining_data(DATADIR)


random.shuffle(features)
test = open("features_train_02.csv", 'w', newline='')
wr = csv.writer(test, dialect='excel')

for item in features:
    wr.writerow(item)

print(len(features))

features.clear()
'''

create_taining_data(DATADIR_TEST)

test = open("features_test_02.csv", 'w', newline='')
wr = csv.writer(test, dialect='excel')
for item in features:
    wr.writerow(item)


print(len(features))

print("----- %s minutos ------" % ((time.time()-start_time)/60))