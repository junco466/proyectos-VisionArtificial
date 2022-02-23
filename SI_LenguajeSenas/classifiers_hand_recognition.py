import pickle
import itertools
import numpy as np
import random
import csv
from sklearn import svm
from sklearn import tree
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pickle
import time

start_time = time.time()

# Funcion de prediccion creada / Toma como parametros el vector de caracteristicas y el calsificador

CATEGORIES = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

def test(vector, classifier):
    result = classifier.predict(vector)  # Se almacena en una variable el resultado de la prediccion
    print('Letra', CATEGORIES[int(result)])
    

dataset = np.genfromtxt('features_train_02.csv', delimiter=',')
test_dataset = np.genfromtxt('features_test_02.csv', delimiter=',')
# Dataset de cancer 
# dataset = np.loadtxt('breastCancer.txt', delimiter=',')
data, labels = dataset[:, 0:63], dataset[:, 63]


# Divison del dataset entre entrenamiento y prueba
data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.20, random_state=42)
labels_train2 = labels_train.T

# Vector de caracteristicas para prueba de prediccion
caracteristicas = test_dataset[8]
caracteristicas = caracteristicas.reshape(1, -1)  # Se reorganizan los datos

'''
# Maquina de soporte vectorial
clfSVM = svm.SVC(kernel='rbf', gamma=0.7, C=1)  # Clasificador
clfSVM = clfSVM.fit(data_train, labels_train)  # Entrenamiento
testSVM = clfSVM.score(data_test, labels_test)  # Prueba
print('Puntuacion Maquina de soporte vectorial:', testSVM)  # Puntaje
test(caracteristicas, clfSVM)  # Se invoca funcion para predicción
print('************************************************************')
'''
#Save model
filename = 'model_01.sav'
#pickle.dump(clfSVM, open(filename, 'wb'))

#load model
print('**********************Loaded model***************************')
loaded_model = pickle.load(open(filename, 'rb'))
test_loaded_model = loaded_model.score(data_test, labels_test)  # Prueba
print('Puntuacion Maquina de soporte vectorial:', test_loaded_model)  # Puntaje
test(caracteristicas, loaded_model)
print('*************************************************************')


print("----- %s segundos ------" % (time.time()-start_time))