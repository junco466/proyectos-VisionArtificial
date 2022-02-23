import numpy as np
import cv2


cap = cv2.VideoCapture(0)
flag = False

azulBajo = np.array([100,100,20],np.uint8)
azulAlto = np.array([125,255,255],np.uint8)

rojoBajo1 = np.array([0, 100, 20], np.uint8)
rojoAlto1 = np.array([8, 255, 255], np.uint8)
rojoBajo2 = np.array([175, 100, 20], np.uint8)
rojoAlto2 = np.array([179, 255, 255], np.uint8)

amarilloBajo = np.array([27,100,20],np.uint8)
amarilloAlto = np.array([33,255,255],np.uint8)

amarilloBajo = np.array([27,100,20],np.uint8)
amarilloAlto = np.array([35,255,255],np.uint8)

verdeBajo = np.array([40,100,20])
verdeAlto = np.array([70,255,255])

cyanBajo = np.array([87,100,20])
cyanAlto = np.array([95,255,255])

naranjaBajo = np.array([20,100,20])
naranjaAlto = np.array([28,255,255])

rosaBajo = np.array([148,100,20])
rosaAlto = np.array([155,255,255])

moradoBajo = np.array([0.391*100,0.619*100,0.104*100])
moradoAlto = np.array([0.160*100,0.441*100,0.093*100])

while True:

    # Captura cuadro a cuadro
    _, frame = cap.read()

    hsv = gray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

    segAzul = cv2.inRange(hsv,azulBajo,azulAlto)

    segRojo1 = cv2.inRange(hsv,rojoBajo1,rojoAlto1) 
    segRojo2 = cv2.inRange(hsv,rojoBajo2,rojoAlto2)
    maskRed = cv2.add(segRojo1, segRojo2)
    #maskRedvis = cv2.bitwise_and(frame, frame, mask= maskRed)  

    segAmarillo = cv2.inRange(hsv,amarilloBajo,amarilloAlto)

    segVerde = cv2.inRange(hsv,verdeBajo,verdeAlto)

    segCyan = cv2.inRange(hsv,cyanBajo,cyanAlto)

    segNaranja = cv2.inRange(hsv,naranjaBajo,naranjaAlto)  

    segMorado = cv2.inRange(yuv,moradoAlto,moradoBajo) 

    segRosa = cv2.inRange(hsv,rosaBajo,rosaAlto)  

    cv2.imshow('azul',segAzul)
    cv2.imshow('rojo',maskRed)
    cv2.imshow('amarillo',segAmarillo)
    cv2.imshow('verde',segVerde)
    cv2.imshow('cyan',segCyan)
    cv2.imshow('naranja',segNaranja)
    cv2.imshow('rosa',segRosa)
    cv2.imshow('morado',segMorado)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): # Se esperan 30ms para el cierre de la ventana o hasta que el usuario precione la tecla q
        break

cap.release()
cv2.destroyAllWindows()