import numpy as np
import cv2

cap = cv2.VideoCapture(0)
flag = False

while True:
    # Captura cuadro a cuadro
    _, liveFrame = cap.read()
    
    if flag:
        subs = cv2.subtract(liveFrame, pastFrame)
        _, newDetection = cv2.threshold(subs, 130, 255, cv2.THRESH_BINARY)
        if (np.array_equal(newDetection,detection)):
            detection[:] = newDetection
            pastFrame[:] = liveFrame
        else:
            print('Objetos detectado')
            detection[:] = newDetection
            pastFrame[:] = liveFrame
    else:
        pastFrame = np.empty_like (liveFrame)
        detection = np.empty_like (liveFrame)
        newDetection = np.empty_like(liveFrame)
        pastFrame[:] = liveFrame
        flag = True

    cv2.imshow('escena', detection)
    if cv2.waitKey(1) & 0xFF == ord('q'): # Se esperan 30ms para el cierre de la ventana o hasta que el usuario precione la tecla q
        break

# Al terminar finalice la captura

cap.release()
cv2.destroyAllWindows()