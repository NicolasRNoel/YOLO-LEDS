import cv2
from ultralytics import YOLO


model = YOLO('yolov8n.pt')


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("No se pudo acceder a la cámara.")
        break

   
    resultados = model(frame, verbose=False)

    carro_detectado = False
    moto_detectada = False

    
    for r in resultados:
        for caja in r.boxes:
            clase = int(caja.cls[0]) 
            
           
            if clase == 2:
                carro_detectado = True
            elif clase == 3:
                moto_detectada = True

   
    color_rojo = (0, 0, 255) if carro_detectado else (50, 50, 100)
    cv2.circle(frame, (50, 50), 30, color_rojo, -1)
    cv2.putText(frame, "CARRO", (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color_rojo, 2)

   
    color_verde = (0, 255, 0) if moto_detectada else (50, 100, 50)
    cv2.circle(frame, (50, 150), 30, color_verde, -1)
    cv2.putText(frame, "MOTO", (100, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, color_verde, 2)

    
    cv2.imshow('Simulador YOLO - Presiona Q para salir', frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()