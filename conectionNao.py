import os
import numpy as np
import cv2
import tensorflow as tf
import socket


import mediapipe as mp

try:
    # Cargar el modelo para el reconocimiento en tiempo real
    model = tf.keras.models.load_model('asl_model.h5')

    # Inicializar MediaPipe para detección de manos
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1)
    mp_drawing = mp.solutions.drawing_utils

    # Definir dimensiones de entrada del modelo
    img_height, img_width = 64, 64  # Asegúrate de que coincida con el modelo

    # Configurar el socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('localhost', 12348))  # Conectarse al servidor en el puerto 12346

    # Captura de video para reconocimiento en tiempo real
    cap = cv2.VideoCapture(1)  # Cambiar a 0 para usar la cámara predeterminada

    print("Presiona 'q' para salir del reconocimiento en tiempo real.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convertir el marco a RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        # Procesar la mano detectada
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Recortar la región de la mano
                x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1])
                y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0])
                x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1])
                y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0])
                hand_frame = frame[y_min:y_max, x_min:x_max]

                if hand_frame.size > 0:
                    # Preprocesar la imagen de la mano para la predicción
                    hand_frame = cv2.resize(hand_frame, (img_width, img_height))  # Cambiar a (64, 64)
                    hand_frame = np.expand_dims(hand_frame, axis=0) / 255.0

                    # Predicción
                    predictions = model.predict(hand_frame)
                    predicted_class = np.argmax(predictions)
                    label = chr(predicted_class + 65)  # Convertir a letra (suponiendo que las clases son A-Z)

                    # Mostrar la predicción en el cuadro de video
                    cv2.putText(frame, "Letra: {}".format(label), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Enviar la predicción al servidor
                    client_socket.send(label.encode())  # Enviar la letra reconocida al servidor

        # Mostrar el cuadro de video
        cv2.imshow('Reconocimiento de Lenguaje de Señas', frame)

        # Salir con 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    client_socket.close()  # Cerrar el socket al final

except Exception as e:
    print("Ha ocurrido un error: {}".format(e))