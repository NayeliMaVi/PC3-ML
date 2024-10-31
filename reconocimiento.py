import cv2
import numpy as np
from tensorflow.keras.models import load_model
import socket
import os

# Cargar el modelo entrenado
model = load_model('asl_model.h5')

# Dimensiones de entrada de la imagen
img_height = 64
img_width = 64

# Función para cargar y preprocesar la imagen
def preprocess_image(image_path):
    # Cargar la imagen usando OpenCV
    image = cv2.imread(image_path)
    
    if image is None:
        print("Error: No se pudo cargar la imagen. Verifica la ruta.")
        return None  # Cambiado para manejar mejor el error
    
    # Redimensionar la imagen a 64x64
    image = cv2.resize(image, (img_height, img_width))
    
    # Convertir BGR a RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Normalizar la imagen
    image = image / 255.0
    
    # Cambiar la forma de la imagen a (1, 64, 64, 3) para el modelo
    image = np.expand_dims(image, axis=0)
    
    return image

# Configuración del socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('localhost', 12348))  # Cambia al puerto en el que el servidor escucha

while True:
    # Solicitar la ruta de la imagen al usuario
    image_path = input("Ingresa la ruta de la imagen (o 'exit' para salir): ")
    
    if image_path.lower() == 'exit':
        break  # Salir del bucle si el usuario escribe 'exit'

    # Preprocesar la imagen
    processed_image = preprocess_image(image_path)
    
    if processed_image is None:
        continue  # Si la imagen no se carga, pedir otra entrada
    
    # Realizar la predicción
    prediction = model.predict(processed_image)
    
    # Obtener la clase con la mayor probabilidad
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    # Convertir el índice a la letra correspondiente
    predicted_letter = chr(predicted_class + 65)  # Sumar 65 para convertir a letra ASCII
    
    # Enviar la letra predicha al servidor
    client_socket.send(predicted_letter.encode())  # Enviar la letra como un mensaje

    print(f"La seña de la mano predicha es: {predicted_letter}")

# Cerrar el socket después de enviar la predicción
client_socket.close()