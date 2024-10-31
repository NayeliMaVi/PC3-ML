
import os
import numpy as np
import cv2
import tensorflow as tf
import socket


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import mediapipe as mp

# Ruta a tus carpetas de entrenamiento y prueba
train_dir = 'dataset/asl-train'
test_dir = 'dataset/asl-test'

# Parámetros del generador de datos y del modelo
batch_size = 32
img_height = 64
img_width = 64
epochs = 10

# Generador de datos para entrenamiento con aumentos
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

# Generador de datos para prueba (sin aumentos)
test_datagen = ImageDataGenerator(rescale=1./255)

# Cargar datos de entrenamiento con etiquetas
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='sparse'
)

# Función para cargar y preprocesar las imágenes de prueba
def load_test_images(test_dir, img_height, img_width):
    images = []
    labels = []
    for filename in os.listdir(test_dir):
        if filename.endswith("_test.jpg"):  # Asegura que los archivos coincidan con el formato
            img_path = os.path.join(test_dir, filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (img_height, img_width))
            img = img / 255.0  # Normalización
            images.append(img)
            label = ord(filename[0].upper()) - 65  # Convertir la letra inicial a un índice (A = 0, B = 1, ...)
            labels.append(label)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

# Cargar las imágenes de prueba
test_images, test_labels = load_test_images(test_dir, img_height, img_width)

# Definición del modelo CNN
def create_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(img_height, img_width, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(26, activation='softmax')  # 26 letras del alfabeto
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Crear y entrenar el modelo
model = create_model()
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=(test_images, test_labels),
    epochs=epochs
)

# Guardar el modelo entrenado
model.save('asl_model.h5')


# try:
#     # Cargar el modelo para el reconocimiento en tiempo real
#     model = tf.keras.models.load_model('asl_model.h5')

#     # Inicializar MediaPipe para detección de manos
#     mp_hands = mp.solutions.hands
#     hands = mp_hands.Hands(max_num_hands=1)
#     mp_drawing = mp.solutions.drawing_utils

#     # Definir dimensiones de entrada del modelo
#     img_height, img_width = 64, 64  

#     # Configurar el socket
#     client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     client_socket.connect(('localhost', 12348))  # Conectarse al servidor en el puerto 12346

#     # Captura de video para reconocimiento en tiempo real
#     cap = cv2.VideoCapture(1)  # Cambiar a 0

#     print("Presiona 'q' para salir del reconocimiento en tiempo real.")
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Convertir el marco a RGB
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         result = hands.process(rgb_frame)

#         # Procesar la mano detectada
#         if result.multi_hand_landmarks:
#             for hand_landmarks in result.multi_hand_landmarks:
#                 mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#                 # Recortar la región de la mano
#                 x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1])
#                 y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0])
#                 x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1])
#                 y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0])
#                 hand_frame = frame[y_min:y_max, x_min:x_max]

#                 if hand_frame.size > 0:
#                     # Preprocesar la imagen de la mano para la predicción
#                     hand_frame = cv2.resize(hand_frame, (img_width, img_height))  
#                     hand_frame = np.expand_dims(hand_frame, axis=0) / 255.0

#                     # Predicción
#                     predictions = model.predict(hand_frame)
#                     predicted_class = np.argmax(predictions)
#                     label = chr(predicted_class + 65)  # Convertir a letra (suponiendo que las clases son A-Z)

#                     # Mostrar la predicción en el cuadro de video
#                     cv2.putText(frame, "Letra: {}".format(label), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#                     # Enviar la predicción al servidor
#                     client_socket.send(label.encode())  # Enviar la letra reconocida al servidor

#         # Mostrar el cuadro de video
#         cv2.imshow('Reconocimiento de Lenguaje de Señas', frame)

#         # Salir con 'q'
#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     client_socket.close()  # Cerrar el socket al final

# except Exception as e:
#     print("Ha ocurrido un error: {}".format(e))