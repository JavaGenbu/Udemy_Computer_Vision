import torch  # Importa PyTorch para el modelo YOLOv5
import cv2  # Importa OpenCV para operaciones de imagen
import numpy as np  # Importa NumPy para operaciones numéricas
import mss  # Importa MSS para capturar la pantalla
import time  # Importa la biblioteca time para medir el tiempo de procesamiento
import pygame  # Importa Pygame para mostrar la pantalla

# Inicializar Pygame
pygame.init()
screen = pygame.display.set_mode((1280, 720))  # Configura la ventana de Pygame con un tamaño más pequeño

# Cargar el modelo YOLOv5 entrenado con imágenes de 1920x1080
model_path = '../models/yolov5_results/weights/best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# Configurar la captura de pantalla utilizando MSS
sct = mss.mss()
monitor = sct.monitors[1]  # Selecciona el monitor 1 (puedes cambiar el índice según tu configuración)

# Definir la función para capturar la pantalla y realizar la inferencia en tiempo real
def real_time_inference():
    running = True  # Variable para controlar el bucle de inferencia
    while running:
        start_time = time.time()  # Registra el tiempo de inicio para calcular FPS

        # Capturar la pantalla del área especificada
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)  # Convierte la captura a un array de NumPy

        # Convertir la imagen a formato BGR (de BGRA)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # Realizar la inferencia directamente en la imagen original (1920x1080)
        results = model(img)

        # Procesar los resultados y dibujar las cajas delimitadoras en la imagen original
        for detection in results.xyxy[0]:
            # Extraer las coordenadas de las detecciones, la confianza y el ID de la clase
            x1, y1, x2, y2, confidence, class_id = detection
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convertir las coordenadas a enteros
            label = f'{model.names[int(class_id)]} {confidence:.2f}'  # Crear la etiqueta con el nombre de la clase y la confianza

            # Dibujar la caja delimitadora en la imagen original
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Añadir el texto de la etiqueta sobre la caja delimitadora
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Redimensionar la imagen para ajustarla a la ventana de Pygame
        img_resized = cv2.resize(img, (1280, 720))

        # Convertir la imagen BGR a RGB para Pygame
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        # Convertir el array de NumPy a una superficie de Pygame
        img_surface = pygame.surfarray.make_surface(img_rgb.transpose(1, 0, 2))

        # Mostrar la imagen en la ventana de Pygame
        screen.blit(img_surface, (0, 0))
        pygame.display.update()

        # Calcular y mostrar los cuadros por segundo (FPS)
        fps = 1 / (time.time() - start_time)
        print(f'FPS: {fps:.2f}')

        # Manejar eventos de Pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # Cerrar la ventana
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:  # Salir del bucle si se presiona la tecla 'q'
                running = False

    pygame.quit()  # Cerrar Pygame

# Ejecutar la función de inferencia en tiempo real si el script se ejecuta directamente
if __name__ == "__main__":
    real_time_inference()
