import cv2  # Importa la biblioteca OpenCV para el procesamiento de imágenes y video
import numpy as np  # Importa numpy para manejar matrices
import os  # Importa os para interactuar con el sistema operativo y manejar archivos y carpetas
import time  # Importa time para manejar operaciones relacionadas con el tiempo
from datetime import datetime  # Importa datetime para manejar fechas y horas
from PIL import ImageGrab  # Importa ImageGrab de Pillow para la captura de pantalla

def record_screen_and_extract_frames(output_folder):
    """
    Graba la pantalla y extrae frames automáticamente a 1 frame por segundo hasta que se detenga el script manualmente.

    Args:
        output_folder (str): La carpeta donde se guardarán los frames capturados.
    """
    # Verifica si la carpeta de salida especificada existe
    if not os.path.exists(output_folder):
        # Si la carpeta de salida no existe, se crea
        os.makedirs(output_folder)

    frame_count = 0  # Inicializa el contador de frames en 0

    try:
        # Bucle infinito para capturar la pantalla continuamente
        while True:
            # Captura una captura de pantalla de toda la pantalla
            img = ImageGrab.grab()
            # Convierte la imagen capturada a una matriz numpy
            frame = np.array(img)
            # Convierte la imagen de RGB a escala de grises
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            # Aplica un desenfoque gaussiano a la imagen
            frame = cv2.GaussianBlur(frame, (5, 5), 0)
            # Redimensiona la imagen a 640x480
            frame = cv2.resize(frame, (640, 360))

            # Define el nombre del archivo de frame con un contador de 4 dígitos, asegurando que los archivos se nombren secuencialmente
            frame_filename = os.path.join(output_folder, f'frame_{frame_count:04d}.jpg')
            # Guarda el frame como un archivo .jpg en la carpeta de salida especificada
            cv2.imwrite(frame_filename, frame)

            print(f'Guardado: {frame_filename}')  # Imprime un mensaje de confirmación

            # Incrementa el contador de frames en 1
            frame_count += 1
            # Espera 1 segundo antes de capturar el siguiente frame para mantener la tasa de 1 frame por segundo
            time.sleep(1)
    finally:
        # Libera todos los recursos de ventanas de OpenCV, en caso de que alguna ventana haya sido creada
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Obtiene la marca de tiempo actual en el formato YYYYMMDD_HHMMSS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Define la carpeta de salida para los frames, incluyendo una subcarpeta única basada en la marca de tiempo
    output_folder = os.path.join(os.getcwd(), '..', 'data', 'raw', f'gameplay_{timestamp}')

    # Llama a la función para grabar la pantalla y extraer frames, pasando la carpeta de salida como argumento
    record_screen_and_extract_frames(output_folder)
