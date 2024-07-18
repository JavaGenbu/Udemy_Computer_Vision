import os  # Importa el módulo os para interactuar con el sistema de archivos
import sys  # Importa el módulo sys para interactuar con el intérprete de Python
import cv2  # Importa la biblioteca OpenCV para procesamiento de imágenes y videos
import numpy as np  # Importa NumPy para operaciones con matrices
import pygame  # Importa Pygame para la visualización de la pantalla
import mss  # Importa MSS para la captura de pantalla
import time  # Importa el módulo time para manejar operaciones relacionadas con el tiempo
import torch  # Importa PyTorch para trabajar con modelos de aprendizaje profundo

# Asegurarse de que la ruta de yolov5 esté en sys.path
# Esto es necesario para que Python pueda encontrar los archivos necesarios de YOLOv5
yolov5_path = os.path.join(os.path.dirname(__file__), '../yolov5')
if yolov5_path not in sys.path:
    sys.path.insert(0, yolov5_path)

# Importa funciones y clases necesarias de YOLOv5
# DetectMultiBackend es la clase principal para cargar y ejecutar el modelo
# non_max_suppression se utiliza para eliminar detecciones redundantes
# scale_boxes se utiliza para ajustar las coordenadas de las cajas de detección
# select_device se utiliza para seleccionar el dispositivo de computación (CPU o GPU)
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.utils.torch_utils import select_device

# Definición de la clase ObjectDetector
class ObjectDetector:
    def __init__(self, weights_path, device='cpu', imgsz=640):
        # Inicializa el dispositivo (CPU o GPU) para la inferencia
        self.device = select_device(device)
        # Carga el modelo YOLOv5 con los pesos especificados
        self.model = DetectMultiBackend(weights_path, device=self.device, dnn=False)
        # Obtiene propiedades del modelo, como la distancia entre pasos (stride), nombres de las clases, etc.
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        # Define el tamaño de las imágenes de entrada para el modelo
        self.imgsz = imgsz

    def capture_screen(self, region=None):
        # Utiliza MSS para capturar la pantalla
        with mss.mss() as sct:
            # Selecciona el monitor a capturar
            monitor = sct.monitors[1] if region is None else region
            # Captura la pantalla
            screen = sct.grab(monitor)
            # Convierte la captura a una matriz de NumPy
            img = np.array(screen)
            # Convierte la imagen capturada a escala de grises
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            # Aplica desenfoque gaussiano para suavizar la imagen
            img_blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
            # Redimensiona la imagen a 640x360
            img_resized = cv2.resize(img_blurred, (640, 360))
            return img_resized  # Devuelve la imagen procesada

    def run_detection(self, frame):
        # Redimensiona la imagen al tamaño esperado por el modelo
        img = cv2.resize(frame, (self.imgsz, self.imgsz))
        # Convierte la imagen de escala de grises a RGB replicando los canales
        img = np.stack([img, img, img], axis=-1)
        # Transpone y ajusta la imagen al formato esperado por YOLOv5
        img = img.transpose((2, 0, 1))[::-1]
        # Asegura que la imagen esté en memoria contigua para optimización
        img = np.ascontiguousarray(img)

        # Convierte la imagen a un tensor y la envía al dispositivo (CPU o GPU)
        img = torch.from_numpy(img).to(self.device)
        # Convierte el tensor a FP16 si el modelo lo requiere, de lo contrario, a float32
        img = img.half() if self.model.fp16 else img.float()
        # Normaliza los valores de los píxeles a [0, 1]
        img /= 255.0
        # Añade una dimensión al tensor si es necesario
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Realiza la inferencia utilizando el modelo cargado
        pred = self.model(img, augment=False, visualize=False)
        # Aplica supresión de no-máximos para eliminar detecciones redundantes
        pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)
        return pred  # Devuelve las predicciones

    def process_detections(self, frame, detections):
        # Itera sobre las detecciones por cada imagen
        for i, det in enumerate(detections):
            # Crea una copia del frame original
            im0 = frame.copy()
            if len(det):
                # Escala las cajas de detección de acuerdo con el tamaño original de la imagen
                det[:, :4] = scale_boxes((self.imgsz, self.imgsz), det[:, :4], im0.shape).round()
                # Itera sobre cada detección y dibuja la caja en la imagen
                for *xyxy, conf, cls in reversed(det):
                    # Crea la etiqueta con el nombre de la clase y la confianza
                    label = f'{self.names[int(cls)]} {conf:.2f}'
                    # Dibuja la caja en la imagen
                    self.plot_one_box(xyxy, im0, label=label, color=(255, 0, 0), line_thickness=2)
        return im0  # Devuelve la imagen con las detecciones

    def plot_one_box(self, xyxy, im, color=(128, 128, 128), label=None, line_thickness=3):
        # Define el grosor de la línea
        tl = line_thickness or int(round(0.002 * max(im.shape[0:2])))
        # Define el color de la caja
        color = color or [random.randint(0, 255) for _ in range(3)]
        # Define las coordenadas de la caja
        c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
        # Dibuja la caja en la imagen
        cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            # Define el grosor de la fuente
            tf = max(tl - 1, 1)
            # Obtiene el tamaño del texto
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            # Define las coordenadas del fondo del texto
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            # Dibuja el fondo del texto
            cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)
            # Escribe el texto en la imagen
            cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def main():
    # Define la ruta al archivo de pesos del modelo YOLOv5
    weights_path = os.path.join(os.path.dirname(__file__), '../models/yolov5_results/weights/best.pt')
    # Inicializa el detector de objetos
    detector = ObjectDetector(weights_path, device='cuda:0' if torch.cuda.is_available() else 'cpu')

    # Define la resolución para la visualización
    img_shape = (640, 360)
    # Inicializa Pygame
    pygame.init()
    # Crea una ventana de Pygame con la resolución definida
    screen = pygame.display.set_mode(img_shape)
    # Establece el título de la ventana
    pygame.display.set_caption("Screen Capture and Detection")

    # Contador de frames
    frame_count = 0
    # Registra el tiempo de inicio
    start_time = time.time()

    try:
        while True:  # Bucle infinito para la captura continua
            # Registra el tiempo de inicio del bucle
            loop_start_time = time.time()
            # Captura la pantalla en escala de grises
            img_gray = detector.capture_screen()

            # Realiza la detección
            detections = detector.run_detection(img_gray)
            # Procesa las detecciones y obtiene la imagen resultante
            result = detector.process_detections(img_gray, detections)

            # Calcula los FPS
            frame_count += 1
            # Calcula el tiempo transcurrido
            elapsed_time = time.time() - start_time
            if elapsed_time > 0:
                # Calcula los FPS
                fps = frame_count / elapsed_time

            # Muestra los FPS en la imagen
            cv2.putText(result, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # Convierte la imagen al formato de Pygame
            result_rgb = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
            surface = pygame.surfarray.make_surface(result_rgb.swapaxes(0, 1))

            # Muestra la imagen en la ventana de Pygame
            screen.blit(surface, (0, 0))
            pygame.display.flip()

            # Maneja eventos de Pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    return

            # Espera el tiempo necesario para mantener 60 FPS
            loop_elapsed_time = time.time() - loop_start_time
            sleep_time = max(1.0 / 60 - loop_elapsed_time, 0)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("Script interrupted by user.")
    finally:
        # Cierra Pygame
        pygame.quit()

if __name__ == "__main__":
    # Ejecuta la función principal
    main()
