import os  # Importa el módulo os para interactuar con el sistema de archivos
import shutil  # Importa el módulo shutil para copiar archivos
import random  # Importa el módulo random para mezclar aleatoriamente las imágenes


def split_data(images_folder, labels_folder, train_images_folder, val_images_folder, train_labels_folder,
               val_labels_folder, train_ratio=0.8):
    """
    Divide las imágenes y las anotaciones en carpetas de entrenamiento y validación.

    Args:
        images_folder (str): Carpeta que contiene las imágenes originales.
        labels_folder (str): Carpeta que contiene las anotaciones correspondientes.
        train_images_folder (str): Carpeta donde se almacenarán las imágenes de entrenamiento.
        val_images_folder (str): Carpeta donde se almacenarán las imágenes de validación.
        train_labels_folder (str): Carpeta donde se almacenarán las etiquetas de entrenamiento.
        val_labels_folder (str): Carpeta donde se almacenarán las etiquetas de validación.
        train_ratio (float): Proporción de datos que se utilizarán para el entrenamiento.
    """
    # Verifica si las carpetas de imágenes y etiquetas originales existen
    if not os.path.exists(images_folder):
        raise FileNotFoundError(f"La carpeta de imágenes {images_folder} no existe.")
    if not os.path.exists(labels_folder):
        raise FileNotFoundError(f"La carpeta de etiquetas {labels_folder} no existe.")

    # Verifica si las carpetas de entrenamiento existen; si no, las crea
    if not os.path.exists(train_images_folder):
        os.makedirs(train_images_folder)
    if not os.path.exists(train_labels_folder):
        os.makedirs(train_labels_folder)

    # Verifica si las carpetas de validación existen; si no, las crea
    if not os.path.exists(val_images_folder):
        os.makedirs(val_images_folder)
    if not os.path.exists(val_labels_folder):
        os.makedirs(val_labels_folder)

    # Obtiene una lista de todos los archivos de imágenes en la carpeta de imágenes
    images = [f for f in os.listdir(images_folder) if f.endswith('.jpg')]
    random.shuffle(images)  # Mezcla aleatoriamente la lista de imágenes

    # Calcula la cantidad de imágenes que se utilizarán para el entrenamiento
    train_count = int(len(images) * train_ratio)
    # Divide la lista de imágenes en conjuntos de entrenamiento y validación
    train_images = images[:train_count]
    val_images = images[train_count:]

    def copy_files(image_list, source_images_folder, source_labels_folder, dest_images_folder, dest_labels_folder):
        """
        Copia las imágenes y sus etiquetas correspondientes a la carpeta de destino.

        Args:
            image_list (list): Lista de nombres de archivos de imágenes a copiar.
            source_images_folder (str): Carpeta de origen de las imágenes.
            source_labels_folder (str): Carpeta de origen de las etiquetas.
            dest_images_folder (str): Carpeta de destino de las imágenes.
            dest_labels_folder (str): Carpeta de destino de las etiquetas.
        """
        for image in image_list:
            image_path = os.path.join(source_images_folder, image)  # Ruta completa a la imagen
            label_path = os.path.join(source_labels_folder,
                                      image.replace('.jpg', '.txt'))  # Ruta completa a la etiqueta correspondiente

            # Verifica si la etiqueta correspondiente existe
            if os.path.exists(label_path):
                shutil.copy(image_path,
                            os.path.join(dest_images_folder, image))  # Copia la imagen a la carpeta de destino
                shutil.copy(label_path, os.path.join(dest_labels_folder, image.replace('.jpg',
                                                                                       '.txt')))  # Copia la etiqueta a la carpeta de destino
                os.remove(image_path)  # Elimina la imagen original
                os.remove(label_path)  # Elimina la etiqueta original

    # Copia las imágenes y etiquetas de entrenamiento a la carpeta de entrenamiento
    copy_files(train_images, images_folder, labels_folder, train_images_folder, train_labels_folder)
    # Copia las imágenes y etiquetas de validación a la carpeta de validación
    copy_files(val_images, images_folder, labels_folder, val_images_folder, val_labels_folder)


if __name__ == "__main__":
    # Define las rutas absolutas a las carpetas de imágenes y etiquetas originales
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Obtiene la ruta del directorio actual del script
    images_folder = os.path.join(base_dir, '../data/annotations/yolov5_results/images')  # Carpeta de imágenes originales
    labels_folder = os.path.join(base_dir, '../data/annotations/yolov5_results/labels')  # Carpeta de etiquetas originales

    # Define las rutas absolutas a las carpetas de entrenamiento y validación
    train_images_folder = os.path.join(base_dir, '../data/processed/images/train')
    val_images_folder = os.path.join(base_dir, '../data/processed/images/val')
    train_labels_folder = os.path.join(base_dir, '../data/processed/labels/train')
    val_labels_folder = os.path.join(base_dir, '../data/processed/labels/val')

    # Llama a la función split_data para dividir y organizar los datos en entrenamiento y validación
    split_data(images_folder, labels_folder, train_images_folder, val_images_folder, train_labels_folder,
               val_labels_folder)
