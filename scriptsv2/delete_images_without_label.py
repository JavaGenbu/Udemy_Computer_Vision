import os

def remove_images_without_labels(images_folder, labels_folder):
    """
    Elimina imágenes en la carpeta de imágenes si no tienen una etiqueta correspondiente en la carpeta de etiquetas.

    Args:
        images_folder (str): Carpeta que contiene las imágenes.
        labels_folder (str): Carpeta que contiene las etiquetas correspondientes.
    """
    # Itera sobre los archivos en la carpeta de imágenes
    for image_file in os.listdir(images_folder):
        if image_file.endswith('.jpg'):
            image_path = os.path.join(images_folder, image_file)
            label_path = os.path.join(labels_folder, image_file.replace('.jpg', '.txt'))

            # Si la etiqueta no existe, elimina la imagen
            if not os.path.exists(label_path):
                os.remove(image_path)
                print(f'Eliminada imagen sin etiqueta: {image_path}')

# Definir las rutas absolutas a las carpetas de imágenes y etiquetas de entrenamiento y validación
base_dir = os.path.dirname(os.path.abspath(__file__))  # Obtiene la ruta del directorio actual del script
train_images_folder = os.path.join(base_dir, '../data/processed/images/train')
val_images_folder = os.path.join(base_dir, '../data/processed/images/val')
train_labels_folder = os.path.join(base_dir, '../data/processed/labels/train')
val_labels_folder = os.path.join(base_dir, '../data/processed/labels/val')

# Eliminar imágenes sin etiquetas en las carpetas de entrenamiento y validación
remove_images_without_labels(train_images_folder, train_labels_folder)
remove_images_without_labels(val_images_folder, val_labels_folder)
