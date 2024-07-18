import os  # Importa el módulo os para interactuar con el sistema de archivos

def update_labels(labels_folder):
    """
    Actualiza las etiquetas en los archivos de anotaciones, cambiando el índice de clase 15 a 0.

    Args:
        labels_folder (str): Carpeta que contiene los archivos de anotaciones.
    """
    # Verifica si la carpeta de etiquetas existe
    if not os.path.exists(labels_folder):
        raise FileNotFoundError(f"La carpeta de etiquetas {labels_folder} no existe.")

    # Recorre todos los archivos en la carpeta de etiquetas
    for label_file in os.listdir(labels_folder):
        if label_file.endswith('.txt'):  # Asegura que solo se procesen archivos .txt
            label_file_path = os.path.join(labels_folder, label_file)  # Ruta completa al archivo de etiquetas
            with open(label_file_path, 'r') as file:
                lines = file.readlines()  # Lee todas las líneas del archivo

            updated_lines = []
            for line in lines:
                parts = line.strip().split()
                if parts[0] == '15':  # Si la etiqueta es 15, cámbiala a 0
                    parts[0] = '0'
                updated_lines.append(' '.join(parts))

            with open(label_file_path, 'w') as file:
                file.write('\n'.join(updated_lines) + '\n')  # Escribe las líneas actualizadas al archivo

if __name__ == "__main__":
    # Define las rutas relativas a las carpetas de etiquetas de entrenamiento y validación
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Obtiene la ruta del directorio actual del script
    train_labels_folder = os.path.join(base_dir, '../data/processed/labels/train')  # Carpeta de etiquetas de entrenamiento
    val_labels_folder = os.path.join(base_dir, '../data/processed/labels/val')  # Carpeta de etiquetas de validación

    # Actualiza las etiquetas en las carpetas de entrenamiento y validación
    update_labels(train_labels_folder)
    update_labels(val_labels_folder)

    # Actualiza el archivo classes.txt para que solo contenga valorant_enemy
    classes_file = os.path.join(base_dir, '../data/annotations/classes.txt')
    with open(classes_file, 'w') as file:
        file.write('valorant_enemy\n')

    print("Actualización de etiquetas completada.")
