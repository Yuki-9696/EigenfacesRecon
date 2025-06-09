"""
Reconocimiento facial mediante metodo de EigenFaces
"""
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

source_folder_path = "C:/ReconEigenfaces/frontalimages" # Carpeta fuente pre-procesamiento
train_folder_path = "C:/ReconEigenfaces/Train" # Carpeta de entrenamiento
test_folder_path = "C:/ReconEigenfaces/Train" # Carpeta de evaluacion
img = cv2.imread('C:/ReconEigenfaces/Graduacion_small.jpg') # Imagen grupal a detectar

train_files = os.listdir(train_folder_path)  #Obtiene la lista de las imagenes de entrenamiento     
test_files = os.listdir(test_folder_path)
train_matrix = []
error_map = []
umbral_error = 1200 # 
window_size = 55
step = 1
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w = gray.shape
lado = (27)
E = 1000*np.ones(img.shape)
error_image = np.full((h, w), np.nan)
img_deteccion = img.copy()

# Función para reconocer una nueva imagen
def recognize_face(image_path, mean_face, eigenfaces, train_projections, train_files):
    # Leer y procesar imagen de prueba
    test_image = cv2.imread(image_path)
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    test_image = cv2.resize(test_image, (window_size, window_size))
    test_image = test_image.reshape(-1, 1)
    test_image_norm = test_image - mean_face  # Restar rostro promedio
    test_projection = np.dot(eigenfaces.T, test_image_norm) # Proyectar la imagen de prueba
    distances = np.linalg.norm(train_projections - test_projection, axis=0) # Calcular distancias con todas las imágenes de entrenamiento 
    min_index = np.argmin(distances) # Encontrar el índice del vector más cercano
    min_distance = distances[min_index] # 
    umbral = 4000 # Configuración del umbral de distancia
    if min_distance < umbral:
        recognized_file = train_files[min_index]
        print(f"Rostro reconocido: {recognized_file}")
    else:
        print("Rostro no reconocido.")

# Función de detección de rostros en ventana que devuelve el error
def detectar_error(ventana, mean_face, eigenfaces):
    ventana = cv2.resize(ventana, (window_size, window_size))
    ventana = cv2.equalizeHist(ventana)
    vec = ventana.flatten().reshape(-1,1).astype(np.float32)
    vec_norm = vec - mean_face
    w = np.dot(eigenfaces.T, vec_norm)   # Proyección
    reconstruido = np.dot(eigenfaces, w)
    error = np.linalg.norm(vec_norm - reconstruido) #calcula la distancia euclidiana entre dos vectores
    return error

for idx, train_file in enumerate(train_files):
    train_image = train_folder_path + "/" + train_file  #Se define cada imagen como la combinación del directorio y el nombre
    train_image = cv2.imread(train_image)    # Leer la imagen (como matriz NxN)
    train_image = cv2.cvtColor(train_image, cv2.COLOR_BGR2GRAY)     #Verifica que las imagenes sean Blanco y negro
    train_image = cv2.resize(train_image, (window_size, window_size))
    train_image = cv2.equalizeHist(train_image)
    train_image = train_image.reshape(-1,1) #( # de filas, # de columnas) Así tendrá las filas necesarias para que sea una matriz de 1 columna
    if len(train_matrix) == 0:  #Verifica que la matriz esté vacia (Error que sucede al concatenar una matriz con una vacia)
        train_matrix = train_image  #La primer imagen (Vector) se inserta directamente en la matriz para darle forma
    else:
        train_matrix = np.concatenate((train_matrix, train_image), axis=1)  #Concatena el vector columna de la imagen nueva en la matriz de covarianzas 

C_train = np.array(train_matrix) #Asegurarse que sea un np.array
M=C_train.shape[1]
mean_face = (np.sum(C_train, axis=1)) / M
mean_face = mean_face.reshape(-1, 1)    # Convierte la imagen promedio a un vector columna
C_norm = C_train - mean_face
ATA = np.dot(C_norm.T, C_norm)
valores, vectores = np.linalg.eig(ATA)
indices_ordenados = np.argsort(valores)[::-1]  # np.argsort devuelve los indices que ordenarian el array (de menor a mayor)
                                               # [::-1] invierte el orden para resultar de mayor a menor (mas importante primero)
valores_ordenados = valores[indices_ordenados] #Ordena los eigenvalores según su importancia (de mayor a menor) acorde al indice
vectores_ordenados = vectores[:, indices_ordenados] #Ordena los eigenvectores según su importancia (de mayor a menor) acorde al indice
eigenfaces = C_norm @ vectores_ordenados # @ Indica una multiplicacion de matrices
eigenfaces = eigenfaces / np.linalg.norm(eigenfaces, axis=0)
train_projections = np.dot(eigenfaces.T, C_norm)
numero_eigenfaces = eigenfaces.shape[1]

for ren in range(lado, h - lado, step):
    for col in range(lado, w - lado, step):
        R = gray[ren-lado:ren+lado+1, col-lado:col+lado+1]
        error = detectar_error(R, mean_face, eigenfaces)
        error_map.append((ren, col, ren+window_size, col+window_size, error))
        error_image[ren, col] = error

# Deteccion de Errores Mínimos 
for y in range(lado, h - lado):
    for x in range(lado, w - lado):
        ventana_error = error_image[y-lado:y+lado+1, x-lado:x+lado+1] 
        if error_image[y, x] == np.min(ventana_error):
            if error_image[y, x] < umbral_error:
                top_left = (x - window_size // 2, y - window_size // 2)
                bottom_right = (x + window_size // 2, y + window_size // 2)
                cv2.rectangle(img_deteccion, top_left, bottom_right, (0, 255, 0), 2)
                cv2.putText(img_deteccion, f"{error_image[y,x]:.0f}", (x, y-0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

cv2.imshow("Caras", img_deteccion)
cv2.waitKey(0)
cv2.destroyAllWindows()
plt.imshow(cv2.cvtColor(img_deteccion, cv2.COLOR_BGR2RGB))
plt.title(f"Detección de rostros con umbral: {umbral_error}")#revisar
plt.axis('off')
plt.show()

