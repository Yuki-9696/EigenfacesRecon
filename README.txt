El proyecto consiste en lograr reconocimiento facial en una imagen mediante el uso de Eigenfaces. Para esto, es posible tomar como base el documento "Eigenfaces for Face Detection/Recognition" basado en "Eigenfaces for Recognition" de M. Turk and A. Pentland. 
Este documento comprende de tres partes, la idea y funcionamiento principal del método Eigenfaces, y sus subsecuentes aplicaciones para reconocimiento facial y detección facial. 
A partir del análisis del documento se realizó la síntesis del presente proyecto, siendo su contenido el siguiente:
1. Construcción del set de entrenamiento.
1.1 Obtención de imágenes para entrenamiento. Para esto se descargó un paquete de imágenes de uso libre de "FEI Face Database" (https://fei.edu.br/~cet/facedatabase.html).
1.2 Preprocesamiento del conjunto de entrenamiento. Las imágenes se encuentran a color, y nos interesa que estén en formato Blanco y Negro para facilitar el proceso, además de que para este proyecto se propone el uso de dimensiones de 55x55 pixeles.
2. Computo de la matriz de eigenvectores y eigenvalores "Eigenfaces".
2.1 Se lee cada imagen de la carpeta de entrenamiento que formará la matriz de imágenes de entrenamiento.
2.1.1 Se lee la imagen.
2.1.2 Se convierte a blanco y negro.
2.1.3 Se reescala la imagen a la medida deseada. Para fines del cálculo, es importante que sea una imagen cuadrada.
2.1.4 Se ecualiza la imagen*. Este paso no es necesario para lograr el método, pero ayuda a resaltar las características del conjunto de entrenamiento y las de la imagen a reconocer.
2.1.5 Se convierte la imagen de matriz a un vector columna
2.1.6 Se añade el vector obtenido a una nueva matriz, que será la matriz de entrenamiento. Cada vector columna compondrá un renglón de esta matriz, resultando en una matriz de dimensiones (columnas = # de pixeles, filas = # de imágenes).
3. Se obtiene el vector promedio de la matriz (rostro promedio)
4. Se substrae el rostro promedio de la matriz de entrenamiento
5. Se computa la matriz de covarianzas.
6. Se computan los eigenvectores de la matriz A^T*A.
7. Se conservan solo los k eigenvectores más importantes*. No es necesario, pero reducirá el tiempo de cómputo.

Para detección de rostros, se necesita comparar la imagen que se desee detectar con la traspuesta de la matriz de Eigenfaces, al resultado se le calcula la distancia euclidiana respecto a las Eigenfaces. El resultado será un valor que indicará que tan parecido es la imagen evaluada a un rostro promedio. Es posible comparar este valor con un umbral ajustable para discriminar rostros de no-rostros en una imagen.

Para analizar una variedad de imágenes, se puede aplicar el método sin reescalar la imagen original. Esto se logra aplicando la detección a un área determinada de la imagen, cuando se detecte un rostro (en función del umbral) marcar la detección.

Para esto se propone lo siguiente:

1. Ciclo de detección.
1.1 Se realiza un ciclo donde se aplica el método Eigenfaces a una ventana, barriendo la imagen pixel por pixel. El resultado de cada ciclo se almacenará en una matriz de errores.
2. Ciclo de calculo de mínimos. Considerando la función del ciclo de detección, la matriz de errores contendrá valores altos y bajos. Para encontrar cuales de los valores son mínimos, se aplica un segundo barrido donde se detecten los mínimos. Para esto:
2.1 Se hace un barrido de la matriz de errores utilizando una ventana de mismas dimensiones que las imágenes de entrenamiento. Se calcula si el valor del centro de la matriz es el mínimo de la ventana, y en caso de ser mínimo se marca en la imagen original (o una copia) la posición circundante al pixel de error mínimo de la matriz de errores.
 

