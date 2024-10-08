from skimage.io import imread, imshow, imsave
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt
import numpy as np
import os

base_dir = os.path.dirname(__file__)

imagen1_path = os.path.join(base_dir, 'imagenes', 'lobo.jpg')
imagen2_path = os.path.join(base_dir, 'imagenes', 'perico.jpg')
imagen1_crop_path = os.path.join(base_dir, 'imagenes', 'lobo_crop.jpg')
imagen2_crop_path = os.path.join(base_dir, 'imagenes', 'perico_crop.jpg')

# Funciones
def recortar_imagen(image, ruta_img_crop, x_inicial, x_final, y_inicial, y_final):
    try:
        # Recortar la imagen
        image_crop = image[x_inicial:x_final, y_inicial:y_final]

        # Guardar la imagen recortada en la ruta indicada
        imsave(ruta_img_crop, img_as_ubyte(image_crop))

    except Exception as e:
        print("Ha ocurrido un error:", str(e))

def image_to_matiz(image):
    try:
        matriz_imagen = np.array(image)
        return matriz_imagen
    except Exception as e:
        print("Ha ocurrido un error:", str(e))

def transpuesta(matriz):
    try:
        matriz_transpuesta = np.transpose(matriz, (1, 0, 2))
        return matriz_transpuesta
    except Exception as e:
        print("Ha ocurrido un error:", str(e))

def convertir_a_escala_grises(imagen_color: np.ndarray) -> np.ndarray:
    try:
        if imagen_color.shape[-1] == 3:
            imagen_gris = np.mean(imagen_color, axis=2)
            return imagen_gris
        else:
            print("La imagen no tiene 3 canales (RGB).")
            return None

    except Exception as e:
        print("Ha ocurrido un error:", str(e))

def calcular_inversa_matriz(imagen_gris: np.ndarray) -> np.ndarray:
    try:
        if imagen_gris.shape[0] != imagen_gris.shape[1]:
            print("La matriz no es cuadrada, no se puede calcular la inversa.")
            return None

        inversa = np.linalg.inv(imagen_gris)
        return inversa
    except np.linalg.LinAlgError:
        print("La matriz no tiene inversa.")
        return None

def imagen_por_escalar(imagen_gris: np.ndarray, escalar: float) -> np.ndarray:
    imagen_ajustada = imagen_gris * escalar
    imagen_ajustada = np.clip(imagen_ajustada, 0, 255).astype(np.uint8)
    return imagen_ajustada
# Termina ajustando el contraste de la imagen.

def voltear_imagen(imagen: np.ndarray) -> np.ndarray:
    if len(imagen.shape) == 3:
        filas, columnas, canales = imagen.shape
    elif len(imagen.shape) == 2:
        filas, columnas = imagen.shape
        canales = 1

    identidad = np.eye(filas)

    W = np.fliplr(identidad)

    imagen_volteada = np.zeros_like(imagen)

    for i in range(filas):
        for j in range(columnas):
            if canales == 1:
                imagen_volteada[i, j] = imagen[(filas - 1 - i), j]
            else:
                imagen_volteada[i, j, :] = imagen[(filas - 1 - i), j, :]

    return imagen_volteada

def negativo_imagen(imagen):
    try:
        """
        numpy.full_like(a, fill_value, dtype=None, order='K', subok=True, shape=None, *, device=None)[source]
            Return a full array with the same shape and type as a given array.

            Parameters:
                a:array_like
                The shape and data-type of a define these same attributes of the returned array.

                fill_valuearray_like
                Fill value.
        """
        #se utiliza np.full_like para hace una matriz igual a la de la imagen pero con todo 255 dentro 
        matriz_auxiliar = np.full_like(imagen, 255)
        
        imagen_negativo = matriz_auxiliar - imagen
        return imagen_negativo

    except Exception as e:
        print("Error al calcular el negativo", str(e))


# Variables
imagen1 = imread(imagen1_path)
imagen2 = imread(imagen2_path)

recortar_imagen(imagen1, imagen1_crop_path, 0, 400, 400, 800)
recortar_imagen(imagen2, imagen2_crop_path, 0, 400, 0, 400)

imagen1_crop = imread(imagen1_crop_path)
imagen2_crop = imread(imagen2_crop_path)

imagen1_matriz = image_to_matiz(imagen1_crop)
imagen2_matriz = image_to_matiz(imagen2_crop)

imagen1_traspuesta = transpuesta(imagen1_matriz)
imagen2_traspuesta = transpuesta(imagen2_matriz)

imagen1_gris = convertir_a_escala_grises(imagen1_crop)
imagen2_gris = convertir_a_escala_grises(imagen2_crop)

imagen1_inversa = calcular_inversa_matriz(imagen1_gris)
imagen2_inversa = calcular_inversa_matriz(imagen2_gris)

imagen1_gris_escalar_alto = imagen_por_escalar(imagen1_gris, 5)
imagen1_gris_escalar_bajo = imagen_por_escalar(imagen1_gris, 0.2)

imagen1_volteada = voltear_imagen(imagen1_gris)

imagen1_negativa = negativo_imagen(imagen1_gris)

# Prints
print(f"Tamaño de la imagen 1: {imagen1.shape}")
print(f"Tamaño de la imagen 2: {imagen2.shape}")

print("")

print(f"Tamaño de la imagen 1 recortada: {imagen1_crop.shape}")
print(f"Tamaño de la imagen 2 recortada: {imagen2_crop.shape}")

print("")

print("Imagen 1:")
print(imagen1_matriz)

print("Imagen 2:")
print(imagen2_matriz)

print("")

print(f"Tamaño de la imagen 1 traspuesta: {imagen1_traspuesta.shape}")
print(f"Tamaño de la imagen 2 traspuesta: {imagen2_traspuesta.shape}")

print("")

print(f"Tamaño de la imagen 1 en escala de grises: {imagen1_gris.shape}")
print(f"Tamaño de la imagen 2 en escala de grises: {imagen2_gris.shape}")

print("")

print("Imagen 1 inversa:")
print(imagen1_inversa)

print("Imagen 2 inversa:")
print(imagen2_inversa)

# Plots
imshow(imagen1)
plt.title("Imagen 1")
plt.show()

imshow(imagen2)
plt.title("Imagen 2")
plt.show()

imshow(imagen1_crop)
plt.title("Imagen 1 recortada")
plt.show()

imshow(imagen2_crop)
plt.title("Imagen 2 recortada")
plt.show()

plt.imshow(imagen1_traspuesta)
plt.title("Imagen 1 traspuesta")
plt.show()

plt.imshow(imagen2_traspuesta)
plt.title("Imagen 2 traspuesta")
plt.show()

plt.imshow(imagen1_gris, cmap='gray')
plt.title("Imagen 1 en escala de grises")
plt.show()

plt.imshow(imagen2_gris, cmap='gray')
plt.title("Imagen 2 en escala de grises")
plt.show()

print(calcular_inversa_matriz(imagen1_gris))
print(calcular_inversa_matriz(imagen2_gris))

plt.subplot(2, 2, 1)
plt.imshow(imagen1_gris_escalar_alto, cmap='gray')
plt.title(f'Imagen 1 - Escalar (α={5})')
plt.show()

plt.subplot(2, 2, 1)
plt.imshow(imagen1_gris_escalar_bajo, cmap='gray')
plt.title(f'Imagen 1 - Escalar (α={0.2})')
plt.show()

plt.imshow(imagen1_volteada, cmap='gray')
plt.title("Imagen 1 volteada")
plt.show()

plt.imshow(imagen1_negativa, cmap='gray')
plt.title("Imagen 1 negativa")
plt.show()
