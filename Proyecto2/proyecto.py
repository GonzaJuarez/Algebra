from skimage.io import imread, imshow, imsave
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt
import numpy as np


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


# Variables

imagen1 = imread('C:/Users/Usuario/Desktop/Algebra/Algebra/Proyecto2/imagenes/lobo.jpg')
imagen2 = imread('C:/Users/Usuario/Desktop/Algebra/Algebra/Proyecto2/imagenes/perico.jpg')

recortar_imagen(imagen1, 'C:/Users/Usuario/Desktop/Algebra/Algebra/Proyecto2/imagenes/lobo_crop.jpg', 0, 400, 400, 800)

recortar_imagen(imagen2, 'C:/Users/Usuario/Desktop/Algebra/Algebra/Proyecto2/imagenes/perico_crop.jpg', 0, 400, 0, 400)

imagen1_crop = imread('C:/Users/Usuario/Desktop/Algebra/Algebra/Proyecto2/imagenes/lobo_crop.jpg')
imagen2_crop = imread('C:/Users/Usuario/Desktop/Algebra/Algebra/Proyecto2/imagenes/perico_crop.jpg')

imagen1_matriz = image_to_matiz(imagen1_crop)
imagen2_matriz = image_to_matiz(imagen2_crop)

imagen1_traspuesta = transpuesta(imagen1_matriz)
imagen2_traspuesta = transpuesta(imagen2_matriz)

imagen1_gris = convertir_a_escala_grises(imagen1_traspuesta)
imagen2_gris = convertir_a_escala_grises(imagen2_traspuesta)

imagen1_inversa = calcular_inversa_matriz(imagen1_gris)
imagen2_inversa = calcular_inversa_matriz(imagen2_gris)


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

print(f"Tamaño de la imagen 1: {imagen1_traspuesta.shape}")

print("")

print("Imagen 1 traspuesta:")
print(imagen1_traspuesta)

print("Imagen 2 traspuesta:")
print(imagen2_traspuesta)

print("")

print(f"Tamaño de la imagen 1: {imagen1_gris.shape}")
print(f"Tamaño de la imagen 2: {imagen2_gris.shape}")

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

plt.imshow(imagen1_gris,  cmap='gray')
plt.title("Imagen 1 traspuesta escala de grises")
plt.show()
plt.imshow(imagen2_gris,  cmap='gray')
plt.title("Imagen 2 traspuesta escala de grises")
plt.show()