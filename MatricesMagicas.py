#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np

# Define las matrices
A = np.array([[1, 2, 3],
              [4, 5, 6]])

B = np.array([[7, 8],
              [9, 10],
              [11, 12]])

# Multiplica las matrices utilizando np.dot
# C = np.dot(A, B)

# Alternativamente, puedes usar el operador @
C = A @ B

print("Matriz A:")
print(A)

print("\nMatriz B:")
print(B)

print("\nProducto de A y B (A * B):")
print(C)


# In[4]:


# Cuadrado Mágico con Constante 15
A = np.array([[8, 1, 6],
              [3, 5, 7],
              [4, 9, 2]])

# Cuadrado Mágico con Constante 21
B = np.array([[10, 3, 8],
              [5, 7, 9],
              [6, 11, 4]])

# Multiplicar las matrices
C = np.dot(A, B)

# Imprimir las matrices y el resultado
print("Cuadrado Mágico A (constante 15):")
print(A)

print("\nCuadrado Mágico B (constante 21):")
print(B)

print("\nProducto de A y B:")
print(C)


# In[6]:


import numpy as np

def create_encoding_matrix(n, constant_value):
    """Crear una matriz de codificación con una diagonal principal constante."""
    matrix = np.zeros((n, n), dtype=int)
    # Configurar la diagonal principal con el valor constante
    np.fill_diagonal(matrix, constant_value)
    # Llenar otros valores con un patrón específico
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i, j] = (i + j) % 10 + 1  # Puedes ajustar los valores aquí
    return matrix

def encode_data(data_matrix, encoding_matrix):
    """Codificar datos utilizando la matriz de codificación."""
    return np.dot(data_matrix, encoding_matrix)

def decode_data(encoded_matrix, encoding_matrix):
    """Decodificar datos utilizando la matriz inversa de la matriz de codificación."""
    encoding_matrix_inv = np.linalg.inv(encoding_matrix)
    return np.dot(encoded_matrix, encoding_matrix_inv)

# Parámetros
n = 3  # Orden de la matriz
constant_value = 7  # Valor constante en la diagonal principal
data = np.array([[8, 1, 6],
              [3, 5, 7],
              [4, 9, 2]])

# Crear matriz de codificación
encoding_matrix = create_encoding_matrix(n, constant_value)

# Codificar datos
encoded_data = encode_data(data, encoding_matrix)
print("Datos Codificados:")
print(encoded_data)

# Decodificar datos
decoded_data = decode_data(encoded_data, encoding_matrix)
print("\nDatos Decodificados:")
print(decoded_data)


# In[10]:


import numpy as np
from scipy.linalg import lu

def create_magic_square(n):
    """Crear una matriz cuadrada mágica de tamaño n x n."""
    if n % 2 == 1:
        # Algoritmo de construcción para cuadrados mágicos de orden impar
        magic_square = np.zeros((n, n), dtype=int)
        i, j = 0, n // 2
        for num in range(1, n * n + 1):
            magic_square[i, j] = num
            i, j = (i - 1) % n, (j + 1) % n
            if magic_square[i, j]:
                i, j = (i + 2) % n, (j - 1) % n
        return magic_square
    else:
        raise ValueError("El tamaño debe ser impar para este método")

def create_encoding_matrix_from_magic(magic_square):
    """Crear una matriz de codificación basada en una matriz cuadrada mágica."""
    n = magic_square.shape[0]
    encoding_matrix = np.copy(magic_square)
    
    # Alterar la matriz mágica para crear una matriz de codificación
    for i in range(n):
        for j in range(n):
            if i != j:
                encoding_matrix[i, j] = (encoding_matrix[i, j] + (i + j)) % 10 + 1
    
    return encoding_matrix

def encode_data(data_matrix, encoding_matrix):
    """Codificar datos utilizando la matriz de codificación."""
    return np.dot(data_matrix, encoding_matrix)

def decode_data(encoded_matrix, encoding_matrix):
    """Decodificar datos utilizando la matriz inversa de la matriz de codificación."""
    encoding_matrix_inv = np.linalg.inv(encoding_matrix)
    return np.dot(encoded_matrix, encoding_matrix_inv)

# Parámetros
n = 3  # Tamaño de la matriz
data = np.array([[8, 1, 6],
                 [3, 5, 7],
                 [4, 9, 2]])

# Crear matriz cuadrada mágica
magic_square = create_magic_square(n)
print("Matriz Cuadrada Mágica:")
print(magic_square)

# Crear matriz de codificación basada en la matriz mágica
encoding_matrix = create_encoding_matrix_from_magic(magic_square)
print("\nMatriz de Codificación:")
print(encoding_matrix)

# Codificar datos
encoded_data = encode_data(data, encoding_matrix)
print("\nDatos Codificados:")
print(encoded_data)

# Decodificar datos
decoded_data = decode_data(encoded_data, encoding_matrix)
print("\nDatos Decodificados:")
print(decoded_data)


# In[12]:


import numpy as np

# Cuadrado Mágico con Constante 15
magic_square = np.array([[9, 2, 7],
                         [4, 6, 8],
                         [5, 10, 3]])

# Multiplicar la matriz por sí misma
result_matrix = np.dot(magic_square, magic_square)

# Imprimir la matriz original y el resultado de la multiplicación
print("Cuadrado Mágico con Constante 15:")
print(magic_square)

print("\nResultado de la multiplicación de la matriz por sí misma:")
print(result_matrix)


# In[16]:


import numpy as np

# Cuadrado Mágico con Constante 15
magic_square = np.array([[234, 227, 232],
                         [229, 231, 233],
                         [230, 235, 228]])

# Multiplicar la matriz por sí misma
result_matrix_1 = np.dot(magic_square, magic_square)

# Multiplicar el resultado por sí mismo
result_matrix_2 = np.dot(result_matrix_1, result_matrix_1)

# Imprimir los resultados
print("Cuadrado Mágico letra A  alfabeto creado por RESET")
print(magic_square)

print("\nResultado de la primera multiplicación de la matriz por sí misma:")
print(result_matrix_1)

print("\nResultado de la segunda multiplicación de la matriz por sí misma:")
print(result_matrix_2)


# In[ ]:




