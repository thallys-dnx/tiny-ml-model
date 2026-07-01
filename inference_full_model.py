"""
inference_full_model.py - Inferencia com modelo Keras completo (.h5)

Este script realiza a classificacao de digitos manuscritos usando o modelo
Keras completo (float32). Carrega uma imagem PNG, pre-processa e executa
a predicao retornando o digito identificado e o nivel de confianca.

Pipeline de inferencia:
    1. Carrega o modelo Keras (.h5)
    2. Carrega e pre-processa a imagem de entrada
    3. Executa a predicao (forward pass)
    4. Exibe o digito previsto e as probabilidades

Uso:
    python3 inference_full_model.py

    Para usar uma imagem diferente, altere a variavel IMAGE_PATH.

Requisitos:
    - mnist_cnn_model.h5 (executar train.py primeiro)
    - Imagem de teste em formato PNG (escala de cinza, digito em fundo escuro)

Dependencias:
    - tensorflow
    - Pillow
    - numpy
"""

import sys
import os
import numpy as np
import tensorflow as tf
from PIL import Image


# Configuracoes
MODEL_PATH = 'mnist_cnn_model.h5'
IMAGE_PATH = 'test_img/tres.png'
TARGET_SIZE = (28, 28)


def load_model(model_path):
    """
    Carrega o modelo Keras treinado.

    Args:
        model_path (str): Caminho para o arquivo .h5

    Returns:
        tf.keras.Model: Modelo carregado e pronto para inferencia

    Raises:
        SystemExit: Se o modelo nao for encontrado ou falhar ao carregar
    """
    if not os.path.exists(model_path):
        print(f"Erro: O arquivo do modelo '{model_path}' nao foi encontrado.")
        print("Execute 'python3 train.py' primeiro para gerar o modelo.")
        sys.exit(1)

    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Modelo Keras '{model_path}' carregado com sucesso!")
        return model
    except Exception as e:
        print(f"Erro ao carregar o modelo Keras: {e}")
        sys.exit(1)


def preprocess_image(image_path, target_size=(28, 28)):
    """
    Carrega e pre-processa uma imagem para inferencia.

    O pre-processamento segue os mesmos passos usados no treinamento:
    - Conversao para escala de cinza
    - Redimensionamento para 28x28 pixels
    - Normalizacao dos pixels para intervalo [0, 1]
    - Reshape para formato (1, 28, 28, 1) - batch + canal

    Args:
        image_path (str): Caminho para a imagem PNG
        target_size (tuple): Dimensoes alvo (altura, largura). Padrao: (28, 28)

    Returns:
        np.ndarray: Imagem pre-processada com shape (1, 28, 28, 1) e dtype float32

    Raises:
        SystemExit: Se a imagem nao for encontrada
    """
    if not os.path.exists(image_path):
        print(f"Erro: Imagem de teste '{image_path}' nao encontrada.")
        print("Forneca um caminho valido para uma imagem de digito.")
        sys.exit(1)

    # Carrega em escala de cinza ('L' = luminance, 8-bit pixels)
    img = Image.open(image_path).convert('L')

    # Redimensiona para o tamanho esperado pelo modelo
    img = img.resize(target_size)

    # Converte para array numpy e normaliza [0, 255] -> [0, 1]
    img_array = np.array(img, dtype=np.float32) / 255.0

    # Adiciona dimensao de batch (axis=0) e canal (axis=-1)
    # Shape final: (1, 28, 28, 1)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)

    return img_array


def predict(model, image):
    """
    Realiza a predicao no modelo Keras.

    Args:
        model (tf.keras.Model): Modelo carregado
        image (np.ndarray): Imagem pre-processada (1, 28, 28, 1)

    Returns:
        tuple: (digito_previsto, confianca, probabilidades)
            - digito_previsto (int): Digito com maior probabilidade (0-9)
            - confianca (float): Probabilidade do digito previsto
            - probabilidades (np.ndarray): Array com 10 probabilidades
    """
    predictions = model.predict(image)

    predicted_digit = np.argmax(predictions)
    confidence = np.max(predictions)
    probabilities = predictions[0]

    return predicted_digit, confidence, probabilities


def display_results(predicted_digit, confidence, probabilities):
    """
    Exibe os resultados da inferencia de forma formatada.

    Args:
        predicted_digit (int): Digito previsto (0-9)
        confidence (float): Nivel de confianca (0-1)
        probabilities (np.ndarray): Probabilidades para cada classe
    """
    print("\n" + "=" * 40)
    print("RESULTADOS DA INFERENCIA")
    print("=" * 40)
    print(f"\n  Digito Previsto: {predicted_digit}")
    print(f"  Confianca: {confidence:.4f} ({confidence * 100:.2f}%)")
    print(f"\n  Probabilidades por classe:")
    for digit, prob in enumerate(probabilities):
        bar = "#" * int(prob * 30)
        marker = " <--" if digit == predicted_digit else ""
        print(f"    [{digit}] {prob:.4f} {bar}{marker}")
    print("=" * 40)


def main():
    """Pipeline principal de inferencia com modelo completo."""
    print("=" * 50)
    print("TinyML - Inferencia com Modelo Keras (.h5)")
    print("=" * 50)

    # 1. Carregar modelo
    print(f"\n[1/3] Carregando modelo '{MODEL_PATH}'...")
    model = load_model(MODEL_PATH)

    input_shape = model.input_shape
    print(f"  Formato de entrada esperado: {input_shape}")

    # 2. Pre-processar imagem
    print(f"\n[2/3] Pre-processando imagem '{IMAGE_PATH}'...")
    image = preprocess_image(IMAGE_PATH, target_size=(input_shape[1], input_shape[2]))
    print(f"  Shape: {image.shape}, dtype: {image.dtype}")

    # 3. Realizar predicao
    print("\n[3/3] Realizando inferencia...")
    predicted_digit, confidence, probabilities = predict(model, image)

    # Exibir resultados
    display_results(predicted_digit, confidence, probabilities)


if __name__ == '__main__':
    main()
