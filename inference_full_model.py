import tensorflow as tf
import numpy as np
from PIL import Image
import os

MODEL_PATH = 'mnist_cnn_model.h5' # Caminho para o seu modelo Keras salvo

IMAGE_PATH = 'test_img/tres.png' # Exemplo: uma imagem de um '3'

# --- 1. Carregar o Modelo Keras (.h5) ---
if not os.path.exists(MODEL_PATH):
    print(f"Erro: O arquivo do modelo '{MODEL_PATH}' não foi encontrado.")
    print("Certifique-se de ter treinado e salvo o modelo, e que o arquivo está no diretório correto.")
    exit()

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Modelo Keras '{MODEL_PATH}' carregado com sucesso!")
except Exception as e:
    print(f"Erro ao carregar o modelo Keras: {e}")
    exit()

# Obter o formato de entrada esperado pelo modelo
# O input_shape é geralmente (None, 28, 28, 1) onde None é o tamanho do lote
input_shape = model.input_shape
print(f"Formato de entrada esperado pelo modelo: {input_shape}")

# --- 2. Preparar a Imagem de Entrada ---
def preprocess_image_for_keras(image_path, target_size=(28, 28)):
    if not os.path.exists(image_path):
        print(f"Erro: Imagem de teste '{image_path}' não encontrada.")
        print("Por favor, forneça um caminho válido para sua imagem de dígito.")
        return None

    # Carregar a imagem em escala de cinza
    img = Image.open(image_path).convert('L') # 'L' para escala de cinza

    # Redimensionar para o tamanho esperado pelo modelo
    img = img.resize(target_size)

    # Converter para um array NumPy e normalizar os pixels para 0-1
    img_array = np.array(img, dtype=np.float32) / 255.0

    # Adicionar dimensões de lote e canal (ex: de (28, 28) para (1, 28, 28, 1))
    # A dimensão do lote é 1 para uma única inferência.
    # A dimensão do canal é 1 para imagens em escala de cinza.
    img_array = np.expand_dims(img_array, axis=0)  # Adiciona dimensão do lote
    img_array = np.expand_dims(img_array, axis=-1) # Adiciona dimensão do canal

    return img_array

# Carregar e pré-processar a imagem
# O target_size deve corresponder à altura e largura do seu input_shape
# Por exemplo, se input_shape for (None, 28, 28, 1), target_size é (28, 28)
processed_image = preprocess_image_for_keras(IMAGE_PATH, target_size=(input_shape[1], input_shape[2]))

if processed_image is None:
    exit() # Sai se a imagem não puder ser processada

print(f"Imagem pré-processada com shape: {processed_image.shape}, dtype: {processed_image.dtype}")

# --- 3. Realizar a Previsão ---
print("Realizando inferência...")
predictions = model.predict(processed_image)

# --- 4. Interpretar os Resultados ---
# As previsões são geralmente logits ou probabilidades após softmax.
# Para MNIST, esperamos 10 valores (um para cada dígito de 0 a 9).
predicted_digit = np.argmax(predictions) # Pega o índice com a maior probabilidade
confidence = np.max(predictions) # Pega a maior probabilidade

print(f"\nResultados da Inferência:")
print(f"Probabilidades (ou logits) para cada dígito: {predictions[0]}")
print(f"Dígito Previsto: {predicted_digit}")
print(f"Confiança: {confidence:.4f}") # Exibe a confiança com 4 casas decimais