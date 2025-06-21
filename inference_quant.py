import tensorflow as tf
import numpy as np
from PIL import Image

# Caminho para o seu modelo TFLite
MODEL_PATH = 'mnist_cnn_model.tflite'
# Caminho para uma imagem de teste (substitua pela sua)
# Pode ser uma imagem que você desenhou ou uma do dataset MNIST.
# Certifique-se de que ela tenha um dígito e seja 28x28 pixels.
IMAGE_PATH = 'test_img/tres.png'

# --- 1. Carregar o modelo TensorFlow Lite ---
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Obter detalhes dos tensores de entrada e saída
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Verificar o formato esperado pelo modelo
# Geralmente será algo como: [[1, 28, 28, 1], dtype=float32] ou dtype=int8
print(f"Formato de entrada esperado: {input_details[0]['shape']}, Tipo: {input_details[0]['dtype']}")
print(f"Formato de saída esperado: {output_details[0]['shape']}, Tipo: {output_details[0]['dtype']}")

# --- 2. Preparar a imagem de entrada ---
def preprocess_image(image_path, input_shape, input_dtype):
    # Carregar a imagem em escala de cinza
    img = Image.open(image_path).convert('L') # 'L' para escala de cinza

    # Redimensionar para o tamanho esperado pelo modelo (ex: 28x28)
    # input_shape[1] e input_shape[2] seriam altura e largura (28, 28)
    img = img.resize((input_shape[1], input_shape[2]))

    # Converter para um array NumPy
    img_array = np.array(img, dtype=np.float32)

    # Normalizar os pixels (se o modelo foi treinado com dados normalizados)
    # Para modelos treinados com float32 e normalização 0-1
    if input_dtype == np.float32:
        img_array = img_array / 255.0
    # Para modelos quantizados para int8
    elif input_dtype == np.int8:
        # A quantização int8 mapeia valores de 0-255 para -128 a 127 ou 0 a 255
        # Depende do seu processo de quantização.
        # Se a entrada do modelo espera valores int8, você pode precisar ajustar o mapeamento
        # para corresponder ao scale e zero_point do tensor de entrada.
        # Geralmente, para imagens uint8, basta converter o tipo:
        img_array = img_array.astype(np.int8) # Ou np.uint8 e depois para int8

        # Se houver scale e zero_point para a entrada, use-os:
        # scale, zero_point = input_details[0]['quantization']
        # if scale != 0:
        #     img_array = img_array / scale + zero_point
        # img_array = np.round(img_array).astype(np.int8) # Certifique-se que esteja no tipo correto
    else:
        raise ValueError(f"Tipo de entrada não suportado: {input_dtype}")

    # Adicionar dimensões de lote e canal (ex: de (28, 28) para (1, 28, 28, 1))
    # input_shape[0] é o tamanho do lote (geralmente 1 para inferência única)
    # input_shape[3] é o número de canais (1 para escala de cinza)
    img_array = np.expand_dims(img_array, axis=0) # Adiciona dimensão do lote
    img_array = np.expand_dims(img_array, axis=-1) # Adiciona dimensão do canal

    # Certificar-se de que o array tem o dtype correto
    return img_array.astype(input_dtype)

# Carregar e pré-processar a imagem
input_data = preprocess_image(IMAGE_PATH, input_details[0]['shape'], input_details[0]['dtype'])

# --- 3. Definir o tensor de entrada ---
interpreter.set_tensor(input_details[0]['index'], input_data)

# --- 4. Invocar o interpretador ---
interpreter.invoke()

# --- 5. Obter os resultados ---
output_data = interpreter.get_tensor(output_details[0]['index'])

# Desquantizar a saída se o modelo for quantizado (output_dtype == np.int8)
if output_details[0]['dtype'] == np.int8:
    # A desquantização usa o scale e zero_point do tensor de saída
    scale, zero_point = output_details[0]['quantization']
    if scale != 0:
        output_data = (output_data.astype(np.float32) - zero_point) * scale

# O resultado é geralmente um array de probabilidades (softmax)
# Para um modelo de dígitos, a posição do maior valor indica o dígito
predicted_digit = np.argmax(output_data)
probabilities = tf.nn.softmax(output_data).numpy() # Converte para probabilidades
print(f"Probabilidades: {probabilities[0]}")
print(f"Dígito previsto: {predicted_digit}")
print("-" * 30)