"""
inference_quant.py - Inferencia com modelo quantizado TensorFlow Lite (INT8)

Este script realiza a classificacao de digitos manuscritos usando o modelo
quantizado em INT8 (.tflite). Simula o comportamento de inferencia que seria
executado em um microcontrolador, incluindo o tratamento de quantizacao/
desquantizacao dos dados de entrada e saida.

Pipeline de inferencia:
    1. Carrega o modelo TFLite via Interpreter
    2. Carrega e pre-processa a imagem (com quantizacao INT8)
    3. Executa a inferencia no interpreter
    4. Desquantiza a saida e exibe o resultado

Diferenca para o modelo completo:
    - Usa TFLite Interpreter em vez de model.predict()
    - Entrada e saida sao quantizadas (INT8 em vez de float32)
    - Necessita desquantizacao manual da saida para interpretar resultados
    - Mais eficiente em memoria e velocidade (ideal para MCUs)

Uso:
    python3 inference_quant.py

    Para usar uma imagem diferente, altere a variavel IMAGE_PATH.

Requisitos:
    - mnist_cnn_model.tflite (executar quantizer.py primeiro)
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
MODEL_PATH = 'mnist_cnn_model.tflite'
IMAGE_PATH = 'test_img/tres.png'


def load_tflite_model(model_path):
    """
    Carrega o modelo TFLite e aloca os tensores.

    O TFLite Interpreter e a engine de inferencia para modelos .tflite.
    Diferente do Keras, ele gerencia manualmente os tensores de entrada/saida.

    Args:
        model_path (str): Caminho para o arquivo .tflite

    Returns:
        tuple: (interpreter, input_details, output_details)
            - interpreter: Instancia do TFLite Interpreter
            - input_details: Metadados do tensor de entrada (shape, dtype, quantizacao)
            - output_details: Metadados do tensor de saida (shape, dtype, quantizacao)

    Raises:
        SystemExit: Se o modelo nao for encontrado
    """
    if not os.path.exists(model_path):
        print(f"Erro: Modelo '{model_path}' nao encontrado.")
        print("Execute 'python3 quantizer.py' primeiro para gerar o modelo quantizado.")
        sys.exit(1)

    # Inicializa o interpreter e aloca memoria para os tensores
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Obtem metadados dos tensores de entrada e saida
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return interpreter, input_details, output_details


def print_model_info(input_details, output_details):
    """
    Exibe informacoes sobre o formato do modelo quantizado.

    Args:
        input_details (list): Detalhes do tensor de entrada
        output_details (list): Detalhes do tensor de saida
    """
    print(f"  Entrada - Shape: {input_details[0]['shape']}, "
          f"Tipo: {input_details[0]['dtype'].__name__}")
    print(f"  Saida   - Shape: {output_details[0]['shape']}, "
          f"Tipo: {output_details[0]['dtype'].__name__}")

    # Exibe parametros de quantizacao se disponveis
    if 'quantization' in input_details[0]:
        scale, zero_point = input_details[0]['quantization']
        if scale != 0:
            print(f"  Quantizacao entrada - Scale: {scale}, Zero Point: {zero_point}")

    if 'quantization' in output_details[0]:
        scale, zero_point = output_details[0]['quantization']
        if scale != 0:
            print(f"  Quantizacao saida   - Scale: {scale}, Zero Point: {zero_point}")


def preprocess_image(image_path, input_shape, input_dtype):
    """
    Carrega e pre-processa uma imagem para inferencia quantizada.

    O pre-processamento depende do tipo de entrada do modelo:
    - float32: Normaliza pixels para [0, 1] (igual ao treinamento)
    - int8: Converte pixels para intervalo [-128, 127]

    Args:
        image_path (str): Caminho para a imagem PNG
        input_shape (np.ndarray): Shape esperado pelo modelo (ex: [1, 28, 28, 1])
        input_dtype (type): Tipo de dado esperado (np.float32 ou np.int8)

    Returns:
        np.ndarray: Imagem pre-processada no formato e tipo corretos

    Raises:
        SystemExit: Se a imagem nao for encontrada
        ValueError: Se o tipo de entrada nao for suportado
    """
    if not os.path.exists(image_path):
        print(f"Erro: Imagem '{image_path}' nao encontrada.")
        sys.exit(1)

    # Carrega em escala de cinza
    img = Image.open(image_path).convert('L')

    # Redimensiona para o tamanho esperado (altura x largura do input_shape)
    img = img.resize((input_shape[1], input_shape[2]))

    # Converte para array numpy
    img_array = np.array(img, dtype=np.float32)

    # Aplica normalizacao/quantizacao conforme o tipo esperado
    if input_dtype == np.float32:
        # Modelo float: normaliza para [0, 1]
        img_array = img_array / 255.0
    elif input_dtype == np.int8:
        # Modelo INT8: converte de [0, 255] para [-128, 127]
        # O mapeamento INT8 desloca o range para valores com sinal
        img_array = img_array.astype(np.int8)
    else:
        raise ValueError(f"Tipo de entrada nao suportado: {input_dtype}")

    # Adiciona dimensoes de batch e canal: (28, 28) -> (1, 28, 28, 1)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)

    # Garante o dtype correto para o interpreter
    return img_array.astype(input_dtype)


def run_inference(interpreter, input_details, output_details, input_data):
    """
    Executa a inferencia no TFLite Interpreter.

    Passos:
    1. Define o tensor de entrada com os dados pre-processados
    2. Invoca o interpreter (executa o forward pass)
    3. Recupera o tensor de saida

    Args:
        interpreter: Instancia do TFLite Interpreter
        input_details (list): Metadados do tensor de entrada
        output_details (list): Metadados do tensor de saida
        input_data (np.ndarray): Dados de entrada pre-processados

    Returns:
        np.ndarray: Dados brutos de saida (podem estar quantizados)
    """
    # Define os dados no tensor de entrada
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Executa a inferencia
    interpreter.invoke()

    # Recupera o resultado do tensor de saida
    output_data = interpreter.get_tensor(output_details[0]['index'])

    return output_data


def dequantize_output(output_data, output_details):
    """
    Desquantiza a saida INT8 para valores float.

    A desquantizacao usa a formula:
        valor_real = (valor_quantizado - zero_point) * scale

    Isso converte os valores INT8 de volta para o espaco float original,
    permitindo interpretar os resultados como probabilidades.

    Args:
        output_data (np.ndarray): Dados de saida quantizados (INT8)
        output_details (list): Metadados com parametros de quantizacao

    Returns:
        np.ndarray: Dados desquantizados em float32
    """
    if output_details[0]['dtype'] == np.int8:
        scale, zero_point = output_details[0]['quantization']
        if scale != 0:
            # Formula de desquantizacao: real = (quant - zero_point) * scale
            output_data = (output_data.astype(np.float32) - zero_point) * scale

    return output_data


def display_results(output_data, predicted_digit):
    """
    Exibe os resultados da inferencia de forma formatada.

    Args:
        output_data (np.ndarray): Saida desquantizada do modelo
        predicted_digit (int): Digito com maior probabilidade
    """
    # Aplica softmax para converter logits em probabilidades
    probabilities = tf.nn.softmax(output_data).numpy()

    print("\n" + "=" * 40)
    print("RESULTADOS DA INFERENCIA (Modelo Quantizado)")
    print("=" * 40)
    print(f"\n  Digito Previsto: {predicted_digit}")

    confidence = probabilities[0][predicted_digit]
    print(f"  Confianca: {confidence:.4f} ({confidence * 100:.2f}%)")

    print(f"\n  Probabilidades por classe:")
    for digit, prob in enumerate(probabilities[0]):
        bar = "#" * int(prob * 30)
        marker = " <--" if digit == predicted_digit else ""
        print(f"    [{digit}] {prob:.4f} {bar}{marker}")
    print("=" * 40)


def main():
    """Pipeline principal de inferencia com modelo quantizado."""
    print("=" * 50)
    print("TinyML - Inferencia com Modelo Quantizado (.tflite)")
    print("=" * 50)

    # 1. Carregar modelo TFLite
    print(f"\n[1/4] Carregando modelo '{MODEL_PATH}'...")
    interpreter, input_details, output_details = load_tflite_model(MODEL_PATH)
    print_model_info(input_details, output_details)

    # 2. Pre-processar imagem
    print(f"\n[2/4] Pre-processando imagem '{IMAGE_PATH}'...")
    input_data = preprocess_image(
        IMAGE_PATH,
        input_details[0]['shape'],
        input_details[0]['dtype']
    )
    print(f"  Shape: {input_data.shape}, dtype: {input_data.dtype}")

    # 3. Executar inferencia
    print("\n[3/4] Executando inferencia no TFLite Interpreter...")
    output_data = run_inference(interpreter, input_details, output_details, input_data)

    # 4. Desquantizar e interpretar resultado
    print("\n[4/4] Desquantizando saida e interpretando resultado...")
    output_data = dequantize_output(output_data, output_details)
    predicted_digit = np.argmax(output_data)

    # Exibir resultados
    display_results(output_data, predicted_digit)


if __name__ == '__main__':
    main()
