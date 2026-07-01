"""
quantizer.py - Quantizacao INT8 do modelo CNN para TensorFlow Lite

Este script converte o modelo Keras (.h5) treinado para o formato TensorFlow Lite
com quantizacao INT8 completa. A quantizacao reduz o tamanho do modelo em ~4x e
permite execucao eficiente em microcontroladores e dispositivos embarcados.

Processo de quantizacao:
    1. Carrega o modelo Keras treinado (.h5)
    2. Configura o TFLiteConverter com quantizacao INT8
    3. Usa um dataset representativo (100 amostras) para calibracao
    4. Converte e salva o modelo quantizado (.tflite)

Requisitos:
    - O arquivo 'mnist_cnn_model.h5' deve existir (executar train.py primeiro)

Uso:
    python3 quantizer.py

Saida:
    - mnist_cnn_model.tflite: Modelo quantizado em INT8

Dependencias:
    - tensorflow
"""

import sys
import tensorflow as tf


# Configuracoes
INPUT_MODEL_PATH = 'mnist_cnn_model.h5'
OUTPUT_MODEL_PATH = 'mnist_cnn_model.tflite'
NUM_CALIBRATION_SAMPLES = 100


def load_keras_model(model_path):
    """
    Carrega o modelo Keras salvo em formato HDF5.

    Args:
        model_path (str): Caminho para o arquivo .h5

    Returns:
        tf.keras.Model: Modelo Keras carregado

    Raises:
        SystemExit: Se o arquivo nao for encontrado ou houver erro no carregamento
    """
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Modelo Keras '{model_path}' carregado com sucesso!")
        return model
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        print(f"Verifique se '{model_path}' existe no diretorio correto.")
        print("Execute 'python3 train.py' primeiro para gerar o modelo.")
        sys.exit(1)


def get_representative_dataset(num_samples=100):
    """
    Gera um dataset representativo para calibracao da quantizacao.

    O dataset representativo e usado pelo quantizador para determinar
    os ranges de valores (scale e zero_point) de cada camada, minimizando
    a perda de precisao na conversao float32 -> int8.

    Args:
        num_samples (int): Numero de amostras para calibracao (padrao: 100)

    Returns:
        function: Funcao geradora que produz amostras de calibracao
    """
    # Carrega o dataset MNIST para obter amostras reais
    (train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255.0

    def representative_data_gen():
        """Gera amostras individuais do dataset para calibracao."""
        dataset = tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(num_samples)
        for input_value in dataset:
            yield [input_value]

    return representative_data_gen


def configure_converter(model, representative_dataset_fn):
    """
    Configura o TFLiteConverter com quantizacao INT8 completa.

    Configuracoes aplicadas:
    - Otimizacao DEFAULT (quantizacao de pesos e ativacoes)
    - Dataset representativo para calibracao
    - Operacoes restritas a TFLITE_BUILTINS_INT8
    - Entrada e saida em formato INT8

    Args:
        model (tf.keras.Model): Modelo Keras a ser convertido
        representative_dataset_fn (function): Funcao geradora do dataset representativo

    Returns:
        tf.lite.TFLiteConverter: Converter configurado para quantizacao INT8
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Habilita otimizacao (quantizacao)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Define dataset representativo para calibracao dos ranges INT8
    converter.representative_dataset = representative_dataset_fn

    # Restringe operacoes a INT8 (necessario para deploy em MCU)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    # Define tipos de entrada e saida como INT8
    # Isso garante que todo o pipeline de inferencia use inteiros
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    return converter


def convert_and_save(converter, output_path):
    """
    Converte o modelo e salva no formato TFLite.

    Args:
        converter (tf.lite.TFLiteConverter): Converter configurado
        output_path (str): Caminho para salvar o arquivo .tflite

    Returns:
        int: Tamanho do modelo em bytes
    """
    print("Convertendo modelo para TensorFlow Lite...")
    tflite_model = converter.convert()
    print("Conversao concluida com sucesso!")

    # Salva o modelo em disco
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    model_size = len(tflite_model)
    print(f"Modelo salvo em: {output_path}")
    print(f"Tamanho do modelo quantizado: {model_size / 1024:.1f} KB")

    return model_size


def main():
    """Pipeline principal de quantizacao."""
    print("=" * 50)
    print("TinyML - Quantizacao INT8 para TensorFlow Lite")
    print("=" * 50)

    # 1. Carregar o modelo Keras
    print(f"\n[1/4] Carregando modelo '{INPUT_MODEL_PATH}'...")
    model = load_keras_model(INPUT_MODEL_PATH)

    # 2. Preparar dataset representativo
    print(f"\n[2/4] Preparando dataset representativo ({NUM_CALIBRATION_SAMPLES} amostras)...")
    representative_dataset_fn = get_representative_dataset(NUM_CALIBRATION_SAMPLES)
    print("  Dataset representativo configurado.")

    # 3. Configurar converter com quantizacao INT8
    print("\n[3/4] Configurando quantizacao INT8...")
    converter = configure_converter(model, representative_dataset_fn)
    print("  - Otimizacao: DEFAULT (quantizacao completa)")
    print("  - Operacoes: TFLITE_BUILTINS_INT8")
    print("  - Entrada/Saida: INT8")

    # 4. Converter e salvar
    print(f"\n[4/4] Convertendo e salvando em '{OUTPUT_MODEL_PATH}'...")
    convert_and_save(converter, OUTPUT_MODEL_PATH)

    print("\nQuantizacao concluida com sucesso!")
    print("O modelo esta pronto para deploy em dispositivos embarcados.")


if __name__ == '__main__':
    main()
