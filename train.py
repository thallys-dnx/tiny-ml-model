"""
train.py - Treinamento de CNN para classificacao de digitos MNIST

Este script treina uma Rede Neural Convolucional (CNN) no dataset MNIST
para classificar digitos manuscritos de 0 a 9. O modelo treinado e salvo
no formato Keras HDF5 (.h5) para uso posterior em inferencia ou quantizacao.

Arquitetura da CNN:
    Input (28x28x1) -> Conv2D(32) -> MaxPool -> Conv2D(64) -> MaxPool
    -> Conv2D(64) -> Flatten -> Dense(64) -> Dense(10, softmax)

Uso:
    python3 train.py

Saida:
    - mnist_cnn_model.h5: Modelo treinado salvo em disco

Dependencias:
    - tensorflow
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def load_and_preprocess_data():
    """
    Carrega o dataset MNIST e aplica pre-processamento.

    O MNIST contem 70.000 imagens de digitos manuscritos (28x28 pixels):
    - 60.000 para treinamento
    - 10.000 para teste

    Pre-processamento aplicado:
    - Reshape para formato (N, 28, 28, 1) - adiciona canal de cor
    - Conversao para float32
    - Normalizacao dos pixels de [0, 255] para [0, 1]

    Returns:
        tuple: (train_images, train_labels, test_images, test_labels)
    """
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    # Reshape: adiciona dimensao de canal (grayscale = 1 canal)
    # Normalizacao: divide por 255 para escalar pixels entre 0 e 1
    train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
    test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

    return train_images, train_labels, test_images, test_labels


def build_model():
    """
    Define a arquitetura da CNN para classificacao de digitos.

    A rede utiliza tres blocos convolucionais seguidos de camadas densas:
    - Bloco 1: Conv2D(32 filtros, kernel 3x3) + MaxPooling
    - Bloco 2: Conv2D(64 filtros, kernel 3x3) + MaxPooling
    - Bloco 3: Conv2D(64 filtros, kernel 3x3)
    - Classificador: Flatten + Dense(64) + Dense(10, softmax)

    Returns:
        tf.keras.Model: Modelo CNN compilado
    """
    model = models.Sequential([
        # Primeiro bloco convolucional
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),

        # Segundo bloco convolucional
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Terceiro bloco convolucional
        layers.Conv2D(64, (3, 3), activation='relu'),

        # Camadas de classificacao
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')  # 10 classes para digitos (0-9)
    ])

    return model


def compile_model(model):
    """
    Compila o modelo com otimizador, funcao de perda e metricas.

    Configuracao:
    - Otimizador: Adam (learning rate adaptativa)
    - Loss: SparseCategoricalCrossentropy (labels inteiros, nao one-hot)
    - Metrica: Acuracia

    Args:
        model (tf.keras.Model): Modelo a ser compilado
    """
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )


def train_model(model, train_images, train_labels, test_images, test_labels, epochs=5):
    """
    Treina o modelo no dataset de treinamento.

    Args:
        model (tf.keras.Model): Modelo compilado
        train_images (np.ndarray): Imagens de treinamento (60000, 28, 28, 1)
        train_labels (np.ndarray): Labels de treinamento (60000,)
        test_images (np.ndarray): Imagens de validacao (10000, 28, 28, 1)
        test_labels (np.ndarray): Labels de validacao (10000,)
        epochs (int): Numero de epocas de treinamento (padrao: 5)

    Returns:
        tf.keras.callbacks.History: Historico de treinamento com metricas
    """
    history = model.fit(
        train_images, train_labels,
        epochs=epochs,
        validation_data=(test_images, test_labels)
    )
    return history


def main():
    """Pipeline principal de treinamento."""
    print("=" * 50)
    print("TinyML - Treinamento CNN para MNIST")
    print("=" * 50)

    # 1. Carregar e pre-processar os dados
    print("\n[1/4] Carregando dataset MNIST...")
    train_images, train_labels, test_images, test_labels = load_and_preprocess_data()
    print(f"  - Treinamento: {train_images.shape[0]} imagens")
    print(f"  - Teste: {test_images.shape[0]} imagens")

    # 2. Construir o modelo
    print("\n[2/4] Construindo arquitetura da CNN...")
    model = build_model()
    model.summary()

    # 3. Compilar o modelo
    print("\n[3/4] Compilando modelo...")
    compile_model(model)

    # 4. Treinar o modelo
    print("\n[4/4] Treinando modelo (5 epocas)...")
    history = train_model(model, train_images, train_labels, test_images, test_labels)

    # 5. Salvar o modelo
    output_path = 'mnist_cnn_model.h5'
    model.save(output_path)
    print(f"\nModelo salvo em: {output_path}")
    print("Treinamento concluido com sucesso!")


if __name__ == '__main__':
    main()
