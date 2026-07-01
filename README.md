# TinyML - Classificador de Digitos MNIST com CNN

Um pipeline completo de TinyML para classificacao de digitos manuscritos (0-9) usando uma Rede Neural Convolucional (CNN). O projeto cobre desde o treinamento do modelo ate a quantizacao INT8 para deploy em dispositivos embarcados com recursos limitados.

## Visao Geral

Este projeto demonstra o fluxo de trabalho tipico de TinyML:

1. **Treinamento** - Treinar uma CNN no dataset MNIST usando TensorFlow/Keras
2. **Quantizacao** - Converter o modelo para TensorFlow Lite com quantizacao INT8
3. **Inferencia** - Realizar predicoes tanto com o modelo completo quanto com o modelo quantizado

## Arquitetura do Modelo

A CNN utiliza a seguinte arquitetura:

```
Input (28x28x1)
    -> Conv2D(32, 3x3, ReLU)
    -> MaxPooling2D(2x2)
    -> Conv2D(64, 3x3, ReLU)
    -> MaxPooling2D(2x2)
    -> Conv2D(64, 3x3, ReLU)
    -> Flatten
    -> Dense(64, ReLU)
    -> Dense(10, Softmax)
```

**Parametros do treinamento:**
- Otimizador: Adam
- Loss: Sparse Categorical Crossentropy
- Epocas: 5
- Dataset: MNIST (60.000 treino / 10.000 teste)

## Estrutura do Projeto

```
tiny-ml-model/
├── train.py                  # Script de treinamento da CNN
├── quantizer.py              # Script de quantizacao INT8 para TFLite
├── inference_full_model.py   # Inferencia com modelo Keras completo (.h5)
├── inference_quant.py        # Inferencia com modelo quantizado (.tflite)
├── mnist_cnn_model.h5        # Modelo treinado (Keras)
├── mnist_cnn_model.tflite    # Modelo quantizado (TensorFlow Lite)
├── requirements.txt          # Dependencias do projeto
├── test_img/
│   └── tres.png              # Imagem de teste (digito 3)
└── README.md
```

## Pre-requisitos

- Python 3.8+
- pip (gerenciador de pacotes Python)

## Instalacao

1. Clone o repositorio:

```bash
git clone https://github.com/thallys-dnx/tiny-ml-model.git
cd tiny-ml-model
```

2. Crie e ative um ambiente virtual:

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# ou
venv\Scripts\activate     # Windows
```

3. Instale as dependencias:

```bash
pip install -r requirements.txt
```

## Uso

### 1. Treinar o Modelo

Treina a CNN no dataset MNIST e salva o modelo como `mnist_cnn_model.h5`:

```bash
python3 train.py
```

**Saida esperada:**
- Arquivo `mnist_cnn_model.h5` (modelo Keras treinado)
- Acuracia de validacao ~99% apos 5 epocas

### 2. Quantizar o Modelo

Converte o modelo Keras para TensorFlow Lite com quantizacao INT8:

```bash
python3 quantizer.py
```

**Saida esperada:**
- Arquivo `mnist_cnn_model.tflite` (modelo quantizado)
- Reducao significativa no tamanho do modelo (~4x menor)

### 3. Realizar Inferencia

#### Com o modelo completo (Keras .h5):

```bash
python3 inference_full_model.py
```

#### Com o modelo quantizado (TFLite INT8):

```bash
python3 inference_quant.py
```

**Ambos os scripts:**
- Carregam a imagem de teste em `test_img/tres.png`
- Pre-processam a imagem (escala de cinza, redimensionamento 28x28, normalizacao)
- Exibem o digito previsto e a confianca/probabilidades

### Usar sua propria imagem

Para testar com uma imagem diferente, edite a variavel `IMAGE_PATH` nos scripts de inferencia:

```python
IMAGE_PATH = 'caminho/para/sua/imagem.png'
```

> **Dica:** A imagem deve conter um digito manuscrito (0-9) em fundo escuro, preferencialmente em escala de cinza.

## Quantizacao INT8

A quantizacao converte os pesos do modelo de float32 para int8, o que:

| Beneficio | Descricao |
|-----------|-----------|
| Tamanho reduzido | Modelo ~4x menor em disco |
| Inferencia mais rapida | Operacoes inteiras sao mais rapidas em MCUs |
| Menor consumo de energia | Ideal para dispositivos embarcados |
| Compatibilidade | Pronto para deploy em microcontroladores (ex: ARM Cortex-M) |

O processo utiliza um **dataset representativo** (100 amostras do MNIST) para calibrar a quantizacao e minimizar a perda de acuracia.

## Dependencias

| Pacote | Finalidade |
|--------|-----------|
| `tensorflow` | Treinamento, quantizacao e inferencia |
| `Pillow` | Carregamento e manipulacao de imagens |

## Proximos Passos

- [ ] Exportar modelo para formato C array (deploy em MCU)
- [ ] Adicionar suporte a mais datasets
- [ ] Implementar data augmentation no treinamento
- [ ] Adicionar metricas de performance (latencia, uso de memoria)
- [ ] Criar script de benchmark comparando modelo full vs quantizado

## Licenca

Este projeto e de uso livre para fins educacionais e de aprendizado.

---

Desenvolvido como projeto de estudo em TinyML e Machine Learning embarcado.
