import tensorflow as tf

# 1. Carregar o modelo Keras
try:
    model = tf.keras.models.load_model('mnist_cnn_model.h5')
    print("Modelo Keras carregado com sucesso!")
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    print("Por favor, verifique se 'mnist_cnn_model.h5' existe no diretório correto.")
    exit()

# 2. Instanciar o TFLiteConverter corretamente
# Use tf.lite.TFLiteConverter em vez de tf.keras.TFLiteConverter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 3. Opcional: Aplicar quantização (essencial para TinyML)
# Para quantização, é recomendado usar um dataset de amostras.
try:
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255.0

    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_data_gen():
        # Use um subconjunto pequeno e representativo dos dados de treinamento.
        # Por exemplo, 100 amostras são geralmente suficientes.
        for input_value in tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(100):
            yield [input_value]

    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Se você quer entrada e saída quantizadas para INT8, defina-os.
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    print("Configuração de quantização INT8 aplicada.")

except NameError:
    print("Variável 'train_images' não encontrada para quantização. O modelo será convertido sem quantização completa ou você pode remover a configuração de quantização.")
except Exception as e:
    print(f"Erro durante a configuração da quantização: {e}")
    print("O modelo pode ser convertido sem quantização ou você pode ajustar as configurações.")


# 4. Converter o modelo
tflite_model = converter.convert()
print("Modelo convertido para TensorFlow Lite com sucesso!")

# 5. Salvar o modelo TFLite
with open('mnist_cnn_model.tflite', 'wb') as f:
    f.write(tflite_model)
print("Modelo TensorFlow Lite salvo como 'mnist_cnn_model.tflite'.")