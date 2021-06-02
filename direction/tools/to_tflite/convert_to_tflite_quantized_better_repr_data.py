import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

def representative_dataset():
    for _ in range(100):
      data = 50*np.random.rand(1, 5, 512, 1)
      yield [data.astype(np.float32)]

model_path = "/mnt/md0/sstjaernholm/neutrino-dnn/final_models"
run_name = "F1.1"

print("Converting model...")

model = load_model(f"{model_path}/model.run{run_name}.h5")

# Convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Quantize according to this guide: https://www.tensorflow.org/lite/performance/post_training_quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # or tf.uint8
converter.inference_output_type = tf.int8  # or tf.uint8

# Do the conversion
tflite_quant_model = converter.convert()

# Save the model
with open(f'model.run{run_name}_quantized_REPR_DATA.tflite', 'wb') as f:
    f.write(tflite_quant_model)

print("Done!")