import tensorflow as tf
from tensorflow.keras.models import load_model

model_path = "/mnt/md0/sstjaernholm/neutrino-dnn/final_models"
run_name = "F1.1"

print("Converting model...")

model = load_model(f"{model_path}/model.run{run_name}.h5")

# Convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model
with open(f'model.run{run_name}.tflite', 'wb') as f:
    f.write(tflite_model)

print("Done!")