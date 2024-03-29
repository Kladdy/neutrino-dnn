# GPU allocation
from gpuutils import GpuUtils
GpuUtils.allocate(gpu_count=1, framework='keras')

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
# --------------

# Imports
import os
import numpy as np
import pickle
import argparse
from tf_notification_callback import SlackCallback
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv1D, Flatten, Dropout
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, AveragePooling1D, Input, Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence, plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from generator import TrainDataset, ValDataset, n_events_per_file, n_files_train, n_files_val, batch_size
from constants import saved_model_dir
# -------

# Values
feedback_freq = 5 # Only train on 1/feedback_freq of data per epoch
webhook = os.getenv("SLACKWEBHOOK")
architectures_dir = "architectures"
# ------

# Parse arguments
parser = argparse.ArgumentParser(description='Neural network for neutrino direction reconstruction')
parser.add_argument("run_id", type=str ,help="the id of the run, eg '3.2' for run3.2")

args = parser.parse_args()
run_id = args.run_id

# Save the run name
run_name = f"run{run_id}"

# Make sure saved_models folder exists
if not os.path.exists(saved_model_dir):
    os.makedirs(saved_model_dir)

# Make sure architectures folder exists
if not os.path.exists(f"{saved_model_dir}/{architectures_dir}"):
    os.makedirs(f"{saved_model_dir}/{architectures_dir}")

# ----------- Create model -----------
model = Sequential()

# Initial convolutional layers
model.add(Conv2D(100, (1, 10), strides=(1, 2), padding='valid', input_shape=(5, 512, 1)))
if run_name == "run6.2":
    model.add(BatchNormalization())
model.add(Activation('relu'))
if run_name == "run6.3":
    model.add(BatchNormalization())
if run_name == "run6.5":
    model.add(BatchNormalization())

model.add(Conv2D(100, (1, 10), strides=(1, 2), padding='valid'))
if run_name == "run6.2":
    model.add(BatchNormalization())
model.add(Activation('relu'))
if run_name == "run6.3":
    model.add(BatchNormalization())
if run_name == "run6.5":
    model.add(BatchNormalization())

model.add(Conv2D(100, (1, 10), strides=(1, 2), padding='valid'))
if run_name == "run6.2":
    model.add(BatchNormalization())
model.add(Activation('relu'))
if run_name == "run6.3":
    model.add(BatchNormalization())
if run_name == "run6.5":
    model.add(BatchNormalization())

model.add(Conv2D(100, (1, 10), strides=(1, 2), padding='valid'))
if run_name == "run6.2":
    model.add(BatchNormalization())
model.add(Activation('relu'))
if run_name == "run6.3":
    model.add(BatchNormalization())
if run_name == "run6.5":
    model.add(BatchNormalization())

model.add(Conv2D(100, (1, 10), strides=(1, 2), padding='valid'))
if run_name == "run6.2":
    model.add(BatchNormalization())
model.add(Activation('relu'))
if run_name == "run6.3":
    model.add(BatchNormalization())
if run_name == "run6.5":
    model.add(BatchNormalization())

# model.add(Conv1D(100, 10, strides=1, padding='valid', activation='relu'))
# model.add(Conv1D(100, 10, strides=1, padding='valid', activation='relu'))
# model.add(Conv1D(100, 10, strides=1, padding='valid', activation='relu'))
# model.add(Conv1D(100, 10, strides=1, padding='valid', activation='relu'))
# model.add(Conv1D(5, 3, strides=1, padding='valid', activation='relu', input_shape=(512, 5)))
# model.add(Conv1D(5, 3, strides=1, padding='valid', activation='relu', input_shape=(512, 5)))
# model.add(Conv1D(5, 20, strides=1, padding='valid', activation='relu', input_shape=(512,5)))
# model.add(Conv1D(5, 20, strides=1, padding='valid', activation='relu', input_shape=(512,5)))
# model.add(Conv1D(5, 20, strides=1, padding='valid', activation='relu', input_shape=(512,5)))
# model.add(Conv1D(5, 20, strides=1, padding='valid', activation='relu', input_shape=(512,5)))
# model.add(Conv1D(5, 20, strides=1, padding='valid', activation='relu', input_shape=(512,5)))
# model.add(Conv1D(5, 20, strides=1, padding='valid', activation='relu', input_shape=(512,5)))
if run_name == "run6.4":
    model.add(BatchNormalization())

model.add(Flatten())

# Dense layers
model.add(Dense(5 * 512))
if run_name == "run6.2":
    model.add(BatchNormalization())
model.add(Activation('relu'))
if run_name == "run6.3":
    model.add(BatchNormalization())
if run_name == "run6.6":
    model.add(BatchNormalization())

model.add(Dense(5 * 512))
if run_name == "run6.2":
    model.add(BatchNormalization())
model.add(Activation('relu'))
if run_name == "run6.3":
    model.add(BatchNormalization())
if run_name == "run6.6":
    model.add(BatchNormalization())
    
model.add(Dense(5 * 512))
if run_name == "run6.2":
    model.add(BatchNormalization())
model.add(Activation('relu'))
if run_name == "run6.3":
    model.add(BatchNormalization())
if run_name == "run6.6":
    model.add(BatchNormalization())
    
model.add(Dense(1024))
# model.add(Dropout(.2))
if run_name == "run6.2":
    model.add(BatchNormalization())
model.add(Activation('relu'))
if run_name == "run6.3":
    model.add(BatchNormalization())
if run_name == "run6.6":
    model.add(BatchNormalization())
    
model.add(Dense(256))
# model.add(Dropout(.1))
if run_name == "run6.2":
    model.add(BatchNormalization())
model.add(Activation('relu'))
if run_name == "run6.3":
    model.add(BatchNormalization())
if run_name == "run6.6":
    model.add(BatchNormalization())
    
model.add(Dense(128))
# model.add(Dropout(.1))
# model.add(Activation('relu'))

# Output layer
model.add(Dense(3))
model.compile(loss='mean_squared_error',
              optimizer=Adam(lr=0.00005))
model.summary()
# ------------------------------------

# Save the model (for opening in eg Netron)
#model.save(f'{saved_model_dir}/{architectures_dir}/model_architecture_{run_name}.h5')
plot_model(model, to_file=f'{saved_model_dir}/{architectures_dir}/model_architecture_{run_name}.png', show_shapes=True)
model_json = model.to_json()
with open(f'{saved_model_dir}/{architectures_dir}/model_architecture_{run_name}.json', "w") as json_file:
    json_file.write(model_json)

# Configuring checkpoints
es = EarlyStopping(monitor="val_loss", patience=5),
mc = ModelCheckpoint(filepath=os.path.join(saved_model_dir, f"model.{run_name}.h5"),
                                                    monitor='val_loss', verbose=1,
                                                    save_best_only=True, mode='auto',
                                                    save_weights_only=False)
sc = SlackCallback(webhookURL=webhook, channel="nn-log", modelName=run_name, loss_metrics=['loss', 'val_loss'], getSummary=True)
checkpoint = [es, mc, sc]      

# Configuring CSV-logger
csv_logger = CSVLogger(os.path.join(saved_model_dir, f"model_history_log_{run_name}.csv"), append=True)

# Calculating steps per epoch and batches per file
steps_per_epoch = n_files_train // feedback_freq * (n_events_per_file // batch_size)
n_batches_per_file = n_events_per_file // batch_size
print(f"steps_per_epoch {steps_per_epoch}, n_batches_per_file {n_batches_per_file}")

# Configuring training dataset
dataset_train = tf.data.Dataset.range(n_files_train).prefetch(n_batches_per_file * 10).interleave(
        TrainDataset,
        cycle_length=2,
        num_parallel_calls=2,
        deterministic=False).repeat()

# Configuring validation dataset
dataset_val = tf.data.Dataset.range(n_files_val).prefetch(n_batches_per_file * 10).interleave(
        ValDataset,
        cycle_length=2,
        num_parallel_calls=2,
        deterministic=False)

# Configuring history
history = model.fit(x=dataset_train, steps_per_epoch=steps_per_epoch, epochs=50,
          validation_data=dataset_val, callbacks=[checkpoint, csv_logger])

# Dump history with pickle
with open(os.path.join(saved_model_dir, f'history_{run_name}.pkl'), 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

