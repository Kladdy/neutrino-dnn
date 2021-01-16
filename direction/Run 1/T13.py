import os
from gpuutils import GpuUtils
# GpuUtils.allocate() #this tries to allocate a GPU having 1GB memory
# GpuUtils.allocate(required_memory = 10000)
GpuUtils.allocate(gpu_count=1,
                  framework='keras')
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
import numpy as np
# from matplotlib import pyplot as plt
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import normalize
import matplotlib
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv1D, Flatten, Dropout
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, AveragePooling1D, Input, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
import pickle
import os
import matplotlib.pyplot as plt
# import pandas as pd
import time
from generator3 import TrainDataset, ValDataset, n_events_per_file, n_files_train, n_files_val, batch_size

# For arguments when calling from command line (argument 0 is file name)
import sys
run_name = sys.argv[1] # run name (first argument)
learning_rate = int(sys.argv[2]) # learning rate (second argument)

filename = os.path.splitext(__file__)[0]
path = os.path.join('saved_models', filename)
if not os.path.exists(path):
    os.makedirs(path)

model = Sequential()
model.add(Conv2D(100, (1, 10), strides=(1, 2), padding='valid', activation='relu', input_shape=(5, 512, 1)))
model.add(Conv2D(100, (1, 10), strides=(1, 2), padding='valid', activation='relu'))
model.add(Conv2D(100, (1, 10), strides=(1, 2), padding='valid', activation='relu'))
model.add(Conv2D(100, (1, 10), strides=(1, 2), padding='valid', activation='relu'))
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
model.add(Flatten())
model.add(Dense(5 * 512))
model.add(Activation('relu'))
model.add(Dense(5 * 512))
# model.add(Activation('relu'))
# model.add(Dense(4 * 512))
model.add(Activation('relu'))
model.add(Dense(1024))
# model.add(Dropout(.2))
model.add(Activation('relu'))
model.add(Dense(256))
# model.add(Dropout(.1))
model.add(Activation('relu'))
model.add(Dense(128))
# model.add(Dropout(.1))
# model.add(Activation('relu'))
model.add(Dense(3))
model.compile(loss='mean_squared_error',
              optimizer=Adam(lr=learning_rate))
model.summary()
# a = 1 / 0

checkpoint = ModelCheckpoint(filepath=os.path.join('saved_models', filename, "model.{run_name}.h5"),
                                                    monitor='val_loss', verbose=1,
                                                    save_best_only=True, mode='auto',
                                                    save_weights_only=False)
csv_logger = CSVLogger(os.path.join('saved_models', filename, "model_history_log_{run_name}.csv"), append=True)

steps_per_epoch = n_files_train // 5 * (n_events_per_file // batch_size)
n_batches_per_file = n_events_per_file // batch_size
print(f"steps_per_epoch {steps_per_epoch}, n_batches_per_file {n_batches_per_file}")

dataset_train = tf.data.Dataset.range(n_files_train).prefetch(n_batches_per_file * 10).interleave(
        TrainDataset,
        cycle_length=2,
        num_parallel_calls=2,
        deterministic=False).repeat()

dataset_val = tf.data.Dataset.range(n_files_val).prefetch(n_batches_per_file * 10).interleave(
        ValDataset,
        cycle_length=2,
        num_parallel_calls=2,
        deterministic=False)

history = model.fit(x=dataset_train, steps_per_epoch=steps_per_epoch, epochs=50,
          validation_data=dataset_val, callbacks=[checkpoint, csv_logger])
with open(os.path.join('saved_models', filename, 'history_{run_name}.pkl'), 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

