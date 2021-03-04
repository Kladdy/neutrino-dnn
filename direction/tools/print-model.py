# Print the model given in the file

# Imports
import os
import numpy as np
import pickle
import argparse
from termcolor import colored
import time

#from scipy import stats
import wandb
from wandb.keras import WandbCallback
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv1D, Flatten, Dropout
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, AveragePooling1D, Input, Flatten
from tensorflow.keras.layers import BatchNormalization, Lambda, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence, plot_model
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
# -------

print(colored(f"Printing model...", "yellow"))

# Values
feedback_freq = 3 # Only train on 1/feedback_freq of data per epoch
architectures_dir = "architectures"
learning_rate = 0.00005
epochs = 100
loss_function = "mean_absolute_error"
es_patience = 5
es_min_delta = 0.0001
# ------

# Model params
conv2D_filter_amount = 100
conv2D_filter_size = 15
stride_length = 2

# ----------- Create model -----------
model = Sequential()

# Conv2D block 1
model.add(Conv2D(64, (1, 5), strides=(1, 1), padding='same', activation='relu', input_shape=(5, 512, 1)))
model.add(Conv2D(64, (1, 5), strides=(1, 1), padding='same', activation='relu'))
model.add(Conv2D(64, (1, 5), strides=(1, 1), padding='same', activation='relu'))
model.add(Conv2D(64, (1, 5), strides=(1, 1), padding='same', activation='relu'))
model.add(Conv2D(64, (1, 5), strides=(1, 1), padding='same', activation='relu'))
model.add(Conv2D(64, (1, 5), strides=(1, 1), padding='same', activation='relu'))
model.add(Conv2D(64, (1, 5), strides=(1, 1), padding='same', activation='relu'))

# MaxPooling to reduce size
model.add(MaxPooling2D(pool_size=(1, 2)))

# Conv2D block 2
model.add(Conv2D(64, (1, 5), strides=(1, 1), padding='same', activation='relu'))
model.add(Conv2D(64, (1, 5), strides=(1, 1), padding='same', activation='relu'))
model.add(Conv2D(64, (1, 5), strides=(1, 1), padding='same', activation='relu'))
model.add(Conv2D(64, (1, 5), strides=(1, 1), padding='same', activation='relu'))

# MaxPooling to reduce size
model.add(MaxPooling2D(pool_size=(1, 2)))

# Conv2D block 3
model.add(Conv2D(64, (1, 5), strides=(1, 1), padding='same', activation='relu'))
model.add(Conv2D(64, (1, 5), strides=(1, 1), padding='same', activation='relu'))
model.add(Conv2D(64, (1, 5), strides=(1, 1), padding='same', activation='relu'))
model.add(Conv2D(64, (1, 5), strides=(1, 1), padding='same', activation='relu'))

# MaxPooling to reduce size
model.add(MaxPooling2D(pool_size=(1, 2)))

# Conv2D block 4
model.add(Conv2D(64, (1, 5), strides=(1, 1), padding='same', activation='relu'))
model.add(Conv2D(64, (1, 5), strides=(1, 1), padding='same', activation='relu'))
model.add(Conv2D(64, (1, 5), strides=(1, 1), padding='same', activation='relu'))
model.add(Conv2D(64, (1, 5), strides=(1, 1), padding='same', activation='relu'))

# MaxPooling to reduce size
model.add(MaxPooling2D(pool_size=(1, 2)))

# Conv2D block 5
model.add(Conv2D(64, (1, 5), strides=(1, 1), padding='same', activation='relu'))
model.add(Conv2D(64, (1, 5), strides=(1, 1), padding='same', activation='relu'))
model.add(Conv2D(64, (1, 5), strides=(1, 1), padding='same', activation='relu'))
model.add(Conv2D(64, (1, 5), strides=(1, 1), padding='same', activation='relu'))

# MaxPooling to reduce size
model.add(MaxPooling2D(pool_size=(1, 2)))

# Conv2D block 6
model.add(Conv2D(64, (1, 5), strides=(1, 1), padding='same', activation='relu'))
model.add(Conv2D(64, (1, 5), strides=(1, 1), padding='same', activation='relu'))
model.add(Conv2D(64, (1, 5), strides=(1, 1), padding='same', activation='relu'))
model.add(Conv2D(64, (1, 5), strides=(1, 1), padding='same', activation='relu'))

# MaxPooling to reduce size
model.add(MaxPooling2D(pool_size=(1, 2)))

# Batch normalization
model.add(BatchNormalization())

# Flatten prior to dense layers
model.add(Flatten())

# Dense layers (fully connected)
model.add(Dense(512, activation='relu'))
# model.add(Dropout(.1))
model.add(Dense(256, activation='relu'))
# model.add(Dropout(.1))
model.add(Dense(128, activation='relu'))
# model.add(Dropout(.1))

# Output layer
model.add(Dense(3))
model.add(Lambda(lambda x: K.l2_normalize(x,axis=1)))

model.compile(loss=loss_function,
              optimizer=Adam(lr=learning_rate))
model.summary()
# ------------------------------------

print(colored(f"Done printing model!", "green", attrs=["bold"]))
print("")



