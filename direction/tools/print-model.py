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
conv2D_filter_size = 5
pooling_size = 4
amount_Conv2D_layers_per_block = 3 
amount_Conv2D_blocks = 4
conv2D_filter_amount = 32

# ----------- Create model -----------
model = Sequential()

# Conv2D block 1
model.add(Conv2D(conv2D_filter_amount, (1, conv2D_filter_size), strides=(1, 1), padding='same', activation='relu', input_shape=(5, 512, 1)))

for _ in range(amount_Conv2D_layers_per_block-1):
    model.add(Conv2D(conv2D_filter_amount, (1, conv2D_filter_size), strides=(1, 1), padding='same', activation='relu'))

# MaxPooling to reduce size
model.add(AveragePooling2D(pool_size=(1, pooling_size)))

for i in range(amount_Conv2D_blocks-1):
    # Conv2D block
    for _ in range(amount_Conv2D_layers_per_block):
        model.add(Conv2D(conv2D_filter_amount*2**(i+1), (1, conv2D_filter_size), strides=(1, 1), padding='same', activation='relu'))

    # MaxPooling to reduce size
    model.add(AveragePooling2D(pool_size=(1, pooling_size)))

# Batch normalization
model.add(BatchNormalization())

# Flatten prior to dense layers
model.add(Flatten())

# Dense layers (fully connected)
model.add(Dense(1024, activation='relu'))
model.add(Dense(1024, activation='relu'))
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



