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
from termcolor import colored
import time
from toolbox import load_file, find_68_interval
from radiotools import helper as hp

#from scipy import stats
from tf_notification_callback import SlackCallback
import wandb
from wandb.keras import WandbCallback
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
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

from generator import TrainDataset, ValDataset, n_events_per_file, n_files_train, n_files_val, batch_size
from constants import saved_model_dir, run_version, dataset_name, datapath, data_filename, label_filename, plots_dir, project_name
# -------

# Values
feedback_freq = 3 # Only train on 1/feedback_freq of data per epoch
webhook = os.getenv("SLACKWEBHOOK")
architectures_dir = "architectures"
learning_rate = 0.00005
epochs = 100
loss_function = "mean_absolute_error"
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

# Initialize wandb
run = wandb.init(project=project_name,
                 group=run_version,
                 config={  # and include hyperparameters and metadata
                     "learning_rate": learning_rate,
                     "epochs": epochs,
                     "batch_size": batch_size,
                     "loss_function": loss_function,
                     "architecture": "CNN",
                     "dataset": dataset_name
                 })
run.name = run_name
config = wandb.config


# Model params
conv2D_filter_amount = 100
if run_name == "run14.1":
    conv2D_filter_size = 3
elif run_name == "run14.2":
    conv2D_filter_size = 7
elif run_name == "run14.3":
    conv2D_filter_size = 10
elif run_name == "run14.4":
    conv2D_filter_size = 15
elif run_name == "run14.5":
    conv2D_filter_size = 20

# Send model params to wandb
wandb.log({f"conv2D_filter_amount": conv2D_filter_amount})
wandb.log({f"conv2D_filter_size": conv2D_filter_size})

# ----------- Create model -----------
model = Sequential()

# Initial convolutional layers
model.add(Conv2D(conv2D_filter_amount, (1, conv2D_filter_size), strides=(1, 2), padding='valid', activation='relu', input_shape=(5, 512, 1)))
model.add(Conv2D(conv2D_filter_amount, (1, conv2D_filter_size), strides=(1, 2), padding='valid', activation='relu'))
model.add(Conv2D(conv2D_filter_amount, (1, conv2D_filter_size), strides=(1, 2), padding='valid', activation='relu'))
model.add(Conv2D(conv2D_filter_amount, (1, conv2D_filter_size), strides=(1, 2), padding='valid', activation='relu'))
model.add(Conv2D(conv2D_filter_amount, (1, conv2D_filter_size), strides=(1, 2), padding='valid', activation='relu'))
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

# Batch normalization
model.add(BatchNormalization())

model.add(Flatten())

# Dense layers
model.add(Dense(5 * 512))
model.add(Activation('relu'))
model.add(Dense(5 * 512))
model.add(Activation('relu'))
model.add(Dense(5 * 512))
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

# Output layer
model.add(Dense(3))
model.compile(loss=config.loss_function,
              optimizer=Adam(lr=config.learning_rate))
model.summary()
# ------------------------------------

# Save the model (for opening in eg Netron)
#model.save(f'{saved_model_dir}/{architectures_dir}/model_architecture_{run_name}.h5')
plot_model(model, to_file=f'{saved_model_dir}/{architectures_dir}/model_architecture_{run_name}.png', show_shapes=True)
model_json = model.to_json()
with open(f'{saved_model_dir}/{architectures_dir}/model_architecture_{run_name}.json', "w") as json_file:
    json_file.write(model_json)

# Send amount of parameters to wandb
trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
non_trainable_count = np.sum([K.count_params(w) for w in model.non_trainable_weights])

wandb.log({f"trainable_params": trainable_count})
wandb.log({f"non_trainable_params": non_trainable_count})

# Configuring callbacks
es = EarlyStopping(monitor="val_loss", patience=5),
mc = ModelCheckpoint(filepath=os.path.join(saved_model_dir, f"model.{run_name}.h5"),
                                                    monitor='val_loss', verbose=1,
                                                    save_best_only=True, mode='auto',
                                                    save_weights_only=False)
sc = SlackCallback(webhookURL=webhook, channel="nn-log", modelName=run_name, loss_metrics=['loss', 'val_loss'], getSummary=True)
wb = WandbCallback(save_model=False)
checkpoint = [es, mc, sc, wb]      

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
history = model.fit(x=dataset_train, steps_per_epoch=steps_per_epoch, epochs=config.epochs,
          validation_data=dataset_val, callbacks=[checkpoint, csv_logger])

# Dump history with pickle
with open(os.path.join(saved_model_dir, f'history_{run_name}.pkl'), 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

# Sleep for a few seconds to free up some resources...
time.sleep(5)

# Plot loss and evaluate
os.system(f"python plot_loss.py {run_id}")
os.system(f"python plot_performance.py {run_id}")

# Calculate 68 % interval and sent to wandb
angle_68 = find_68_interval(run_name)

wandb.log({f"68 % interval": angle_68})

run.join()

print(colored(f"Done training {run_name}!", "green", attrs=["bold"]))
print("")



