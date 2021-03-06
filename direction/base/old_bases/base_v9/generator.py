# Imports
import os
import numpy as np
import tensorflow as tf
import time
from constants import datapath, data_filename, label_filename
# -------

np.set_printoptions(precision=4)

n_files = 82
# n_files = 10
n_files_test = 3
norm = 1e-6
# n_files_val = int(0.1 * n_files)
n_files_val = 10
n_files_train = n_files - n_files_val - n_files_test
list_of_file_ids_train = np.arange(n_files_train, dtype=np.int)
list_of_file_ids_val = np.arange(n_files_train, n_files_train + n_files_val, dtype=np.int)
list_of_file_ids_test = np.arange(n_files_train + n_files_val, n_files, dtype=np.int)
n_events_per_file = 100000
batch_size = 64

print(f"training on {n_files_train} files ({n_files_train/n_files*100:.1f}%), validating on {n_files_val} files ({n_files_val/n_files*100:.1f}%), testing on {n_files_test} files ({n_files_test/n_files*100:.1f}%)")

# Convert spherical to cartesian
def spherical_to_cartesian(zenith, azimuth):
    sinZenith = np.sin(zenith)
    x = sinZenith * np.cos(azimuth)
    y = sinZenith * np.sin(azimuth)
    z = np.cos(zenith)
    if hasattr(zenith, '__len__') and hasattr(azimuth, '__len__'):
        return np.array([x, y, z]).T
    else:
        return np.array([x, y, z])


def load_file(i_file, norm=norm):
    # Load data
    data = np.load(os.path.join(datapath, f"{data_filename}{i_file:04d}.npy"), allow_pickle=True)[:, :, :, np.newaxis]
    labels_tmp = np.load(os.path.join(datapath, f"{label_filename}{i_file:04d}.npy"), allow_pickle=True)

    # Convert to cartesian coordinates
    nu_zenith = np.array(labels_tmp.item()["nu_zenith"])
    nu_azimuth = np.array(labels_tmp.item()["nu_azimuth"])
    nu_direction = spherical_to_cartesian(nu_zenith, nu_azimuth)

    # Check for nans and remove them
    idx = ~(np.isnan(data))
    idx = np.all(idx, axis=1)
    idx = np.all(idx, axis=1)
    idx = np.all(idx, axis=1)
    data = data[idx, :, :, :]
    nu_direction = nu_direction[idx]
    data /= norm

    # Normalize direction
    nu_direction = np.array([v/np.linalg.norm(v) for v in nu_direction])

    return data, nu_direction


class TrainDataset(tf.data.Dataset):

    def _generator(file_id):
        if((file_id + 1) == n_files_train):
#             print("reshuffling")
            np.random.shuffle(list_of_file_ids_train)

        # Opening the file
        i_file = list_of_file_ids_train[file_id]
        data, nu_direction = load_file(i_file, norm)
        num_samples = data.shape[0]
        rand_ids = np.arange(num_samples, dtype=np.int)
        np.random.shuffle(rand_ids)
        n_batches = num_samples // batch_size
        for i_batch in range(n_batches):
            # Reading data (line, record) from the file
            y = nu_direction[rand_ids[i_batch * batch_size:(i_batch + 1) * batch_size]]
            x = data[rand_ids[i_batch * batch_size:(i_batch + 1) * batch_size], :, :, :]
            yield x, y

    def __new__(cls, file_id):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=(tf.dtypes.float64, tf.dtypes.float64),
            output_shapes=((64, 5, 512, 1), (64, 3)),
            args=(file_id,)
        )


class ValDataset(tf.data.Dataset):

    def _generator(file_id):
        if((file_id + 1) == n_files_val):
            np.random.shuffle(list_of_file_ids_val)

        # Opening the file
        i_file = list_of_file_ids_val[file_id]
        data, nu_direction = load_file(i_file, norm)
        num_samples = data.shape[0]
        rand_ids = np.arange(num_samples, dtype=np.int)
        np.random.shuffle(rand_ids)
        n_batches = num_samples // batch_size
        for i_batch in range(n_batches):
            # Reading data (line, record) from the file
            y = nu_direction[rand_ids[i_batch * batch_size:(i_batch + 1) * batch_size]]
            x = data[rand_ids[i_batch * batch_size:(i_batch + 1) * batch_size], :, :, :]
            yield x, y

    def __new__(cls, file_id):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=(tf.dtypes.float64, tf.dtypes.float64),
            output_shapes=((64, 5, 512, 1), (64, 3)),
            args=(file_id,)
        )

