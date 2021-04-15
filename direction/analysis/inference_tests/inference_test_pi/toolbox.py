# Imports
import os
import numpy as np
import time
from constants import datapath, data_filename
import pickle
# -------

# Loading data and label files
def load_file(i_file, n_events, norm=1e-6):
    # Load 500 MHz filter
    filt = np.load("bandpass_filters/500MHz_filter.npy")

    t0 = time.time()
    print(f"loading file {i_file}", flush=True)
    data = np.load(os.path.join(datapath, f"{data_filename}{i_file:04d}.npy"), allow_pickle=True, mmap_mode="r")
    
    data = data[0:n_events, :, :]
    data = np.fft.irfft(np.fft.rfft(data, axis=-1) * filt, axis=-1)
    data = data[:, :, :, np.newaxis]

    # check for nans and remove them
    idx = ~(np.isnan(data))
    idx = np.all(idx, axis=1)
    idx = np.all(idx, axis=1)
    idx = np.all(idx, axis=1)
    data = data[idx, :, :, :]

    data /= norm

    return data

