# Imports
import os
import numpy as np
from constants import datapath, data_filename, label_filename
from radiotools import helper as hp
from NuRadioReco.utilities import units
import pickle
# -------

# Loading data and label files
def load_file(i_file, norm=1e-6, bandpass_filter="none"):
    if bandpass_filter == "none":
        data = np.load(os.path.join(datapath, f"{data_filename}{i_file:04d}.npy"), allow_pickle=True)[:, :, :, np.newaxis]
        
    elif bandpass_filter == "300MHz":
        # Load 300 MHz filter
        filt = np.load("bandpass_filters/300MHz_filter.npy")

        data = np.load(os.path.join(datapath, f"{data_filename}{i_file:04d}.npy"), allow_pickle=True)
        data = np.fft.irfft(np.fft.rfft(data, axis=-1) * filt, axis=-1)
        data = data[:, :, :, np.newaxis]

    elif bandpass_filter == "500MHz":
        # Load 500 MHz filter
        filt = np.load("bandpass_filters/500MHz_filter.npy")

        data = np.load(os.path.join(datapath, f"{data_filename}{i_file:04d}.npy"), allow_pickle=True)
        data = np.fft.irfft(np.fft.rfft(data, axis=-1) * filt, axis=-1)
        data = data[:, :, :, np.newaxis]

    labels_tmp = np.load(os.path.join(datapath, f"{label_filename}{i_file:04d}.npy"), allow_pickle=True)
#     print(f"finished loading file {i_file} in {time.time() - t0}s")
    nu_zenith = np.array(labels_tmp.item()["nu_zenith"])
    nu_azimuth = np.array(labels_tmp.item()["nu_azimuth"])
    nu_direction = hp.spherical_to_cartesian(nu_zenith, nu_azimuth)

    # check for nans and remove them
    idx = ~(np.isnan(data))
    idx = np.all(idx, axis=1)
    idx = np.all(idx, axis=1)
    idx = np.all(idx, axis=1)
    data = data[idx, :, :, :]
    nu_direction = nu_direction[idx]
    data /= norm

    return data, nu_direction