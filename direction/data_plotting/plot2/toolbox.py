# Imports
import os
import numpy as np
from constants import datapath, data_filename, label_filename
from radiotools import helper as hp
from NuRadioReco.utilities import units
import pickle
# -------

# Loading data and label files
def load_file(i_file, norm=1e-6):
#     t0 = time.time()
#     print(f"loading file {i_file}", flush=True)
    data = np.load(os.path.join(datapath, f"{data_filename}{i_file:04d}.npy"), allow_pickle=True)[:, :, :, np.newaxis]
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