# Imports
import os
import numpy as np
import time
from constants import dataset
from radiotools import helper as hp
from radiotools import stats
from NuRadioReco.utilities import units
import pickle
# -------

# Loading data and label files
def load_file(i_file, norm=1e-6):
    # Load 500 MHz filter
    filt = np.load("bandpass_filters/500MHz_filter.npy")

    #     t0 = time.time()
    #     print(f"loading file {i_file}", flush=True)
    data = np.load(os.path.join(dataset.datapath, f"{dataset.data_filename}{i_file:04d}.npy"), allow_pickle=True)
    data = np.fft.irfft(np.fft.rfft(data, axis=-1) * filt, axis=-1)
    data = data[:, :, :, np.newaxis]

    labels_tmp = np.load(os.path.join(dataset.datapath, f"{dataset.label_filename}{i_file:04d}.npy"), allow_pickle=True)
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

# Loading data and label files and also other properties
def load_file_all_properties(i_file, norm=1e-6):
    t0 = time.time()
    print(f"loading file {i_file}", flush=True)

    # Load 500 MHz filter
    filt = np.load("bandpass_filters/500MHz_filter.npy")

    data = np.load(os.path.join(dataset.datapath, f"{dataset.data_filename}{i_file:04d}.npy"), allow_pickle=True)
    data = np.fft.irfft(np.fft.rfft(data, axis=-1) * filt, axis=-1)
    data = data[:, :, :, np.newaxis]
    
    labels_tmp = np.load(os.path.join(dataset.datapath, f"{dataset.label_filename}{i_file:04d}.npy"), allow_pickle=True)
    print(f"finished loading file {i_file} in {time.time() - t0}s")
    nu_zenith_data = np.array(labels_tmp.item()["nu_zenith"])
    nu_azimuth_data = np.array(labels_tmp.item()["nu_azimuth"])
    nu_direction_data = hp.spherical_to_cartesian(nu_zenith_data, nu_azimuth_data)

    nu_energy_data = np.array(labels_tmp.item()["nu_energy"])
    nu_flavor_data = np.array(labels_tmp.item()["nu_flavor"])
    shower_energy_data = np.array(labels_tmp.item()["shower_energy"])

    # check for nans and remove them
    idx = ~(np.isnan(data))
    idx = np.all(idx, axis=1)
    idx = np.all(idx, axis=1)
    idx = np.all(idx, axis=1)
    data = data[idx, :, :, :]
    data /= norm

    nu_zenith_data = nu_zenith_data[idx]
    nu_azimuth_data = nu_azimuth_data[idx]
    nu_direction_data = nu_direction_data[idx]
    nu_energy_data = nu_energy_data[idx]
    nu_flavor_data = nu_flavor_data[idx]
    shower_energy_data = shower_energy_data[idx]

    return data, nu_direction_data, nu_zenith_data, nu_azimuth_data, nu_energy_data, nu_flavor_data, shower_energy_data


def calculate_percentage_interval(angle_difference_data, percentage=0.68):
    # Redefine N
    N = angle_difference_data.size
    weights = np.ones(N)

    angle = stats.quantile_1d(angle_difference_data, weights, percentage)

    # OLD METHOD -------------------------------
    # Calculate Rayleigh fit
    # loc, scale = stats.rayleigh.fit(angle)
    # xl = np.linspace(angle.min(), angle.max(), 100) # linspace for plotting

    # Calculate 68 %
    #index_at_68 = int(0.68 * N)
    #angle_68 = np.sort(angle_difference_data)[index_at_68]
    # ------------------------------------------

    return angle

def get_pred_angle_diff_data(run_name):
    prediction_file = f'saved_models/model.{run_name}.h5_predicted.pkl'
    with open(prediction_file, "br") as fin:
        nu_direction_predict, nu_direction = pickle.load(fin)

    # Only pick first 100000 data
    # N = 100000
    # nu_direction_predict = nu_direction_predict[:N]
    # nu_direction = nu_direction[:N]

    angle_difference_data = np.array([hp.get_angle(nu_direction_predict[i], nu_direction[i]) for i in range(len(nu_direction))]) / units.deg

    return angle_difference_data

def find_68_interval(run_name):
    angle_difference_data = get_pred_angle_diff_data(run_name)

    angle_68 = calculate_percentage_interval(angle_difference_data, 68)

    return angle_68

