# Imports
import os
import numpy as np
from constants import datapath, data_filename, label_filename
from radiotools import helper as hp
from radiotools import stats
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


def calculate_68_interval(angle_difference_data):
    # Redefine N
    N = angle_difference_data.size
    weights = np.ones(N)

    angle_68 = stats.quantile_1d(angle_difference_data, weights, 0.68)

    # OLD METHOD -------------------------------
    # Calculate Rayleigh fit
    # loc, scale = stats.rayleigh.fit(angle)
    # xl = np.linspace(angle.min(), angle.max(), 100) # linspace for plotting

    # Calculate 68 %
    # index_at_68 = int(0.68 * N)
    # angle_68 = np.sort(angle_difference_data)[index_at_68]
    # ------------------------------------------

    return angle_68

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

    angle_68 = calculate_68_interval(angle_difference_data)

    return angle_68

