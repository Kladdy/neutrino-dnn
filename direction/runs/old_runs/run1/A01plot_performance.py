import numpy as np
from radiotools import plthelpers as php
from matplotlib import pyplot as plt
from tensorflow import keras
import os
import time
from radiotools import helper as hp
from NuRadioReco.utilities import units
import pickle

# System argument: supply like "python A01plot_performance.py 1.5"
import sys
run = sys.argv[1]

datapath = "/mnt/ssd2/data/energy_reconstruction/ARIANNA-200_Alvarez2000_3sigma_noise/"


def load_file(i_file, norm=1e-6):
#     t0 = time.time()
#     print(f"loading file {i_file}", flush=True)
    data = np.load(os.path.join(datapath, f"data_01_LPDA_2of4_3sigma_{i_file:04d}.npy"), allow_pickle=True)[:, :, :, np.newaxis]
    labels_tmp = np.load(os.path.join(datapath, f"labels_01_LPDA_2of4_3sigma_{i_file:04d}.npy"), allow_pickle=True)
#     print(f"finished loading file {i_file} in {time.time() - t0}s")
    nu_zenith = np.array(labels_tmp.item()["nu_zenith"])
    nu_azimuth = np.array(labels_tmp.item()["nu_azimuth"])
    nu_direction = hp.spherical_to_cartesian(nu_zenith, nu_azimuth)
#     shower_energy_had.reshape(shower_energy_had.shape[0], 1)

    # check for nans and remove them
    idx = ~(np.isnan(data))
    idx = np.all(idx, axis=1)
    idx = np.all(idx, axis=1)
    idx = np.all(idx, axis=1)
    data = data[idx, :, :, :]
    nu_direction = nu_direction[idx]
    data /= norm
#     print(f"finished processing file {i_file} in {time.time() - t0}s")

    return data, nu_direction


model = keras.models.load_model(f'saved_models/T13/model.run{run}.h5')

data, nu_direction = load_file(393)
nu_direction_predict = model.predict(data)

with open(f'saved_models/T13/model.run{run}.h5_predicted.pkl', "bw") as fout:
    pickle.dump([nu_direction_predict, nu_direction], fout, protocol=4)

angle = np.array([hp.get_angle(nu_direction_predict[i], nu_direction[i]) for i in range(len(nu_direction))])
# fig, ax = php.get_histogram(predicted_nu_energy[:, 0], bins=np.arange(17, 20.1, 0.05), xlabel="predicted energy")
fig, ax = php.get_histogram(angle / units.deg, bins=np.linspace(0, 40, 90),
                            xlabel=r"angular difference nu direction")
plt.show()
