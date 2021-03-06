# Imports
import numpy as np
from radiotools import plthelpers as php
from matplotlib import pyplot as plt
from tensorflow import keras
import os
import time
import pickle
from scipy import stats
from radiotools import helper as hp
from NuRadioReco.utilities import units
import argparse
from termcolor import colored
from constants import datapath, data_filename, label_filename, plots_dir, test_file_id
# -------

# Parse arguments
parser = argparse.ArgumentParser(description='Plot angular resolution')
parser.add_argument("run_id", type=str ,help="the id of the run, eg '3.2' for run3.2")

args = parser.parse_args()
run_id = args.run_id

# Save the run name and filename
run_name = f"run{run_id}"
filename = f"model_history_log_{run_name}.csv"

print(colored(f"Plotting angular resolution for {run_name}...", "yellow"))

# Make sure plots folder exists
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Make sure predicted file exists, otherwise run evaluator
prediction_file = f'saved_models/model.{run_name}.h5_predicted.pkl'
if not os.path.isfile(prediction_file):
    print("Prediction file does not exist, running evaluator...")
    os.system(f"python evaluator.py {run_id}")

def load_file(i_file, norm=1e-6):
#     t0 = time.time()
#     print(f"loading file {i_file}", flush=True)
    data = np.load(os.path.join(datapath, f"{data_filename}{i_file:04d}.npy"), allow_pickle=True)[:, :, :, np.newaxis]
    labels_tmp = np.load(os.path.join(datapath, f"{label_filename}{i_file:04d}.npy"), allow_pickle=True)
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


data, nu_direction = load_file(test_file_id)
max_dipole = np.max(np.abs(data[:, :, 4]), axis=1)
max_LPDA = np.max(np.max(np.abs(data[:, :, 0:4]), axis=1), axis=1)
max_any = np.max(np.max(np.abs(data[:, :, 0:5]), axis=1), axis=1)
labels_tmp = np.load(os.path.join(datapath, f"{label_filename}{test_file_id:04d}.npy"), allow_pickle=True)
shower_energy_had = np.array(labels_tmp.item()["shower_energy_had"])
shower_energy_had = np.log10(shower_energy_had + 1)

with open(prediction_file, "br") as fin:
    nu_direction_predict, nu_direction = pickle.load(fin)

N = 100000
nu_direction_predict = nu_direction_predict[:N]
nu_direction = nu_direction[:N]

angle = np.array([hp.get_angle(nu_direction_predict[i], nu_direction[i]) for i in range(len(nu_direction))]) / units.deg

# Redefine N
N = angle.size

# Calculate Rayleigh fit
# loc, scale = stats.rayleigh.fit(angle)
# xl = np.linspace(angle.min(), angle.max(), 100) # linspace for plotting

# Calculate 68 %
index_at_68 = int(0.68 * N)

angle_68 = np.sort(angle)[index_at_68]

# fig, ax = php.get_histogram(predicted_nu_energy[:, 0], bins=np.arange(17, 20.1, 0.05), xlabel="predicted energy")
fig, ax = php.get_histogram(angle, bins=np.linspace(0, 40, 90),
                            xlabel=r"angular difference nu direction")
# ax.plot(xl, N*stats.rayleigh(scale=scale, loc=loc).pdf(xl))
plt.title(f"Angular resolution for {run_name} with\n68 % at angle of {angle_68:.2f}")
fig.savefig(f"plots/angular_resolution_{run_name}.png")

plt.show()

# dE = predicted_nu_energy - nu_energy_test
SNR_bins = np.append(np.arange(1, 20, 1), [10000])
SNR_means = np.arange(1.5, 20.5, 1)
# mean = stats.binned_statistic(max_dipole / 10., dE, bins=SNR_bins)[0]
# std = stats.binned_statistic(max_dipole / 10., dE, bins=SNR_bins, statistic='std')[0]
# fig, ax = plt.subplots(1, 1)
# # ax.errorbar(SNR_means, mean, yerr=std, fmt="o")
# ax.plot(SNR_means, std, "o")
# ax.set_ylim(0, 0.4)
# ax.set_xlabel("SNR dipole")
# ax.set_ylabel("resolution log10(E_true - E_pred)")
# fig.tight_layout()
# fig.savefig("plots/STD_maxSNRdipole.png")
#
mean = stats.binned_statistic(max_LPDA[:, 0] / 10., angle, bins=SNR_bins)[0]
std = stats.binned_statistic(max_LPDA[:, 0] / 10., angle, bins=SNR_bins, statistic='std')[0]
fig, ax = plt.subplots(1, 1)
# ax.errorbar(SNR_means, mean, yerr=std, fmt="o")
ax.plot(SNR_means, mean, "o")
# ax.set_ylim(0, 0.4)
ax.set_xlabel("max SNR LPDA")
ax.set_ylabel("angular resolution")
fig.tight_layout()
fig.savefig(f"plots/mean_maxSNRLPDA_{run_name}.png")
#plt.show()

print(colored(f"Saved angular resolution for {run_name}!", "green", attrs=["bold"]))
print("")


#
# mean = stats.binned_statistic(max_any / 10., dE, bins=SNR_bins)[0]
# std = stats.binned_statistic(max_any / 10., dE, bins=SNR_bins, statistic='std')[0]
# fig, ax = plt.subplots(1, 1)
# # ax.errorbar(SNR_means, mean, yerr=std, fmt="o")
# ax.plot(SNR_means, std, "o")
# ax.set_ylim(0, 0.4)
# ax.set_xlabel("max SNR any channel")
# ax.set_ylabel("resolution log10(E_true - E_pred)")
# fig.tight_layout()
# fig.savefig("plots/STD_maxSNR.png")
#
# bins = np.arange(0, 100, 2)
# bin_means = 0.5 * (bins[1:] + bins[:-1])
# mean = stats.binned_statistic(np.rad2deg(np.array(labels_tmp.item()["nu_zenith"])), dE, bins=bins)[0]
# std = stats.binned_statistic(np.rad2deg(np.array(labels_tmp.item()["nu_zenith"])), dE, bins=bins, statistic='std')[0]
# fig, ax = plt.subplots(1, 1)
# # ax.errorbar(SNR_means, mean, yerr=std, fmt="o")
# ax.plot(bin_means, std, "o")
# ax.set_ylim(0, 0.4)
# ax.set_xlabel("zenith angle [deg]")
# ax.set_ylabel("resolution log10(E_true - E_pred)")
# fig.tight_layout()
# fig.savefig("plots/STD_zenith.png")
#
# bins = np.arange(0, 360, 2)
# bin_means = 0.5 * (bins[1:] + bins[:-1])
# mean = stats.binned_statistic(np.rad2deg(np.array(labels_tmp.item()["nu_azimuth"])), dE, bins=bins)[0]
# std = stats.binned_statistic(np.rad2deg(np.array(labels_tmp.item()["nu_azimuth"])), dE, bins=bins, statistic='std')[0]
# fig, ax = plt.subplots(1, 1)
# # ax.errorbar(SNR_means, mean, yerr=std, fmt="o")
# ax.plot(bin_means, std, "o")
# ax.set_ylim(0, 0.4)
# ax.set_xlabel("azimuth angle [deg]")
# ax.set_ylabel("resolution log10(E_true - E_pred)")
# fig.tight_layout()
# fig.savefig("plots/STD_azimuth.png")
#
# bins = np.linspace(0, 3000, 50)
# bin_means = 0.5 * (bins[1:] + bins[:-1])
# mean = stats.binned_statistic(np.array(labels_tmp.item()["distance"]), dE, bins=bins)[0]
# std = stats.binned_statistic(np.array(labels_tmp.item()["distance"]), dE, bins=bins, statistic='std')[0]
# fig, ax = plt.subplots(1, 1)
# # ax.errorbar(SNR_means, mean, yerr=std, fmt="o")
# ax.plot(bin_means, std, "o")
# ax.set_ylim(0, 0.4)
# ax.set_xlabel("distance [m]")
# ax.set_ylabel("resolution log10(E_true - E_pred)")
# fig.tight_layout()
# fig.savefig("plots/STD_distance.png")
#
# plt.show()
# z, xedges, yedges = np.histogram2d(max_dipole / 10., dE, bins=[SNR_bins, np.linspace(-1, 1, 40)])
# php.get_histogram2d(max_dipole / 10., dE, bins=[np.arange(1, 20, 1), np.linspace(-1, 1, 40)], cscale="log")
# plt.show()
#
# fig, ax = plt.subplots(1, 1)
# ax.scatter(10 ** nu_energy_test, 10 ** predicted_nu_energy, s=5, alpha=0.1)
# ax.plot([10 ** 16, 10 ** 20], [10 ** 16, 10 ** 20], "--k")
# ax.set_xlim(1e16, 1e20)
# ax.set_ylim(1e16, 1e20)
# ax.semilogx(True)
# ax.semilogy(True)
# ax.set_aspect("equal")
# ax.set_xlabel("true energy [eV]")
# ax.set_ylabel("predicted energy [eV]")
# fig.tight_layout()
# fig.savefig("plots/scatter.png")
# fig, ax = plt.subplots(1, 1)
# ax.scatter(nu_energy_test, predicted_nu_energy - nu_energy_test, s=5, alpha=0.1)
# ax.set_xlabel("true energy")
# ax.set_ylabel("predicted -true energy")
# fig.tight_layout()
# fig.savefig("plots/scatter_rel.png")
#
# # fig, ax = php.get_histogram(predicted_nu_energy[:, 0], bins=np.arange(17, 20.1, 0.05), xlabel="predicted energy")
# fig, ax = php.get_histogram((predicted_nu_energy - nu_energy_test), bins=np.arange(-2, 2, 0.05),
#                             xlabel=r"$\log_{10}(E_\mathrm{predicted} - E_\mathrm{true}$")
# fig.savefig("plots/pred_true_energy.png")
# fig, ax = php.get_histogram(nu_energy_test, bins=np.arange(16, 20.1, 0.05), xlabel="true energy")
# fig.savefig("plots/true_energy.png")
#
# fig, ax = plt.subplots(1, 1)
# ax.scatter(np.array(labels_tmp.item()["distance"]), predicted_nu_energy - nu_energy_test, s=5, alpha=0.1)
# ax.set_xlabel("distance")
# ax.set_ylabel("predicted - true energy")
# fig.tight_layout()
# fig.savefig("plots/scatter_distance.png")
#
# fig, ax = plt.subplots(1, 1)
# ax.scatter(np.rad2deg(np.array(labels_tmp.item()["nu_zenith"])), predicted_nu_energy - nu_energy_test, s=5, alpha=0.1)
# ax.set_xlabel("neutrino zenith angle [deg]")
# ax.set_ylabel("predicted - true energy")
# fig.savefig("plots/scatter_zenith.png")
# fig.tight_layout()
#
# fig, ax = plt.subplots(1, 1)
# ax.scatter(np.rad2deg(np.array(labels_tmp.item()["nu_azimuth"])), predicted_nu_energy - nu_energy_test, s=5, alpha=0.1)
# ax.set_xlabel("neutrino azimuth angle [deg]")
# ax.set_ylabel("predicted - true energy")
# fig.tight_layout()
# fig.savefig("plots/scatter_azimuth.png")
# plt.show()
