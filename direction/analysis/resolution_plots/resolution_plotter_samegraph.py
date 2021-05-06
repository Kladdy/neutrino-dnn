# # GPU allocation
# from gpuutils import GpuUtils
# GpuUtils.allocate(gpu_count=1, framework='keras')

# import tensorflow as tf
# physical_devices = tf.config.list_physical_devices('GPU')
# for device in physical_devices:
#     tf.config.experimental.set_memory_growth(device, True)
# # --------------

# Imports
import matplotlib.pyplot as plt
import numpy as np
from constants import plots_dir, datapath, data_filename, label_filename, test_file_ids, dataset_run
from toolbox import get_pred_angle_diff_data, load_file_all_properties
import sys
import argparse
import os
import time
import pickle
from NuRadioReco.utilities import units
from scipy import stats
from termcolor import colored
from mpl_toolkits.mplot3d import Axes3D
from itertools import product, combinations
from radiotools import plthelpers as php
from tensorflow import keras
from radiotools import helper as hp
# -------

def plot_same(x_data_1, x_data_2, x_data_3, ax1_data_y_1, ax1_data_y_2, ax1_data_y_3, ax2_data_y_1, ax2_data_y_2, ax2_data_y_3):
    print(f"Plotting {file_name}...")

    fig_same, ax1 = plt.subplots()

    ax1.set_xlabel(x_label)
    ax1.set_ylabel(ax1_y_label)

    # Set ax1 to high order to make it be in front so label is in front, and datapoints
    #ax1.set_zorder(1)

    if file_name == "nu_energy":
        ax1.set_xscale('log')

    # Remove last peices of data as their bins are weird for some azimuth
    if file_name == "nu_energy" or file_name == "nu_azimuth" or file_name == "nu_SNR":
        x_data_1 = x_data_1[0:-1]
        x_data_2 = x_data_2[0:-1]
        x_data_3 = x_data_3[0:-1]
        ax1_data_y_1 = ax1_data_y_1[0:-1]
        ax1_data_y_2 = ax1_data_y_2[0:-1]
        ax1_data_y_3 = ax1_data_y_3[0:-1]

    # Remove any bins with zero events for nu_zenith
    if file_name == "nu_zenith":
        ind_count_not_0 = ax2_data_y_1 != 0
        x_data_1 = x_data_1[ind_count_not_0]
        ax1_data_y_1 = ax1_data_y_1[ind_count_not_0]

        ind_count_not_0 = ax2_data_y_2 != 0
        x_data_2 = x_data_2[ind_count_not_0]
        ax1_data_y_2 = ax1_data_y_2[ind_count_not_0]

        ind_count_not_0 = ax2_data_y_3 != 0
        x_data_3 = x_data_3[ind_count_not_0]
        ax1_data_y_3 = ax1_data_y_3[ind_count_not_0]

    # lns1 = ax1.plot(x_data_1, ax1_data_y_1, "*", color=ax1_color, label = emission_models[0])
    # lns2 = ax1.plot(x_data_2, ax1_data_y_2, "*", color=ax1_color, label = emission_models[1])
    # lns3 = ax1.plot(x_data_3, ax1_data_y_3, "*", color=ax1_color, label = emission_models[2])

    lns1 = ax1.plot(x_data_1, ax1_data_y_1, "*", label = emission_models[0], color=colours[0])
    lns2 = ax1.plot(x_data_2, ax1_data_y_2, "*", label = emission_models[1], color=colours[1])
    lns3 = ax1.plot(x_data_3, ax1_data_y_3, "*", label = emission_models[2], color=colours[2])

    # Set axis limits so they are same on all plots
    if file_name == "nu_energy":
        ax1.set_ylim(0, 7)
    elif file_name == "nu_SNR":
        ax1.set_ylim(0, 13.5)
    elif file_name == "nu_zenith":
        ax1.set_ylim(0, 27)
    elif file_name == "nu_azimuth":
        ax1.set_ylim(0, 3.7)

    plt.title(plot_title)

    ax1.legend()

    fig_same.tight_layout()  # otherwise the right y-label is slightly clipped

    #plt.subplots_adjust(top=0.88)
    if eps:
        fig_same.savefig(f"{plot_dir}/sigma68_SAMEPLOT_{file_name}_same_statistic_{statistic}.eps", format="eps", bbox_inches='tight')
    else:
        fig_same.savefig(f"{plot_dir}/sigma68_SAMEPLOT_{file_name}_same_statistic_{statistic}.png", bbox_inches='tight')


# Parse arguments
parser = argparse.ArgumentParser(description='Plot performance data')
parser.add_argument('--eps', dest='eps', action='store_true', help="flag to image as .eps instead of .png")
parser.set_defaults(eps=False)

args = parser.parse_args()
eps = args.eps

# Save the run name
run_names = ["runF1.1", "runF2.1", "runF3.1"]
emission_models = ["Alvarez2009 (had.)", "ARZ2020 (had.)", "ARZ2020 (had. + EM)"]
colours = ["tab:green", "tab:red", "tab:purple"]

print(colored(f"Plotting SAMEPLOTS resolution as function of neutrino properties for...", "yellow"))

# See which statistic to calculate...
statistic = "median"
statistic_string = "Median"
print(f"Calulating with statistic {statistic}...")


# Make sure plots folder exists
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

plot_dir = f"{plots_dir}"

# Make sure folder inside plot_folder exists for the plots
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# Make sure same_data file exists, otherwise run evaluator
same_data_file = f'{plots_dir}/plotdata_{run_names[0]}.npy'
if not os.path.isfile(same_data_file):
    print("Same datafile does not exist!")
    raise Exception

# Load same data file
with open(same_data_file, 'rb') as f:
    nu_energy_bins_1 = np.load(f)
    binned_resolution_nu_energy_1 = np.load(f)
    binned_resolution_nu_energy_count_1 = np.load(f)

    nu_azimuth_bins_1 = np.load(f)
    binned_resolution_nu_azimuth_1 = np.load(f)
    binned_resolution_nu_azimuth_count_1 = np.load(f)

    nu_zenith_bins_1 = np.load(f)
    binned_resolution_nu_zenith_1 = np.load(f)
    binned_resolution_nu_zenith_count_1 = np.load(f)

    SNR_means_1 = np.load(f)
    binned_resolution_SNR_mean_1 = np.load(f)
    binned_resolution_SNR_mean_count_1 = np.load(f)

# Make sure same_data file exists, otherwise run evaluator
same_data_file = f'{plots_dir}/plotdata_{run_names[1]}.npy'
if not os.path.isfile(same_data_file):
    print("Same datafile does not exist!")
    raise Exception

# Load same data file
with open(same_data_file, 'rb') as f:
    nu_energy_bins_2 = np.load(f)
    binned_resolution_nu_energy_2 = np.load(f)
    binned_resolution_nu_energy_count_2 = np.load(f)

    nu_azimuth_bins_2 = np.load(f)
    binned_resolution_nu_azimuth_2 = np.load(f)
    binned_resolution_nu_azimuth_count_2 = np.load(f)

    nu_zenith_bins_2 = np.load(f)
    binned_resolution_nu_zenith_2 = np.load(f)
    binned_resolution_nu_zenith_count_2 = np.load(f)

    SNR_means_2 = np.load(f)
    binned_resolution_SNR_mean_2 = np.load(f)
    binned_resolution_SNR_mean_count_2 = np.load(f)



# Make sure same_data file exists, otherwise run evaluator
same_data_file = f'{plots_dir}/plotdata_{run_names[2]}.npy'
if not os.path.isfile(same_data_file):
    print("Same datafile does not exist!")
    raise Exception

# Load same data file
with open(same_data_file, 'rb') as f:
    nu_energy_bins_3 = np.load(f)
    binned_resolution_nu_energy_3 = np.load(f)
    binned_resolution_nu_energy_count_3 = np.load(f)

    nu_azimuth_bins_3 = np.load(f)
    binned_resolution_nu_azimuth_3 = np.load(f)
    binned_resolution_nu_azimuth_count_3 = np.load(f)

    nu_zenith_bins_3 = np.load(f)
    binned_resolution_nu_zenith_3 = np.load(f)
    binned_resolution_nu_zenith_count_3 = np.load(f)

    SNR_means_3 = np.load(f)
    binned_resolution_SNR_mean_3 = np.load(f)
    binned_resolution_SNR_mean_count_3 = np.load(f)


sigma_68_string = "_{68}"

ax2_data_y_1 = binned_resolution_nu_zenith_count_1
ax2_data_y_2 = binned_resolution_nu_zenith_count_2
ax2_data_y_3 = binned_resolution_nu_zenith_count_3

# Energy resolution & count on same axis
# Constants:
ax1_color = 'tab:blue'
ax2_color = 'tab:orange'
x_label = r"True $\nu$ energy (eV)"
ax1_y_label = fr"{statistic_string} $\sigma{sigma_68_string}$ in bin (°)"
ax2_y_label = "Events"

x_data_1 = nu_energy_bins_1
x_data_2 = nu_energy_bins_2
x_data_3 = nu_energy_bins_3
ax1_data_y_1 = binned_resolution_nu_energy_1
ax1_data_y_2 = binned_resolution_nu_energy_2
ax1_data_y_3 = binned_resolution_nu_energy_3


file_name = "nu_energy"
plot_title = fr"{statistic_string} value of $\sigma{sigma_68_string}$ as a function of $\nu$ energy"
legend_loc = "upper center"
# Constants END

plot_same(x_data_1, x_data_2, x_data_3, ax1_data_y_1, ax1_data_y_2, ax1_data_y_3, ax2_data_y_1, ax2_data_y_2, ax2_data_y_3)
# ______________________________________


# Azimuth resolution & count on same axis
# Constants:
x_label = r"True $\nu$ azimuth angle (°)"
ax1_y_label = fr"{statistic_string} $\sigma{sigma_68_string}$ in bin (°)"
ax2_y_label = "Events"

x_data_1 = nu_azimuth_bins_1
x_data_2 = nu_azimuth_bins_2
x_data_3 = nu_azimuth_bins_3
ax1_data_y_1 = binned_resolution_nu_azimuth_1
ax1_data_y_2 = binned_resolution_nu_azimuth_2
ax1_data_y_3 = binned_resolution_nu_azimuth_3


file_name = "nu_azimuth"
plot_title = fr"{statistic_string} value of $\sigma{sigma_68_string}$ as a function of $\nu$ azimuth angle"
legend_loc = "upper center"
# Constants END

plot_same(x_data_1, x_data_2, x_data_3, ax1_data_y_1, ax1_data_y_2, ax1_data_y_3, ax2_data_y_1, ax2_data_y_2, ax2_data_y_3)
# ______________________________________



# Zenith resolution & count on same axis
# Constants:
x_label = r"True $\nu$ zenith angle (°)"
ax1_y_label = fr"{statistic_string} $\sigma{sigma_68_string}$ in bin (°)"
ax2_y_label = "Events"

x_data_1 = nu_zenith_bins_1
x_data_2 = nu_zenith_bins_2
x_data_3 = nu_zenith_bins_3
ax1_data_y_1 = binned_resolution_nu_zenith_1
ax1_data_y_2 = binned_resolution_nu_zenith_2
ax1_data_y_3 = binned_resolution_nu_zenith_3


file_name = "nu_zenith"
plot_title = fr"{statistic_string} value of $\sigma{sigma_68_string}$ as a function of $\nu$ zenith angle"
legend_loc = "upper left"
# Constants END

plot_same(x_data_1, x_data_2, x_data_3, ax1_data_y_1, ax1_data_y_2, ax1_data_y_3, ax2_data_y_1, ax2_data_y_2, ax2_data_y_3)
# ______________________________________


# SNR resolution & count on same axis
# Constants:
x_label = r"Event SNR"
ax1_y_label = fr"{statistic_string} $\sigma{sigma_68_string}$ in bin (°)"
ax2_y_label = "Events"

x_data_1 = SNR_means_1
x_data_2 = SNR_means_2
x_data_3 = SNR_means_3
ax1_data_y_1 = binned_resolution_SNR_mean_1
ax1_data_y_2 = binned_resolution_SNR_mean_2
ax1_data_y_3 = binned_resolution_SNR_mean_3


file_name = "nu_SNR"
plot_title = fr"{statistic_string} value of $\sigma{sigma_68_string}$ as a function of event SNR"
legend_loc = "upper right"
# Constants END

plot_same(x_data_1, x_data_2, x_data_3, ax1_data_y_1, ax1_data_y_2, ax1_data_y_3, ax2_data_y_1, ax2_data_y_2, ax2_data_y_3)
# ______________________________________

print(colored(f"Plotting  SAMEGRAPHG angular resolution depending on properties!", "green", attrs=["bold"]))
print("")