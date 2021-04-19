import os
import numpy as np
import pickle
#from tensorflow import keras
import time
from toolbox import get_histogram2d
from constants import datapath, n_files, n_files_val, dataset, dataset_name, dataset_em
import datasets
import argparse
from radiotools import helper as hp
from radiotools import stats
#from radiotools import plthelpers
from NuRadioReco.utilities import units
from matplotlib import pyplot as plt
from termcolor import colored
from generate_noise_realizations import load_one_file, realize_noise
import healpy

def get_pred_angle_diff_data():
    prediction_file = f'plots/model.{run_name}.h5_predicted_file_{i_file}_{i_event}_{n_noise_iterations}.pkl'
    with open(prediction_file, "br") as fin:
        nu_direction_predict, nu_direction = pickle.load(fin)

    # Only pick first 100000 data
    # N = 100000
    # nu_direction_predict = nu_direction_predict[:N]
    # nu_direction = nu_direction[:N]

    angle_difference_data = np.array([hp.get_angle(nu_direction_predict[i], nu_direction[0]) for i in range(len(nu_direction_predict))]) / units.deg

    return nu_direction_predict, nu_direction, angle_difference_data

def angle_to_spherical_rad(nu_direction):
    x = nu_direction[0]
    y = nu_direction[1]
    z = nu_direction[2]
    nu_direction_spherical = hp.cartesian_to_spherical(x,y,z)
    theta_deg = nu_direction_spherical[0]
    phi_deg = nu_direction_spherical[1]

    return theta_deg, phi_deg

def angle_to_spherical_deg(nu_direction):
    theta_rad, phi_rad = angle_to_spherical_rad(nu_direction)

    return theta_rad / units.deg, phi_rad / units.deg

# Parse arguments
parser = argparse.ArgumentParser(description='Plot data from antennas')
parser.add_argument("run_id", type=str ,help="the id of the run, eg '3.2' for run3.2")
parser.add_argument("i_file", type=int ,help="the id of the file")
parser.add_argument("i_event", type=int ,help="the id of the event")
parser.add_argument("n_noise_iterations", type=int ,help="amount of noise relizations")
parser.add_argument('--eps', dest='eps', action='store_true')
parser.set_defaults(feature=False)

args = parser.parse_args()
run_id = args.run_id
i_file = args.i_file
i_event = args.i_event
n_noise_iterations = args.n_noise_iterations
eps = args.eps

# Save the run name and filename
run_name = f"run{run_id}"

print(colored(f"Starting skymap plotting (2d histogram) for {run_name}, file {i_file}, event {i_event}...", "yellow"))

# Load data
print("Loading data...")
#data, nu_direction = load_one_file(i_file, i_event)
nu_direction_predict, nu_direction, angle_difference_data = get_pred_angle_diff_data()

# Plot true angles
cartesian_truth = nu_direction[0]
theta_truth_deg, phi_truth_deg = angle_to_spherical_rad(cartesian_truth)


# Plot predicted angles
n_noise_iterations = nu_direction_predict.shape[0]

theta_pred_deg_list = np.zeros(n_noise_iterations)
phi_pred_deg_list = np.zeros(n_noise_iterations)

for i in range(n_noise_iterations):
    cartesian_pred = nu_direction_predict[i]
    theta_pred_deg, phi_pred_deg = angle_to_spherical_deg(cartesian_pred)

    # Append to list of angles
    theta_pred_deg_list[i] = theta_pred_deg
    phi_pred_deg_list[i] = phi_pred_deg

if run_name == "runF1.1":
    emission_model = "Alvarez2009 (had.)"
elif run_name == "runF2.1":
    emission_model = "ARZ2020 (had.)"
elif run_name == "runF3.1":
    emission_model = "ARZ2020 (had. + EM)"

plot_title = f"Skymap for dataset {emission_model}, 2D histogram,\n{n_noise_iterations} noise realizations"
file_name = f"plots/skymap_2dhistogram_{run_name}_file_{i_file}_event_{i_event}_realizations_{n_noise_iterations}.png"
xlabel = r"$\theta (°)$"
ylabel = r"$\phi (°)$"

fig, ax, im = get_histogram2d(theta_pred_deg_list, phi_pred_deg_list, fname=file_name, title=plot_title, xlabel=xlabel, ylabel=ylabel, bins=40)

fig.savefig(file_name)
#plt.close(fig)

# if eps:
#     plt.savefig(f"plots/skymap_2dhistogram_{run_name}_file_{i_file}_event_{i_event}_realizations_{n_noise_iterations}.eps", format="eps")
# else:  
#     plt.savefig()
#plt.savefig("static/moll_nside32_nest.png", dpi=DPI)

print(colored(f"Done plotting skymap (2d histogram) for {run_name}!", "green", attrs=["bold"]))
print("")


