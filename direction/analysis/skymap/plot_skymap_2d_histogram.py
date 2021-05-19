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
import matplotlib.patches as mpatches

def get_pred_angle_diff_data():
    prediction_file = f'plots/model.{run_name}.h5_predicted_file_{i_file}_{i_event}_{n_noise_iterations}.pkl'
    with open(prediction_file, "br") as fin:
        nu_direction_predict, nu_direction, nu_energy = pickle.load(fin)

    # Only pick first 100000 data
    # N = 100000
    # nu_direction_predict = nu_direction_predict[:N]
    # nu_direction = nu_direction[:N]

    angle_difference_data = np.array([hp.get_angle(nu_direction_predict[i], nu_direction[0]) for i in range(len(nu_direction_predict))]) / units.deg

    return nu_direction_predict, nu_direction, nu_energy, angle_difference_data

# Parse arguments
parser = argparse.ArgumentParser(description='Plot data from antennas')
parser.add_argument("run_id", type=str ,help="the id of the run, eg '3.2' for run3.2")
parser.add_argument("i_file", type=int ,help="the id of the file")
parser.add_argument("i_event", type=int ,help="the id of the event")
parser.add_argument("n_noise_iterations", type=int ,help="amount of noise relizations")
parser.add_argument('--eps', dest='eps', action='store_true')
parser.set_defaults(eps=False)

args = parser.parse_args()
run_id = args.run_id
i_file = args.i_file
i_event = args.i_event
n_noise_iterations = args.n_noise_iterations
eps = args.eps

# Save the run name and filename
run_name = f"run{run_id}"

print(colored(f"Starting skymap plotting (2d histogram) for {run_name}, file {i_file}, event {i_event}...", "yellow"))

# Make sure predicted file exists, otherwise run evaluator
prediction_file = f'plots/model.{run_name}.h5_predicted_file_{i_file}_{i_event}_{n_noise_iterations}.pkl'
if not os.path.isfile(prediction_file):
    print("Prediction file does not exist, running evaluator...")
    os.system(f"python evaluate_skymap.py {run_id} {i_file} {i_event} {n_noise_iterations}")

# Load data
print("Loading data...")
#data, nu_direction = load_one_file(i_file, i_event)
nu_direction_predict, nu_direction, nu_energy, angle_difference_data = get_pred_angle_diff_data()

# Get energy:
nu_energy = nu_energy[0]

# Get true angles
print("Getting true angles...")
cartesian_truth = nu_direction[0]
theta_truth_rad, phi_truth_rad = hp.cartesian_to_spherical(*cartesian_truth)


# Get predicted angles
print("Getting predicted angles...")
n_noise_iterations = nu_direction_predict.shape[0]

theta_pred_rad_array = np.zeros(n_noise_iterations)
phi_pred_rad_array = np.zeros(n_noise_iterations)

for i in range(n_noise_iterations):
    cartesian_pred = nu_direction_predict[i]
    theta_pred_rad, phi_pred_rad = hp.cartesian_to_spherical(*cartesian_pred)

    # Append to array of angles
    theta_pred_rad_array[i] = theta_pred_rad
    phi_pred_rad_array[i] = phi_pred_rad

if run_name == "runF1.1":
    emission_model = "Alvarez2009 (had.)"
elif run_name == "runF2.1":
    emission_model = "ARZ2020 (had.)"
elif run_name == "runF3.1":
    emission_model = "ARZ2020 (had. + EM)"

plot_title = f"Skymap for dataset {emission_model}, 2D histogram,\nNoise realized, {n_noise_iterations} iterations"
file_name = f"plots/skymap_2dhistogram_{run_name}_file_{i_file}_event_{i_event}_realizations_{n_noise_iterations}.png"
ylabel = r"$\theta$ (°)"
xlabel = r"$\phi$ (°)"

cmap = "BuPu"
bin_amount = 40
if i_event == 10:
    bins = [np.linspace(0, 90, bin_amount), np.linspace(0, 20, bin_amount)] # event 10
elif i_event == 12:
    bins = [np.linspace(-110, -70, bin_amount), np.linspace(-10, 5, bin_amount)] # event 12
else:
    raise ValueError(f"Binnig not defined for event {i_event}")
fig, ax, im = get_histogram2d(-phi_pred_rad_array / units.deg, (np.pi/2 - theta_pred_rad_array) / units.deg, fname=file_name, title=plot_title, xlabel=xlabel, ylabel=ylabel, bins=bins, cmap=cmap)
#fig, ax, im = get_histogram2d(phi_pred_rad_array / units.deg, theta_pred_rad_array / units.deg, fname=file_name, title=plot_title, xlabel=xlabel, ylabel=ylabel, bins=40)

energy_string = "{:.1e}".format(nu_energy)
print(energy_string)
nu_text = r"_\nu"
legend_text = fr"E${nu_text}$ = {energy_string} eV"

plt.plot(-phi_truth_rad / units.deg, (np.pi/2 - theta_truth_rad) / units.deg, "x", markersize=10, c="green", markeredgewidth=1, label=legend_text) # For legend
plt.plot(-phi_truth_rad / units.deg, (np.pi/2 - theta_truth_rad) / units.deg, "x", markersize=40, c="k", markeredgecolor="k", markeredgewidth=2)
plt.plot(-phi_truth_rad / units.deg, (np.pi/2 - theta_truth_rad) / units.deg, "x", markersize=40, c="lawngreen", markeredgewidth=1)

colat_deg = (np.pi/2 - theta_truth_rad) / units.deg
lon_deg = phi_truth_rad / units.deg

# Handle legend:
handles, labels = plt.gca().get_legend_handles_labels() # get existing handles and labels
colat_legend_label = fr'Colatitude: {colat_deg:.0f}°'
lon_legend_label = fr'Longitude: {-lon_deg:.0f}°'
empty_patch_theta = mpatches.Patch(color='none', label=colat_legend_label) # create a patch with no color
empty_patch_phi = mpatches.Patch(color='none', label=lon_legend_label) # create a patch with no color

handles.append(empty_patch_theta)  # add new patches and labels to list
handles.append(empty_patch_phi)  
labels.append(colat_legend_label)
labels.append(lon_legend_label)

plt.legend(handles, labels, loc="upper right", prop={'size': 10}) # apply new handles and labels to plot

print("theta_pred_rad_array:", theta_pred_rad_array)
print("phi_pred_rad_array:", phi_pred_rad_array)

plt.tight_layout()
# fig.savefig(file_name)

if eps:
    plt.savefig(f"plots/skymap_2dhistogram_{run_name}_file_{i_file}_event_{i_event}_realizations_{n_noise_iterations}.eps", format="eps")
else:  
    plt.savefig(f"plots/skymap_2dhistogram_{run_name}_file_{i_file}_event_{i_event}_realizations_{n_noise_iterations}.png")
#plt.savefig("static/moll_nside32_nest.png", dpi=DPI)

print(colored(f"Done plotting skymap (2d histogram) for {run_name}!", "green", attrs=["bold"]))
print("")


