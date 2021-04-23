import os
import numpy as np
import pickle
#from tensorflow import keras
import time
#from toolbox import load_file
from constants import datapath, n_files, n_files_val, dataset, dataset_name, dataset_em
import datasets
import argparse
from radiotools import helper as hp
from radiotools import stats
from NuRadioReco.utilities import units
from matplotlib import pyplot as plt
from termcolor import colored
from generate_noise_realizations import load_one_file, realize_noise
import healpy
from astrotools import skymap

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

# Parse arguments
parser = argparse.ArgumentParser(description='Plot data from antennas')
parser.add_argument("run_id", type=str ,help="the id of the run, eg '3.2' for run3.2")
parser.add_argument("i_file", type=int ,help="the id of the file")
parser.add_argument("i_event", type=int ,help="the id of the event")
parser.add_argument("n_noise_iterations", type=int ,help="amount of noise relizations")
parser.add_argument("nside", type=int ,help="nside for plotting")
parser.add_argument('--eps', dest='eps', action='store_true', help="flag to image as .eps instead of .png")
parser.set_defaults(eps=False)

args = parser.parse_args()
run_id = args.run_id
i_file = args.i_file
i_event = args.i_event
n_noise_iterations = args.n_noise_iterations
nside = args.nside
eps = args.eps

# Save the run name and filename
run_name = f"run{run_id}"

print(colored(f"Starting skymap plotting (Mollweide projection) for {run_name}, file {i_file}, event {i_event}...", "yellow"))

# Make sure predicted file exists, otherwise run evaluator
prediction_file = f'plots/model.{run_name}.h5_predicted_file_{i_file}_{i_event}_{n_noise_iterations}.pkl'
if not os.path.isfile(prediction_file):
    print("Prediction file does not exist, running evaluator...")
    os.system(f"python evaluate_skymap.py {run_id} {i_file} {i_event} {n_noise_iterations}")

# Load data
print("Loading data...")
#data, nu_direction = load_one_file(i_file, i_event)
nu_direction_predict, nu_direction, angle_difference_data = get_pred_angle_diff_data()

# Get true angles
cartesian_truth = nu_direction[0]
theta_truth_deg, phi_truth_deg = hp.cartesian_to_spherical(*cartesian_truth)


# Get predicted angles
n_noise_iterations = nu_direction_predict.shape[0]

theta_pred_rad_array = np.zeros(n_noise_iterations)
phi_pred_rad_array = np.zeros(n_noise_iterations)

for i in range(n_noise_iterations):
    cartesian_pred = nu_direction_predict[i]
    #theta_pred_rad, phi_pred_rad = angle_to_spherical_rad(cartesian_pred)
    theta_pred_rad, phi_pred_rad = hp.cartesian_to_spherical(*cartesian_pred)

    # Append to array of angles
    theta_pred_rad_array[i] = theta_pred_rad
    phi_pred_rad_array[i] = phi_pred_rad

# healpy plotting:
npix = healpy.nside2npix(nside)

# convert to HEALPix indices
indices = healpy.ang2pix(nside, theta_pred_rad_array, phi_pred_rad_array)

idx, counts = np.unique(indices, return_counts=True)

# fill the fullsky map
hpx_map = np.zeros(npix, dtype=int)
hpx_map[idx] = counts

# Find the emission model name based on which run we are doing
if run_name == "runF1.1":
    emission_model = "Alvarez2009 (had.)"
elif run_name == "runF2.1":
    emission_model = "ARZ2020 (had.)"
elif run_name == "runF3.1":
    emission_model = "ARZ2020 (had. + EM)"

# Plot the Mollweide projection
plot_title = f"Skymap for dataset {emission_model}, Mollweide projection,\n{n_noise_iterations} noise realizations\n"

fig, cb = skymap.heatmap(hpx_map, cmap="magma", label="")

plt.title(plot_title, fontsize=20)

#healpy.mollview(np.log10(hpx_map+1))
#healpy.mollview(hpx_map, cmap="cividis", title=plot_title, xsize=3200, flip="geo")
#healpy.graticule()

print(f"Do save as eps: {eps}")

if eps:
    plt.savefig(f"plots/skymap_mollweide_astrotools_{run_name}_file_{i_file}_event_{i_event}_realizations_{n_noise_iterations}_nside_{nside}.eps", format="eps")
else:  
    plt.savefig(f"plots/skymap_mollweide_astrotools_{run_name}_file_{i_file}_event_{i_event}_realizations_{n_noise_iterations}_nside_{nside}.png")
#plt.savefig("static/moll_nside32_nest.png", dpi=DPI)

print("theta_pred_rad_array:", theta_pred_rad_array)
print("phi_pred_rad_array:", phi_pred_rad_array)


print("theta_pred_rad_array (deg):", theta_pred_rad_array /units.deg)
print("phi_pred_rad_array (deg):", phi_pred_rad_array /units.deg)

# # Plot skypatch DOESNT WORK AS SKYPATCH cant take heatmaps :-(
# ax = plt.gca()
# print(ax.collections)
# print(ax.collections[0])
# print(type(ax.collections[0]))

# mappable = ax.collections[0]

# patch = skymap.PlotSkyPatch(lon_roi=np.deg2rad(-30), lat_roi=np.deg2rad(0), r_roi=0.6, title='Skypatch')

# patch.mark_roi()
# patch.plot_grid()
# patch.plot(theta_pred_rad_array, phi_pred_rad_array, ls='--')
# patch.colorbar(mappable)
# patch.savefig(f"plots/skymap_mollweide_astrotools_skypatch_{run_name}_file_{i_file}_event_{i_event}_realizations_{n_noise_iterations}_nside_{nside}.png")
# #skymap.PlotSkyPatch


print(colored(f"Done plotting skymap (Mollweide projection) for {run_name}!", "green", attrs=["bold"]))
print("")


