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

args = parser.parse_args()
run_id = args.run_id
i_file = args.i_file
i_event = args.i_event
n_noise_iterations = args.n_noise_iterations

# Save the run name and filename
run_name = f"run{run_id}"

print(colored(f"Starting skymap plotting for {run_name}, file {i_file}, event {i_event}...", "yellow"))

# Make sure predicted file exists, otherwise run evaluator
prediction_file = f'plots/model.{run_name}.h5_predicted_file_{i_file}_{i_event}_{n_noise_iterations}.pkl'
if not os.path.isfile(prediction_file):
    print("Prediction file does not exist, running evaluator...")
    os.system(f"python evaluate_skymap.py {run_id} {i_file} {i_event} {n_noise_iterations}")

# Load data
print("Loading data...")
#data, nu_direction = load_one_file(i_file, i_event)
nu_direction_predict, nu_direction, angle_difference_data = get_pred_angle_diff_data()

fig, ax = plt.subplots(1)

# Plot true angles
cart_truth = nu_direction[0]
x = cart_truth[0]
y = cart_truth[1]
z = cart_truth[2]
nu_direction_spherical = hp.cartesian_to_spherical(x,y,z)


ax.plot(nu_direction_spherical[0] / units.deg, nu_direction_spherical[1] / units.deg, "r*", zorder=2)

ax.set_xlabel(r"$\theta$ (°)")
ax.set_ylabel(r"$\phi$ (°)")

# Plot predicted angles
n_noise_iterations = nu_direction_predict.shape[0]

for i in range(n_noise_iterations):
    cart_pred = nu_direction_predict[i]
    x = cart_pred[0]
    y = cart_pred[1]
    z = cart_pred[2]
    nu_direction_pred_spherical = hp.cartesian_to_spherical(x,y,z)
    theta_deg = nu_direction_pred_spherical[0] / units.deg
    phi_deg = nu_direction_pred_spherical[1] / units.deg

    #print(f"theta: {theta_deg}, phi: {phi_deg}")
    ax.plot(theta_deg, phi_deg, "b.", zorder=1)


plt.title(f"Skymap for model {run_name}, file {i_file}, event {i_event}")
plt.legend(["True", f"Predicted ({n_noise_iterations} realizations)"], loc="upper left")

fig.savefig(f"plots/skymap_{run_name}_file_{i_file}_event_{i_event}_realizations_{n_noise_iterations}.png")

print(colored(f"Done plotting skymap for {run_name}!", "green", attrs=["bold"]))
print("")


