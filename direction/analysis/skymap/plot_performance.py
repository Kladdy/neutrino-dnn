# Imports
import numpy as np
from radiotools import plthelpers as php
from matplotlib import pyplot as plt
import os
import time
import pickle
from scipy import stats
from radiotools import helper as hp
from NuRadioReco.utilities import units
from toolbox import calculate_percentage_interval
import argparse
from termcolor import colored
from scipy.optimize import curve_fit
# -------

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
parser.add_argument('--eps', dest='eps', action='store_true')
parser.add_argument('--fit', dest='fit', action='store_true')
parser.set_defaults(eps=False)
parser.set_defaults(fit=False)

args = parser.parse_args()
run_id = args.run_id
i_file = args.i_file
i_event = args.i_event
n_noise_iterations = args.n_noise_iterations
eps = args.eps
fit = args.fit

# Save the run name and filename
run_name = f"run{run_id}"

print(colored(f"Plotting angular resolution for {run_name}...", "yellow"))

# Make sure predicted file exists, otherwise run evaluator
prediction_file = f'plots/model.{run_name}.h5_predicted_file_{i_file}_{i_event}_{n_noise_iterations}.pkl'
if not os.path.isfile(prediction_file):
    print("Prediction file does not exist, running evaluator...")
    os.system(f"python evaluate_skymap.py {run_id} {i_file} {i_event} {n_noise_iterations}")

# Load data
print("Loading data...")
#data, nu_direction = load_one_file(i_file, i_event)
nu_direction_predict, nu_direction, angle_difference_data = get_pred_angle_diff_data()

# Redefine N
N = angle_difference_data.size

# Calculate 68 %
angle_68 = calculate_percentage_interval(angle_difference_data, 0.68)

# fig, ax = php.get_histogram(predicted_nu_energy[:, 0], bins=np.arange(17, 20.1, 0.05), xlabel="predicted energy")
fig, ax = php.get_histogram(angle_difference_data, bins=np.linspace(0, 40, 90),
                            xlabel=r"angular difference $\Delta \Psi$ (°)")
# ax.plot(xl, N*stats.rayleigh(scale=scale, loc=loc).pdf(xl))
plt.title(f"Angular resolution for {run_name} with\n68 % at angle difference of {angle_68:.2f}")

# OLD FIT METHOD
# if fit: 
#     rayleigh_fit = stats.rayleigh.fit(angle_difference_data)

#     # linspace for fit
#     x_fit = np.linspace(0,40,100)
#     # fitted distribution
#     pdf_fitted = stats.rayleigh.pdf(x_fit,loc=rayleigh_fit[0],scale=rayleigh_fit[1])

#     #plot_scale = N/2
#     plot_scale = 1

#     plt.plot(x_fit, pdf_fitted*plot_scale, label=r'Rayleigh fit: loc=%5.2f, scale=%5.2f' % tuple(rayleigh_fit))

# New fit method:
if fit:
    def f(x, A, sigma):
        return A * x/sigma**2 * np.exp(-x**2/2/sigma**2)

    ax = plt.gca()

    p = ax.patches

    xdata = [patch.get_x() for patch in p]
    ydata = [patch.get_height() for patch in p]

    print(xdata)
    print(ydata)

    # line = ax.lines[0]
    # xdata = line.get_xdata()
    # ydata = line.get_ydata()

    popt, pcov = curve_fit(f, xdata, ydata, p0=[700, 5])

    print("testing plotting...")
    plt.plot(xdata, ydata, label="test")

    x_fit = np.linspace(0.8*min(xdata), 1.1*max(xdata))

    plt.plot(x_fit, f(x_fit, *popt), '-', color="mediumorchid",
         label=r'fit: A=%5.2f, $\sigma$=%5.2f' % tuple(popt))

plt.legend()

if eps:
    plt.savefig(f"plots/angular_resolution_{run_name}_file_{i_file}_event_{i_event}_realizations_{n_noise_iterations}.eps", format="eps")
else:  
    plt.savefig(f"plots/angular_resolution_{run_name}_file_{i_file}_event_{i_event}_realizations_{n_noise_iterations}.png")

print(colored(f"Saved angular resolution for {run_name}!", "green", attrs=["bold"]))
print("")