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
import matplotlib.patches as mpatches
# -------

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
nu_direction_predict, nu_direction, nu_energy, angle_difference_data = get_pred_angle_diff_data()

# Redefine N
N = angle_difference_data.size

# Calculate 68 %
angle_68 = calculate_percentage_interval(angle_difference_data, 0.68)

# fig, ax = php.get_histogram(predicted_nu_energy[:, 0], bins=np.arange(17, 20.1, 0.05), xlabel="predicted energy")
bins = np.linspace(0, 30, 90)
fig, ax = php.get_histogram(angle_difference_data, bins=bins,
                            xlabel=r"Space angle difference $\Delta \Psi$ (°)", stats=False,
                            ylabel="Events", kwargs={'color':"lightsteelblue", 'ec':"k"})
 #                           ylabel="Events", kwargs={'color':"steelblue", 'ec':"steelblue"})
# ax.plot(xl, N*stats.rayleigh(scale=scale, loc=loc).pdf(xl))

if run_name == "runF1.1":
    emission_model = "Alvarez2009 (had.)"
elif run_name == "runF2.1":
    emission_model = "ARZ2020 (had.)"
elif run_name == "runF3.1":
    emission_model = "ARZ2020 (had. + EM)"

plt.title(f"Angular resolution for dataset {emission_model},\nNoise realized, {n_noise_iterations} iterations")

sigma_68_text = "_{68}"
# Plot 68 % line
plt.axvline(x=angle_68, ls="--", label=fr"$\sigma{sigma_68_text}=${angle_68:.2f}", color="forestgreen" )

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

    #print(xdata)
    #print(ydata)

    # line = ax.lines[0]
    # xdata = line.get_xdata()
    # ydata = line.get_ydata()

    popt, pcov = curve_fit(f, xdata, ydata, p0=[700, 5])

    #print("testing plotting...")
    #plt.plot(xdata, ydata, label="test")

    x_fit = np.linspace(0.8*min(xdata), 1.1*max(xdata), 200)

    popt_abs = np.abs(popt)
    sigma_value = popt_abs[1]

    plt.plot(x_fit, f(x_fit, *popt), '-', color="darkorange",
         label=fr'fit: $\sigma={sigma_value:5.2f}$')

# Get overflow
overflow = np.sum(angle_difference_data > bins[-1])

# Handle legend:
handles, labels = plt.gca().get_legend_handles_labels() # get existing handles and labels
overflow_legend_label = fr'Overflow: {(overflow*100.0/float(n_noise_iterations)):.0f} %'
empty_patch_overflow = mpatches.Patch(color='none', label=overflow_legend_label) # create a patch with no color

handles.append(empty_patch_overflow)  # add new patches and labels to list
labels.append(overflow_legend_label)

plt.legend(handles, labels, loc="upper right") # apply new handles and labels to plot


if eps:
    plt.savefig(f"plots/angular_resolution_{run_name}_file_{i_file}_event_{i_event}_realizations_{n_noise_iterations}.eps", format="eps")
else:  
    plt.savefig(f"plots/angular_resolution_{run_name}_file_{i_file}_event_{i_event}_realizations_{n_noise_iterations}.png")

print(colored(f"Saved angular resolution for {run_name}!", "green", attrs=["bold"]))
print("")