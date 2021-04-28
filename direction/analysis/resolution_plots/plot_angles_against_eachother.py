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
from toolbox import load_file, calculate_percentage_interval, get_pred_angle_diff_data_and_angles
import argparse
from termcolor import colored
from constants import datapath, data_filename, label_filename, plots_dir, dataset_run
from scipy.optimize import curve_fit
import matplotlib.patches as mpatches
# -------

# Parse arguments
parser = argparse.ArgumentParser(description='Plot performance data')
parser.add_argument('run', help="run name")
parser.add_argument('--eps', dest='eps', action='store_true', help="flag to image as .eps instead of .png")
parser.add_argument('--fit', dest='fit', action='store_true', help="whether or not to do Rayleigh fit")
parser.set_defaults(eps=False)
parser.set_defaults(fit=False)

args = parser.parse_args()
run= args.run
eps = args.eps
fit = args.fit

# Save the run name and filename
run_name = f"run{run}"

if run_name == "runF1.1":
    emission_model = "Alvarez2009 (had.)"
elif run_name == "runF2.1":
    emission_model = "ARZ2020 (had.)"
elif run_name == "runF3.1":
    emission_model = "ARZ2020 (had. + EM)"

print(colored(f"Plotting angles against eachother for {run_name}...", "yellow"))

# Make sure plots folder exists
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Make sure predicted file exists, otherwise run evaluator
prediction_file = f'{plots_dir}/model.{run_name}.h5_predicted.pkl'
if not os.path.isfile(prediction_file):
    print("Prediction file does not exist, running evaluator...")
    os.system(f"python evaluator.py")

# Get angle difference data
print("Get angle difference data...")
nu_direction_predict, nu_direction, angle_difference_data = get_pred_angle_diff_data_and_angles(run_name)

# Redefine N
N = angle_difference_data.size

theta_truth_rad = np.zeros(N)
phi_truth_rad = np.zeros(N)
theta_pred_rad = np.zeros(N)
phi_pred_rad = np.zeros(N)

print("Convert to spherical coords...")
for i in range(N):
    theta_truth_rad[i], phi_truth_rad[i] = hp.cartesian_to_spherical(*nu_direction[i,:])
    theta_pred_rad[i], phi_pred_rad[i] = hp.cartesian_to_spherical(*nu_direction_predict[i,:])


# theta_truth_rad, phi_truth_rad = hp.cartesian_to_spherical(*nu_direction)
# theta_pred_rad, phi_pred_rad = hp.cartesian_to_spherical(*nu_direction_predict)

theta_truth_deg = theta_truth_rad / units.deg
phi_truth_deg = phi_truth_rad / units.deg
theta_pred_deg = theta_pred_rad / units.deg
phi_pred_deg = phi_pred_rad / units.deg

print("Plot theta...")
# Plot theta against eachother
plt.plot(theta_truth_deg, theta_pred_deg, ',')
plt.title(f"Predicted against true neutrino zenith angle for dataset {emission_model}")

if eps:
    plt.savefig(f"{plots_dir}/theta_against_eachother_{run_name}.eps", format="eps")
else:  
    plt.savefig(f"{plots_dir}/theta_against_eachother_{run_name}.png")



# fig, ax = php.get_histogram(predicted_nu_energy[:, 0], bins=np.arange(17, 20.1, 0.05), xlabel="predicted energy")
# fig, ax = php.get_histogram(angle_difference_data, bins=bins,
#                             xlabel=r"Space angle difference $\Delta \Psi$ (°)", stats=False,
#                             ylabel="Events", kwargs={'color':"lightsteelblue", 'ec':"k"})
#  #                           ylabel="Events", kwargs={'color':"steelblue", 'ec':"steelblue"})
# # ax.plot(xl, N*stats.rayleigh(scale=scale, loc=loc).pdf(xl))


# sigma_68_text = "_{68}"
# # Plot 68 % line
# plt.axvline(x=angle_68, ls="--", label=fr"$\sigma{sigma_68_text}=${angle_68:.2f}", color="forestgreen" )




# # New fit method:
# if fit:
#     def f(x, A, sigma):
#         return A * x/sigma**2 * np.exp(-x**2/2/sigma**2)

#     ax = plt.gca()

#     p = ax.patches

#     xdata = [patch.get_x() for patch in p]
#     ydata = [patch.get_height() for patch in p]

#     #print(xdata)
#     #print(ydata)

#     # line = ax.lines[0]
#     # xdata = line.get_xdata()
#     # ydata = line.get_ydata()

#     popt, pcov = curve_fit(f, xdata, ydata, p0=[700, 5])

#     #print("testing plotting...")
#     #plt.plot(xdata, ydata, label="test")

#     x_fit = np.linspace(0.8*min(xdata), 1.1*max(xdata), 200)

#     popt_abs = np.abs(popt)
#     sigma_value = popt_abs[1]

#     plt.plot(x_fit, f(x_fit, *popt), '-', color="darkorange",
#          label=fr'fit: $\sigma={sigma_value:5.2f}$')

# # Get overflow
# overflow = np.sum(angle_difference_data > bins[-1])

# # Handle legend:
# handles, labels = plt.gca().get_legend_handles_labels() # get existing handles and labels
# overflow_legend_label = fr'Overflow: {(overflow*100.0/float(N)):.0f} %'
# empty_patch_overflow = mpatches.Patch(color='none', label=overflow_legend_label) # create a patch with no color

# handles.append(empty_patch_overflow)  # add new patches and labels to list
# labels.append(overflow_legend_label)

# plt.legend(handles, labels, loc="upper right") # apply new handles and labels to plot


plt.clf()

# Plot phi against eachother
print("Plot phi...")

plt.plot(phi_truth_deg, phi_pred_deg, ',')
plt.title(f"Predicted against true neutrino azimuth angle for dataset {emission_model}")


if eps:
    plt.savefig(f"{plots_dir}/phi_against_eachother_{run_name}.eps", format="eps")
else:  
    plt.savefig(f"{plots_dir}/phi_against_eachother_{run_name}.png")





print(colored(f"Plotted angles against eachother for {run_name}!", "green", attrs=["bold"]))
print("")
