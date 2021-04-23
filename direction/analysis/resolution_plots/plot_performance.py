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
from toolbox import load_file, calculate_percentage_interval, get_pred_angle_diff_data
import argparse
from termcolor import colored
from constants import datapath, data_filename, label_filename, plots_dir, dataset_run
from scipy.optimize import curve_fit
# -------

# Parse arguments
parser = argparse.ArgumentParser(description='Plot performance data')
parser.add_argument('--eps', dest='eps', action='store_true', help="flag to image as .eps instead of .png")
parser.add_argument('--fit', dest='fit', action='store_true', help="whether or not to do Rayleigh fit")
parser.set_defaults(eps=False)
parser.set_defaults(fit=False)

args = parser.parse_args()
eps = args.eps
fit = args.fit

# Save the run name and filename
run_name = f"run{dataset_run}"

print(colored(f"Plotting angular resolution for {run_name}...", "yellow"))

# Make sure plots folder exists
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Make sure predicted file exists, otherwise run evaluator
prediction_file = f'{plots_dir}/model.{run_name}.h5_predicted.pkl'
if not os.path.isfile(prediction_file):
    print("Prediction file does not exist, running evaluator...")
    os.system(f"python evaluator.py")

# Get angle difference data
angle_difference_data = get_pred_angle_diff_data(run_name)

# Redefine N
N = angle_difference_data.size

# Calculate 68 %
angle_68 = calculate_percentage_interval(angle_difference_data, 0.68)

# fig, ax = php.get_histogram(predicted_nu_energy[:, 0], bins=np.arange(17, 20.1, 0.05), xlabel="predicted energy")
fig, ax = php.get_histogram(angle_difference_data, bins=np.linspace(0, 30, 90),
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

plt.title(f"Angular resolution for dataset {emission_model}")

sigma_68_text = "_{68}"
# Plot 68 % line
plt.axvline(x=angle_68, ls="--", label=fr"$\sigma{sigma_68_text}=${angle_68:.2f}", color="forestgreen" )


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

    plt.plot(x_fit, f(x_fit, *popt), '-', color="darkorange",
         label=r'fit: A=%5.0f, $\sigma=$%5.2f' % tuple(np.abs(popt)))

plt.legend()

if eps:
    plt.savefig(f"{plots_dir}/angular_resolution_{run_name}.eps", format="eps")
else:  
    plt.savefig(f"{plots_dir}/angular_resolution_{run_name}.png")

print(colored(f"Plotted angular resolution for {run_name}!", "green", attrs=["bold"]))
print("")


# OLD SNR code, irrelevant!
# # plt.show()

# SNR_bins = np.append(np.arange(1, 20, 1), [10000])
# SNR_means = np.arange(1.5, 20.5, 1)

# mean = stats.binned_statistic(max_LPDA[:, 0] / 10., angle_difference_data, bins=SNR_bins)[0]
# std = stats.binned_statistic(max_LPDA[:, 0] / 10., angle_difference_data, bins=SNR_bins, statistic='std')[0]
# fig, ax = plt.subplots(1, 1)
# # ax.errorbar(SNR_means, mean, yerr=std, fmt="o")
# ax.plot(SNR_means, mean, "o")
# # ax.set_ylim(0, 0.4)
# ax.set_xlabel("max SNR LPDA")
# ax.set_ylabel("angular resolution")
# fig.tight_layout()
# fig.savefig(f"{plots_dir}/mean_maxSNRLPDA_{run_name}.png")
# # plt.show()