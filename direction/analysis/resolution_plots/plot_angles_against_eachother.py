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
from toolbox import load_file, calculate_percentage_interval, get_pred_angle_diff_data_and_angles, get_histogram2d
import argparse
from termcolor import colored
from constants import datapath, data_filename, label_filename, plots_dir, dataset_run
from scipy.optimize import curve_fit
import matplotlib.patches as mpatches
# -------

def fit_gaussian(xdata, ydata):
    print("Fitting gaussian...")
    def f(x, A, mu, sigma):
        return A * 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-1/2*((x-mu)/sigma)**2)


    #popt, pcov = curve_fit(f, xdata, ydata)
    # TODO REMOVE THIS!
    # Shift x-data values to adapt for binning
    xdata = [x + 40/90/2 for x in xdata]
    popt, pcov = curve_fit(f, xdata, ydata, p0=[30000, 0, 1])

    x_fit = np.linspace(1.1*min(xdata), 1.1*max(xdata), 200)

    mu_value = popt[1]
    sigma_value = popt[2]

    plt.plot(x_fit, f(x_fit, *popt), '--', color="tab:red",
         label=fr'Gaussian fit: $\mu$={mu_value:5.2f}, $\sigma$={sigma_value:5.2f}')

def fit_cauchy(xdata, ydata):
    print("Fitting cauchy...")
    def f(x, A, x0, gamma):
        return A * 1/(np.pi*gamma*(1+((x-x0)/gamma)**2))


    #popt, pcov = curve_fit(f, xdata, ydata)
    # TODO REMOVE THIS!
    # Shift x-data values to adapt for binning
    xdata = [x + 40/90/2 for x in xdata]
    popt, pcov = curve_fit(f, xdata, ydata)

    x_fit = np.linspace(1.1*min(xdata), 1.1*max(xdata), 200)

    x0_value = popt[1]
    gamma_value = popt[2]

    plt.plot(x_fit, f(x_fit, *popt), '-', color="darkorange",
         label=fr'Cauchy fit: $x_0$={x0_value:5.2f}, $\gamma$={gamma_value:5.2f}')

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

# Set colormap
cmap="magma"

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

# Make plotting in a deeper folder for easy download
plot_dir = plots_dir + "/against_plots"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

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

# Plot theta against eachother
print("Plot theta...")

fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 1]}, sharex=True)


#axs[0].plot(theta_truth_deg, theta_pred_deg, ',')
get_histogram2d(theta_truth_deg, theta_pred_deg, bins = 90, cscale="log", cmap=cmap, ax1=axs[0],cbi_kwargs={'orientation': 'vertical'})
axs[0].set_title(f"Predicted against true neutrino zenith angle and\nresidual for dataset {emission_model}")
#axs[0].set_xlabel(r"$\theta_{true}$ (°)")
axs[0].set_ylabel(r"$\theta_{pred.}$ (°)")

# Get residuals
theta_residuals = theta_truth_deg - theta_pred_deg

#axs[1].plot(theta_truth_deg, theta_residuals, ',')
get_histogram2d(theta_truth_deg, theta_residuals, bins = 90, cscale="log", cmap=cmap, ax1=axs[1],cbi_kwargs={'orientation': 'vertical'})
#axs[1].set_title(f"Residual of predicted against true neutrino zenith angle\nfor dataset {emission_model}")
axs[1].set_xlabel(r"$\theta_{true}$ (°)")
axs[1].set_ylabel(r"$\theta_{pred.} - \theta_{true}$ (°)")

plt.tight_layout()

if eps:
    plt.savefig(f"{plot_dir}/theta_against_eachother_{run_name}.eps", format="eps")
else:  
    plt.savefig(f"{plot_dir}/theta_against_eachother_{run_name}.png", dpi=600)

# Plot theta histograms
bins = np.linspace(-20, 20, 90)
fig, ax = php.get_histogram(theta_residuals, bins=bins,
                            xlabel=r"$\theta_{pred.} - \theta_{true}$ (°)", stats=False,
                            ylabel="Events", kwargs={'color':"lightsteelblue", 'ec':"k"})
 #                           ylabel="Events", kwargs={'color':"steelblue", 'ec':"steelblue"})


p = ax.patches

x_data = [patch.get_x() for patch in p]
y_data = [patch.get_height() for patch in p]
if fit:
    fit_gaussian(x_data, y_data)
    fit_cauchy(x_data, y_data)

ax.set_title(fr"Histogram of $\theta$ residuals for dataset {emission_model}")
plt.legend()
plt.tight_layout()

if eps:
    plt.savefig(f"{plot_dir}/theta_residuals_histogram_{run_name}.eps", format="eps")
else:  
    plt.savefig(f"{plot_dir}/theta_residuals_histogram_{run_name}.png", dpi=600)

# Clear plot
plt.clf()

# Plot phi against eachother
print("Plot phi...")

fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 1]}, sharex=True)

get_histogram2d(phi_truth_deg, phi_pred_deg, bins = 90, cscale="log", cmap=cmap, ax1=axs[0],cbi_kwargs={'orientation': 'vertical'})
#axs[0].plot(phi_truth_deg, phi_pred_deg, ',')
axs[0].set_title(f"Predicted against true neutrino azimuth angle and\nresidual for dataset {emission_model}")
#axs[0].set_xlabel(r"$\phi_{true}$ (°)")
axs[0].set_ylabel(r"$\phi_{pred.}$ (°)")

# Get residuals

phi_residuals = phi_truth_deg - phi_pred_deg
phi_residuals_above_180_idx = phi_residuals > 180
phi_residuals_under_minus180_idx = phi_residuals < -180
phi_residuals[phi_residuals_above_180_idx] = -360 + phi_residuals[phi_residuals_above_180_idx]
phi_residuals[phi_residuals_under_minus180_idx] = 360 + phi_residuals[phi_residuals_under_minus180_idx]

get_histogram2d(phi_truth_deg, phi_residuals, bins = 90, cscale="log", cmap=cmap, ax1=axs[1],cbi_kwargs={'orientation': 'vertical'})
#axs[1].plot(phi_truth_deg, phi_residuals, ',')
#axs[1].set_title(f"Residual of predicted against true neutrino azimuth angle\nfor dataset {emission_model}")
axs[1].set_xlabel(r"$\phi_{true}$ (°)")
axs[1].set_ylabel(r"$\phi_{pred.} - \phi_{true}$ (°)")

plt.tight_layout()

if eps:
    plt.savefig(f"{plot_dir}/phi_against_eachother_{run_name}.eps", format="eps")
else:  
    plt.savefig(f"{plot_dir}/phi_against_eachother_{run_name}.png", dpi=600)

# Plot phi histograms
bins = np.linspace(-20, 20, 90)
fig, ax = php.get_histogram(phi_residuals, bins=bins,
                            xlabel=r"$\phi_{pred.} - \phi_{true}$ (°)", stats=False,
                            ylabel="Events", kwargs={'color':"lightsteelblue", 'ec':"k"})
 #                           ylabel="Events", kwargs={'color':"steelblue", 'ec':"steelblue"})

p = ax.patches

x_data = [patch.get_x() for patch in p]
y_data = [patch.get_height() for patch in p]
if fit:
    fit_gaussian(x_data, y_data)
    fit_cauchy(x_data, y_data)

ax.set_title(fr"Histogram of $\phi$ residuals for dataset {emission_model}")
plt.legend()
plt.tight_layout()

if eps:
    plt.savefig(f"{plot_dir}/phi_residuals_histogram_{run_name}.eps", format="eps")
else:  
    plt.savefig(f"{plot_dir}/phi_residuals_histogram_{run_name}.png", dpi=600)


print(colored(f"Plotted angles against eachother for {run_name}!", "green", attrs=["bold"]))
print("")

# # Get overflow
# overflow = np.sum(angle_difference_data > bins[-1])

# # Handle legend:
# handles, labels = plt.gca().get_legend_handles_labels() # get existing handles and labels
# overflow_legend_label = fr'Overflow: {(overflow*100.0/float(N)):.0f} %'
# empty_patch_overflow = mpatches.Patch(color='none', label=overflow_legend_label) # create a patch with no color

# handles.append(empty_patch_overflow)  # add new patches and labels to list
# labels.append(overflow_legend_label)

# plt.legend(handles, labels, loc="upper right") # apply new handles and labels to plot
