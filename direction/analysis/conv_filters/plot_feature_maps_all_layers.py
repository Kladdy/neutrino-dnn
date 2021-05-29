# Imports
from math import ceil, e, floor
from keras.models import Sequential
from matplotlib import pyplot
from numpy import expand_dims
import numpy as np
import os
import F1F2F3_models
from toolbox import load_one_file
from termcolor import cprint
import argparse
from radiotools import helper as hp
from NuRadioReco.utilities import units

def load_event(i_event):
    event = data[i_event,:,:,:]
    event = expand_dims(event, axis=0)

    return event

def get_biases(layer):
    # retrieve weights from the second hidden layer
    filters, biases = model.layers[layer].get_weights()

    return biases

def find_highest_biases(biases):
    # Find the filters with highest weight
    ind = np.argpartition(biases, -n_filters_to_show)[-n_filters_to_show:]

    # Sort the list
    ind = ind[np.argsort(biases[ind])]

    # Flip list to have maximum index first
    ind_flipped = ind[::-1]

    return ind_flipped

cprint("Beginning to plot feature maps...", "yellow")

parser = argparse.ArgumentParser(description='Plot data from antennas')
parser.add_argument("i_file", type=int ,help="the id of the file")
parser.add_argument("i_event", type=int ,help="the id of the event")

args = parser.parse_args()
i_file = args.i_file
i_event = args.i_event

# Variables
run = "F1.1"
run_name = f"run{run}"
# i_file = 0
# i_event = 45015
base_fig_folder = "/Users/sigge/Dropbox/Universitetet/Kurser/Kandidatarbete/Analysis/neutrino-dnn/direction/analysis/conv_filters/figs/all_layers"

layers_list = np.arange(0,16)

# Load model
model = F1F2F3_models.return_model()
#model.load_weights(f'models/model.{run_name}.h5')
model.load_weights(f'/Users/sigge/Dropbox/Universitetet/Kurser/Kandidatarbete/Analysis/neutrino-dnn/direction/analysis/models/model.{run_name}.h5')

#data, nu_direction = load_file(i_file)

# Load a clean event
#event = load_event(45015)
data, nu_direction = load_one_file(i_file, i_event)
event_data = data[0,:,:,:]
direction_data = nu_direction[0,:]

nu_direction_predict = model.predict(data)

# Get space angle difference
space_angle_difference_deg = hp.get_angle(nu_direction_predict[0], nu_direction[0]) / units.deg

# Plot data

# Make sure plots folder exists
plot_folder = f"{base_fig_folder}/file_{i_file}_event_{i_event}"
if not os.path.exists(plot_folder):
    os.makedirs(plot_folder)
    for i in np.arange(0,4):
        os.makedirs(f"{plot_folder}/block_{i}")


# Getting x axis (1 step is 0.5 ns)
x_axis_double = range(int(len(event_data[0])))
x_axis = [float(x)/2 for x in x_axis_double]

# Plotting
fig, axs = pyplot.subplots(5)
fig.suptitle(f'Plot of 4 LPDA & 1 dipole of SouthPole data for event {i_event} in file {i_file} with 500 MHz bandpass\nwith Alvarez2009 dataset. Space angle difference for event is $\Delta \Psi$={space_angle_difference_deg:.1f}°')

for i in range(5):
    axs[i].plot(x_axis, event_data[i])
    axs[i].set_xlim([min(x_axis), max(x_axis)])
    if i != 4:
        axs[i].set_title(f'LPDA {i+1}')

axs[4].set_title('Dipole')

for ax in axs.flat:
    ax.set(xlabel='time (ns)', ylabel=f'signal (μV)')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

fig.set_size_inches(12, 10)

pyplot.savefig(f"{plot_folder}/signal_file{i_file}_event{i_event}_bandpass_500MHz.png")

fig.clear()
pyplot.close(fig)


max_filters_per_image = 36
columns_per_image = 6

for layer_number in layers_list:
    cprint(f"Doing layer {layer_number}", "blue")

    pyplot.close()

    # Variables
    image_file_number = 0
    ix = 0
    block_number = floor(layer_number / 4)
    text_offset = 0
    #wspace = -0.94
    wspace = -0.97

    if layer_number % 4 == 3:
        print("pooling, skipping...")
        continue

    # redefine model to output right after the first hidden layer
    redefined_model = Sequential(model.layers[0:(layer_number+1)])

    # Make prediction on first event
    feature_maps = redefined_model.predict(data)
    print(feature_maps.shape)
    n_filters_to_show = feature_maps.shape[3]
    n_rows = ceil(n_filters_to_show/columns_per_image)
    aspect = feature_maps.shape[2] / 5.0

    # Find biases and get the index of the top highest
    biases = get_biases(layer_number)
    ind_flipped = find_highest_biases(biases)

    # plot first n filters
    fig_width = 34 / 3.0 * columns_per_image
    f = pyplot.figure(figsize=(fig_width, max_filters_per_image))

    for _ in range(n_filters_to_show):

        #ix_mod_3 = ix % 3

        ax = f.add_subplot(max_filters_per_image, columns_per_image, ix % max_filters_per_image+1)

        # specify subplot and turn of axis
        ax.set_xticks([])
        ax.set_yticks([])

        # plot filter channel in grayscale
        pyplot.imshow(feature_maps[0, :, :, ind_flipped[ix]], cmap='gray', aspect=aspect, interpolation="none")

        pyplot.text(-0.5 + text_offset, 0.6, f'Filter {ix+1}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, size=8)
        pyplot.text(-0.5 + text_offset, 0.3, f'Bias {biases[ind_flipped[ix]]:.2f}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, size=8)

        ix += 1

        if ix == n_filters_to_show or ix % max_filters_per_image == 0:
            #pyplot.subplots_adjust(wspace=0.1, hspace=0.2)
            pyplot.subplots_adjust(wspace=wspace)

            # show the figure
            pyplot.margins(0,0)
            pyplot.savefig(f"{plot_folder}/block_{block_number}/{run_name}_Conv2D_block_{block_number}_layer_{layer_number}_features_file_{image_file_number+1}.png", dpi=950, bbox_inches = 'tight')
            image_file_number += 1
            pyplot.close()
            if ix % max_filters_per_image == 0:
                f = pyplot.figure(figsize=(fig_width, max_filters_per_image))

