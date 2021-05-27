# Imports
from matplotlib import pyplot
import numpy as np
import F1F2F3_models


def get_and_normalize_layer(layer):
    # retrieve weights from the second hidden layer
    filters, biases = model.layers[layer].get_weights()
    # normalize filter values to 0-1 so we can visualize them
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)

    return filters, biases

def find_highest_biases(biases):
    # Find the filters with highest weight
    ind = np.argpartition(biases, -n_filters_to_show)[-n_filters_to_show:]

    # Sort the list
    ind = ind[np.argsort(biases[ind])]

    # Flip list to have maximum index first
    ind_flipped = ind[::-1]

    return ind_flipped
    

def plot_first_n_filters(layer, n_filters, plot_column_index, layer_name, block_name):
    filters, biases = get_and_normalize_layer(layer)

    ind_flipped = find_highest_biases(biases)

    print(biases[ind_flipped])

    # Counter for the plots
    j = 0
    # Loop over the n_filters_to_show filters with highest bias
    for i in ind_flipped:

        # get the filter
        f = filters[:, :, :, i]

        # specify subplot and turn of axis
        ax = pyplot.subplot(n_filters_to_show, plot_column_amount, j*plot_column_amount+plot_column_index, label=f"{block_name}, {layer_name}")
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            ax.set_title(layer_name, fontsize=8)
        # plot filter channel in grayscale
        
        pyplot.imshow(f[:, :, 0], cmap='gray', aspect='equal')

        # add label if leftmost
        if plot_column_index == 1:
            number = j+1
            if number == 10:
                pyplot.text(-0.1, 0.4, '$f_{10}$', horizontalalignment='center',
                            verticalalignment='center', transform=ax.transAxes)
            else:
                pyplot.text(-0.1, 0.4, rf'$f_{number}$', horizontalalignment='center',
                            verticalalignment='center', transform=ax.transAxes)

        j += 1



    # pyplot.subplots_adjust(left=0.1,
    #                 bottom=0.1, 
    #                 right=0.9, 
    #                 top=0.9, 
    #                 wspace=0.4, 
    #                 hspace=0.4)
    pyplot.subplots_adjust(hspace=0.1, wspace=0.1)
    pyplot.suptitle(block_name)

# Variables
run = "F1.1"
run_name = f"run{run}"

# Load model
model = F1F2F3_models.return_model()
#model.load_weights(f'models/model.{run_name}.h5')
model.load_weights(f'/Users/sigge/Dropbox/Universitetet/Kurser/Kandidatarbete/Analysis/neutrino-dnn/direction/analysis/models/model.{run_name}.h5')


amount_of_blocks = 4
amount_of_layers_per_block = 3
plot_column_amount = amount_of_layers_per_block

n_filters_to_show = 9

for block in range(amount_of_blocks):
    block_image_name = f"conv_block_{block}"
    block_name = fr"Convolutional block $C_{block+1}$"
    print(f"Doing {block_name}")
    for layer in range(amount_of_layers_per_block):
        layer_name = f"Convolutional layer {layer+1}"
        print(f" - {layer_name}")
        
        plot_first_n_filters(layer=block*amount_of_layers_per_block + layer + block, n_filters=32*2**(block), plot_column_index=layer+1, layer_name=layer_name, block_name=block_name)
    #pyplot.tight_layout()
    pyplot.savefig(f"/Users/sigge/Dropbox/Universitetet/Kurser/Kandidatarbete/Analysis/neutrino-dnn/direction/analysis/conv_filters/figs/{run_name}_filters_{block_image_name}.eps", format="eps")

# # show the figure
# pyplot.show()