# Imports
from matplotlib import pyplot
from models import F1F2F3_models


def get_and_normalize_layer(layer):
    # retrieve weights from the second hidden layer
    filters, biases = model.layers[layer].get_weights()
    # normalize filter values to 0-1 so we can visualize them
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)

    return filters

def plot_first_n_filters(layer, n_filters, plot_column_index, layer_name, block_name):
    filters = get_and_normalize_layer(layer)
    
    for i in range(n_filters):
        # get the filter
        f = filters[:, :, :, i]

        # specify subplot and turn of axis
        ax = pyplot.subplot(n_filters, plot_column_amount, i*plot_column_amount+plot_column_index)
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.set_title(layer_name, fontsize=8)
        # plot filter channel in grayscale
        
        pyplot.imshow(f[:, :, 0], cmap='gray')

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

for block in range(amount_of_blocks):
    block_name = f"Block_{block}"
    print(f"Doing {block_name}")
    for layer in range(amount_of_layers_per_block):
        layer_name = f"Conv2D_{block*amount_of_layers_per_block + layer}"
        print(f" - {layer_name}")
        
        plot_first_n_filters(layer=block*amount_of_layers_per_block + layer + block, n_filters=32*2**(block), plot_column_index=layer+1, layer_name=layer_name, block_name=block_name)
    pyplot.savefig(f"/Users/sigge/Dropbox/Universitetet/Kurser/Kandidatarbete/Analysis/neutrino-dnn/direction/analysis/figs/{run_name}_filters_{block_name}.png")

# # show the figure
# pyplot.show()