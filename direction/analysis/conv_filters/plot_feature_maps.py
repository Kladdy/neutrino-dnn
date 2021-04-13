# Imports
from keras.models import Model
from matplotlib import pyplot
from numpy import expand_dims
import numpy as np
import F1F2F3_models
from toolbox import load_file

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

# Variables
run = "F1.1"
run_name = f"run{run}"
i_file = 0

# Load model
model = F1F2F3_models.return_model()
#model.load_weights(f'models/model.{run_name}.h5')
model.load_weights(f'/Users/sigge/Dropbox/Universitetet/Kurser/Kandidatarbete/Analysis/neutrino-dnn/direction/analysis/models/model.{run_name}.h5')

data, nu_direction = load_file(i_file)

# redefine model to output right after the first hidden layer
model = Model(inputs=model.inputs, outputs=model.layers[0].output)

# Load a clean event
event = load_event(45015)

# Make prediction on first event
feature_maps = model.predict(event)

# # plot all 32 maps in an 8x4 squares
# columns = 4
# rows = 8
# ix = 1
# for _ in range(columns):
# 	for _ in range(rows):
# 		# specify subplot and turn of axis
# 		ax = pyplot.subplot(columns, rows, ix)
# 		ax.set_xticks([])
# 		ax.set_yticks([])
# 		# plot filter channel in grayscale
# 		pyplot.imshow(feature_maps[0, :, :, ix-1], cmap='gray', aspect=100, interpolation="none")
# 		ix += 1

# Variables
n_filters_to_show = 5

# Find biases and get the index of the top highest
biases = get_biases(1)
ind_flipped = find_highest_biases(biases)

# plot first n filters
ix = 0
for _ in range(n_filters_to_show):

	# specify subplot and turn of axis
	ax = pyplot.subplot(n_filters_to_show, 1, ix+1)
	ax.set_xticks([])
	ax.set_yticks([])
	# plot filter channel in grayscale
	pyplot.imshow(feature_maps[0, :, :, ind_flipped[ix]], cmap='gray', aspect=100, interpolation="none")
	ix += 1

# show the figure
pyplot.show()