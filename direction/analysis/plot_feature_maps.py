# Imports
from keras.models import Model
from matplotlib import pyplot
from numpy import expand_dims
import F1F2F3_models
from toolbox import load_file

def load_event(i_event):
    event = data[i_event,:,:,:]
    event = expand_dims(event, axis=0)

    return event


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

# plot all 32 maps in an 8x4 squares
columns = 4
rows = 8
ix = 1
for _ in range(columns):
	for _ in range(rows):
		# specify subplot and turn of axis
		ax = pyplot.subplot(columns, rows, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		# plot filter channel in grayscale
		pyplot.imshow(feature_maps[0, :, :, ix-1], cmap='gray', aspect=100)
		ix += 1
# show the figure
pyplot.show()