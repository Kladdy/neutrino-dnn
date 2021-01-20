import pandas
import matplotlib.pyplot as plt

# Reading data
path = "/Users/sigge/dl1/mnt/md0/sstjaernholm/neutrino-dnn/direction/run1/saved_models/T13"
run = "1.5"
filename = f"model_history_log_run{run}.csv"

df = pandas.read_csv(f'{path}/{filename}')

epoch = df['epoch']
loss = df['loss']
val_loss = df['val_loss']

# Plotting
fig, axs = plt.plot(epoch, loss, "", epoch, val_loss)
plt.title(f'Plot of model loss for run{run}')


plt.xlabel('Epoch')
plt.ylabel("Loss")
plt.legend(["loss", "val_loss"])
#fig.set_size_inches(12, 10)

plt.savefig(f"plots/plot_loss_run{run}.png")
