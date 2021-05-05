import datasets

# Dataset setup
# Call Dataset(dataset_name, em, noise) with
#     dataset_name:
#         ALVAREZ (only had + noise) / ARZ
#     em:
#         True / False (default)
#     noise:
#         True (default) / False
dataset_run = "F3.1"

if dataset_run == "F1.1":
    dataset_name = "ALVAREZ"
    dataset_em = False
    dataset_noise = True
if dataset_run == "F2.1":
    dataset_name = "ARZ"
    dataset_em = False
    dataset_noise = True
if dataset_run == "F3.1":
    dataset_name = "ARZ"
    dataset_em = True
    dataset_noise = True

dataset = datasets.Dataset(dataset_name, dataset_em, dataset_noise)

test_file_ids = dataset.test_file_ids
datapath = dataset.datapath
data_filename = dataset.data_filename
label_filename = dataset.label_filename
n_files = dataset.n_files
n_files_val = dataset.n_files_val

plots_dir = "plots"