import datasets

project_name = "nu-dir-reco"
run_version = "runF1"
dataset_name = "SouthPole"

# Dataset setup
dataset_name = "Alvarez"
dataset_em = False
dataset_noise = True

dataset = datasets.Dataset(dataset_name, dataset_em, dataset_noise)

test_file_ids = dataset.test_file_ids
datapath = dataset.datapath
data_filename = dataset.data_filename
label_filename = dataset.label_filename
n_files = dataset.n_files
n_files_val = dataset.n_files_val

# Directories
plots_dir = "plots"
saved_model_dir = "saved_models"
