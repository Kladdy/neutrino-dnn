import datasets

project_name = "nu-dir-reco"

# Data set setup
dataset_name = "Alvarez"
dataset_em = False
dataset_noise = True

dataset = datasets.Dataset(dataset_name, dataset_em, dataset_noise)

# Directories
plots_dir = "plots"
saved_model_dir = "saved_models"
