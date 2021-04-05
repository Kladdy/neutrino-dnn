import datasets

project_name = "nu-dir-reco"

# Dataset setup
# Call Dataset(dataset_name, em, noise) with
#     dataset_name:
#         Alvares (only had + noise) / ARZ
#     em:
#         True / False (default)
#     noise:
#         True (default) / False
dataset_name = "Alvarez"
dataset_em = False
dataset_noise = True

dataset = datasets.Dataset(dataset_name, dataset_em, dataset_noise)

# Directories
plots_dir = "plots"
saved_model_dir = "saved_models"
