project_name = "nu-dir-reco"
run_version = "runF1"
dataset_name = "SouthPole"

#datapath = "/mnt/ssd2/data/energy_reconstruction/ARIANNA-200_Alvarez2000_3sigma_noise/"
#datapath = "/mnt/md0/data/SouthPole/single_surface_4LPDA_PA_15m_RNOG_fullsim.json/Alvarez2009_had_noise.yaml/G03generate_events_full_surface_sim/LPDA_2of4_100Hz/4LPDA_1dipole_fullband/"
#datapath = "/mnt/md0/data/SouthPole/single_surface_4LPDA_PA_15m_RNOG_fullsim.json/Alvarez2009_had_noise.yaml/G03generate_events_full_surface_sim/v2/LPDA_2of4_100Hz/4LPDA_1dipole_fullband/"
#data_filename = "data_1-3_LPDA_2of4_100Hz_4LPDA_1dipole_fullband_"
#label_filename = "labels_1-3_LPDA_2of4_100Hz_4LPDA_1dipole_fullband_"

# ARZ2020 Askaryan emission model:
datapath = "/mnt/md0/data/SouthPole/single_surface_4LPDA_PA_15m_RNOG_fullsim.json/ARZ2020_emhad_noise.yaml/G03generate_events_full_surface_sim/LPDA_2of4_100Hz/4LPDA_1dipole_fullband/em_had_separately/"
data_filename = "data_had_emhad_1-3_had_1_LPDA_2of4_100Hz_4LPDA_1dipole_fullband_"
label_filename = "labels_had_emhad_1-3_had_1_LPDA_2of4_100Hz_4LPDA_1dipole_fullband_"

plots_dir = "plots"
saved_model_dir = "saved_models"

# This must be a list of ids (even if only testing on 1 file)
test_file_ids = [38, 39, 40]