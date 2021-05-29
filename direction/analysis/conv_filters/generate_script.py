i_file = 0

i_event_list = range(0, 10)
command_list = [f"python plot_feature_maps_all_layers.py {i_file} {i_event}" for i_event in i_event_list]

command = " && ".join(command_list)
print(command)