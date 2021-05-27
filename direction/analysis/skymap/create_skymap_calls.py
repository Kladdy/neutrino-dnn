parallell_calls = 5

file_id = 1
event_start_id = 1

events = 40

# parser.add_argument("run_id", type=str ,help="the id of the run, eg '3.2' for run3.2")
# parser.add_argument("i_file", type=int ,help="the id of the file")
# parser.add_argument("i_event", type=int ,help="the id of the event")
# parser.add_argument("n_noise_iterations", type=int ,help="amount of noise relizations")

for i in range(parallell_calls):
    print(f"screen -S skymap_call_{i}_performance")
    for j in range(events):
        event_id = i*events + j
        print(f"python plot_performance.py F2.1 {file_id} {event_id} {100000}")

    print("")