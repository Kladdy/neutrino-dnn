from Plot_4LPDA_1dipole_SouthPole import load_file
import sys

# Call this function as python plotter.py i_file i_event
i_file = sys.argv[1]
i_event = sys.argv[2]

data = load_file(i_file)

print(data.shape)
print(data[i_event])