import numpy as np
import os


# path to the map with .txt files:
path = r"./Results 08-06-2026"

# name under which the output file will be saved:
save_name = "errors"


# outputs the mean jet length and standard deviation:

def mean_length(path):
    lengths = []
    with open(path, "r") as f:
        for i, regel in enumerate(f, start = 1):
            if i >= 14:
                length = float(regel.split()[4])
                lengths.append(length)
    lengths = np.array(lengths)
    return (float(lengths.mean()), float(lengths.std()))


# outputs the mean jet lengths and standard deviations for
# each file in a map, along with the names of the files:

def mean_lengths(path):
    names = []
    mean_lengths = []
    errors = []
    for entry in os.scandir(path):
        name = entry.name
        extension = name.split(".")[-1]
        if extension == "txt" and name != f"{save_name}.txt":
            mean, error = mean_length(entry.path)
            names.append(name)
            mean_lengths.append(mean)
            errors.append(error)
    return (names, mean_lengths, errors)
            

# saves the mean jet lengths and standard deviation in a
# text file:

save_path = f"{path}\\{save_name}.txt"
names, means, errors = mean_lengths(path)
lines = []

lines.append("Density (kg/m^3): \n")
lines.append("Surface tension (mNm): \n")
lines.append("Viscosity (mPa s): \n")
lines.append("Needle diameter (micrometers): \n")
lines.append("\n")

for name, mean, error in zip(names, means, errors):
    lines.append(f"{name}:\n")
    lines.append("flow rate (g/min): \n")
    lines.append(f"mean jet length (mm): {mean}\n")
    lines.append(f"standard deviation (mm): {error}\n")
    lines.append("\n")

with open(save_path, "w") as f:
    f.writelines(lines)