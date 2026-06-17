import numpy as np
import matplotlib.pyplot as plt
import math


rho = 964.9
sigma = 0.04370
D = 460 * 10 ** (-6)
A = math.pi * D ** 2 / 4

flow_rates = np.array([
    5.1530,
    5.2935,
    5.4591,
    5.6726,
    5.9732,
    6.1004,
    6.5118,
    6.9042,
    7.3238,
    7.7976,
    8.2209,
    8.6368,
    8.9901,
    9.2961,
    9.6190,
    10.1360,
    10.5778,
    11.0492,
    11.4525,
    11.6668,
    11.9732,
    12.5956,
    12.8185,
    13.0088,
    13.5597,
    13.7405
]) / (6 * 10 ** 4)

jet_lengths = np.array([
    6.079,
    10.631,
    11.148,
    11.758,
    11.997,
    12.498,
    13.411,
    14.713,
    15.691,
    16.846,
    17.686,
    18.915,
    19.687,
    21.634,
    22.682,
    24.230,
    25.854,
    26.698,
    28.684,
    28.881,
    30.454,
    31.021,
    32.137,
    34.081,
    33.908,
    35.737
]) / 1000


h_over_D = jet_lengths / D
u = flow_rates / (rho * A)
We = (rho * u ** 2 * D) / (sigma)


fig = plt.figure()
ax = fig.add_axes([0.15, 0.1, 0.8, 0.8])
ax.plot(We, h_over_D, marker="*", markersize = None, ls="None", color = "blue")
ax.set_title("Jet Breakup")
ax.set_xlabel("$We$")
ax.set_ylabel(r"$\frac{h_0}{D}$")


plt.show()
fig.savefig("./Plots\Jet_lengths_08-06-2026")