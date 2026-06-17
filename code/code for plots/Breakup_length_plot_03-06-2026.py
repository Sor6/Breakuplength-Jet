import numpy as np
import matplotlib.pyplot as plt
import math


rho = 1000
sigma = 0.072
D = 231.6 * 10 ** (-6)
A = math.pi * D ** 2 / 4

flow_rates = np.array([
    2.9627,
    3.1148,
    3.3677,
    3.6485,
    3.9988,
    4.4273,
    4.9103,
    5.3818,
    5.7782,
    6.4253,
    6.7142,
    7.2946,
    7.5486,
    8.0409,
    8.4172,
    8.9166,
    9.3318,
    9.6441,
    9.8873,
    10.2136,
    10.6537,
    10.8663
]) / (6 * 10 ** 4)

jet_lengths = np.array([
    8.449,
    8.934,
    9.878,
    10.818,
    12.536,
    14.409,
    15.939,
    17.645,
    18.490,
    22.043,
    23.004,
    24.902,
    26.625,
    27.011,
    27.845,
    29.087,
    31.054,
    31.130,
    31.982,
    32.226,
    33.400,
    32.949
]) / 1000


h_over_D = jet_lengths / D
u = flow_rates / (rho * A)
We = (rho * u ** 2 * D) / (sigma)


fig = plt.figure()
ax = fig.add_axes([0.15, 0.1, 0.8, 0.8])
ax.plot(We, h_over_D, marker="o", markersize = None, ls="None", color = "red")
ax.set_title("Jet Breakup")
ax.set_xlabel("$We$")
ax.set_ylabel(r"$\frac{h_0}{D}$")


plt.show()
fig.savefig("./Plots\Jet_lengths_03-06-2026")