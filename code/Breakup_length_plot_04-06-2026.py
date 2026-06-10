import numpy as np
import matplotlib.pyplot as plt
import math


rho = 990
sigma = 0.071
D = 231.6 * 10 ** (-6)
A = math.pi * D ** 2 / 4

flow_rates = np.array([
    1.7258,
    1.8023,
    2.0373,
    2.0975,
    2.3512,
    2.5570,
    2.7606,
    2.9844,
    3.1925,
    3.4133,
    3.6219,
    3.7883,
    4.0064,
    4.2476,
    4.2546,
    4.4917,
    4.6243,
    4.7958,
    5.0142,
    5.1842,
    5.3997,
    5.41,
    5.59
]) / (6 * 10 ** 4)

jet_lengths = np.array([
    5.258,
    5.871,
    7.172,
    7.823,
    8.787,
    10.229,
    11.928,
    12.823,
    14.124,
    14.978,
    16.013,
    17.361,
    18.452,
    19.723,
    19.436,
    20.625,
    20.711,
    21.830,
    22.879,
    23.596,
    24.851,
    24.388,
    25.445
]) / 1000


h_over_D = jet_lengths / D
u = flow_rates / (rho * A)
We = (rho * u ** 2 * D) / (sigma)


fig = plt.figure()
ax = fig.add_axes([0.15, 0.1, 0.8, 0.8])
ax.plot(We, h_over_D, marker="^", markersize = None, ls="None", color = "green")
ax.set_title("Jet Breakup")
ax.set_xlabel("$We$")
ax.set_ylabel(r"$\frac{h_0}{D}$")


plt.show()
fig.savefig("./Plots\Jet_lengths_04-06-2026")