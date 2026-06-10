import numpy as np
import matplotlib.pyplot as plt
import math


# 03-06-2026:

rho_03 = 1000
sigma_03 = 0.072
D_03 = 231.6 * 10 ** (-6)
A_03 = math.pi * D_03 ** 2 / 4

flow_rates_03 = np.array([
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

jet_lengths_03 = np.array([
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


h_over_D_03 = jet_lengths_03 / D_03
u_03 = flow_rates_03 / (rho_03 * A_03)
We_03 = (rho_03 * u_03 ** 2 * D_03) / (sigma_03)


# 04-06-2026:

rho_04 = 990
sigma_04 = 0.071
D_04 = 231.6 * 10 ** (-6)
A_04 = math.pi * D_04 ** 2 / 4

flow_rates_04 = np.array([
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

jet_lengths_04 = np.array([
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


h_over_D_04 = jet_lengths_04 / D_04
u_04 = flow_rates_04 / (rho_04 * A_04)
We_04 = (rho_04 * u_04 ** 2 * D_04) / (sigma_04)


We = np.concatenate((We_03, We_04))
h_over_D = np.concatenate((h_over_D_03, h_over_D_04))



fig = plt.figure()
ax = fig.add_axes([0.15, 0.1, 0.8, 0.8])
ax.plot(We_03, h_over_D_03, marker="o", markersize = None, ls="None", color = "red")
ax.plot(We_04, h_over_D_04, marker="^", markersize = None, ls="None", color = "green")
ax.set_title("Jet Breakup")
ax.set_xlabel("$We$")
ax.set_ylabel(r"$\frac{h_0}{D}$")
ax.legend(("Water; $D = 231.6\mu m$", "80% water & 20% ethanol; $D = 231.6\mu m$"))


plt.show()

fig.savefig("./Plots\Jet_lengths")