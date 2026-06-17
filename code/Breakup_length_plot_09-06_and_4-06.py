import numpy as np
import matplotlib.pyplot as plt
import math


# 09-06-2026:

rho_09 = 996
sigma_09 = 0.071
D_09 = 231.6 * 10 ** (-6)
A_09 = math.pi * D_09 ** 2 / 4

flow_rates_09 = np.array([1.9688, 2.9531, 3.9639, 4.9829, 5.4360, 5.8639, 6.3650, 6.8572, 7.0760, 7.3379, 7.5072]) / (6 * 10 ** 4)

jet_lengths_09 = np.array([4.309, 10.455, 14.919, 19.525, 20.7441, 24.250, 27.667, 16.353, 31.069, 28.826, 29.635]) / 1000


h_over_D_09 = jet_lengths_09 / D_09
u_09 = flow_rates_09 / (rho_09 * A_09)
We_09 = (rho_09 * u_09 ** 2 * D_09) / (sigma_09)


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


We = np.concatenate((We_09, We_04))
h_over_D = np.concatenate((h_over_D_09, h_over_D_04))



fig = plt.figure()
ax = fig.add_axes([0.15, 0.1, 0.8, 0.8])
ax.plot(We_09, h_over_D_09, marker="o", markersize = None, ls="None", color = "red")
ax.plot(We_04, h_over_D_04, marker="^", markersize = None, ls="None", color = "green")
ax.set_title("Jet Breakup $D = 231.6\mu m$")
ax.set_xlabel("$We$")
ax.set_ylabel(r"$\frac{h_0}{D}$")
ax.legend(("80% water & 20% ethanol(09-06-2026)", "80% water & 20% ethanol(04-06-2026)"))



plt.savefig(r"C:\Users\taran\Downloads\metingen_09-06-2026\combination_june.png")

plt.show()
