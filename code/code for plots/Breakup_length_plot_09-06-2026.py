import numpy as np
import matplotlib.pyplot as plt
import math


rho = 964 
sigma = 0.04370
D = 231.6 * 10 ** (-6)
A = math.pi * D ** 2 / 4

flow_rates = np.array([1.9688, 2.9531, 3.9639, 4.9829, 5.4360, 5.8639, 6.3650, 6.8572, 7.0760, 7.3379, 7.5072]) / (6 * 10 ** 4)

jet_lengths = np.array([4.309, 10.455, 14.919, 19.525, 20.7441, 24.250, 27.667, 16.353, 31.069, 28.826, 29.635]) / 1000


h_over_D = jet_lengths / D
u = flow_rates / (rho * A)
We = (rho * u ** 2 * D) / (sigma)


fig = plt.figure()
ax = fig.add_axes([0.15, 0.1, 0.8, 0.8])
ax.plot(We, h_over_D, marker="+", markersize = None, ls="None", color = "black")
ax.set_title("Jet Breakup")
ax.set_xlabel("$We$")
ax.set_ylabel(r"$\frac{h_0}{D}$")


plt.savefig(r"C:\Users\taran\Downloads\metingen_09-06-2026\Jet_lengths_09-06-2026.png")

plt.show()
