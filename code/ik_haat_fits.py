import numpy as np
import matplotlib.pyplot as plt
import os
import math
from scipy.odr import ODR, Model, RealData


# the path to the file with all the data:
path = r"C:\Users\bosel\OneDrive\Documenten\GitHub\Breakuplength-Jet\resultaten\08-06-2026\all_data.txt"


def get_data(path):
    flow_rates = []
    mean_jet_lengths = []
    mean_jet_lengths_err = []

    with open(path, "r") as f:
        for line in f:
            lijst = line.split()
            if lijst == []:
                continue
            if lijst[0] == "flow":
                flow_rates.append(float(lijst[-1]))
            elif lijst[0] == "mean":
                mean_jet_lengths.append(float(lijst[-1]))
            elif lijst[0] == "standard":
                mean_jet_lengths_err.append(float(lijst[-1]))
            elif lijst[0] == "Density":
                density = float(lijst[-1])
            elif lijst[0] == "Surface":
                surface_tension = float(lijst[-1]) / 1000
            elif lijst[0] == "Viscosity":
                viscosity = float(lijst[-1]) / 1000
            elif lijst[0] == "Needle":
                diameter = float(lijst[-1]) / (10 ** 6)

    flow_rates = np.array(flow_rates) / (6 * 10 ** 4)
    mean_jet_lengths = np.array(mean_jet_lengths) / 1000
    mean_jet_lengths_err = np.array(mean_jet_lengths_err) / 1000
    flow_rates_err = flow_rates * 0.5 / 60

    L_over_D = mean_jet_lengths / diameter
    L_over_D_err = mean_jet_lengths_err / diameter
    area = math.pi * diameter ** 2 / 4
    u = flow_rates / (density * area)
    u_err = flow_rates_err / (density * area)
    We = (density * u ** 2 * diameter) / surface_tension
    We_err = 2 * We * u_err / u
    Re = (density * u * diameter) / viscosity
    Re_err = (density * u_err * diameter) / viscosity
    log_We = np.log(We)
    log_we_err = We_err / We
    log_Re = np.log(Re)
    log_Re_err = Re_err / Re
    log_L_over_D = np.log(L_over_D)
    log_L_over_D_err = L_over_D_err / L_over_D
    log_u = np.log(u)
    log_u_err = u_err / u
    log_L = np.log(mean_jet_lengths)
    log_L_err = mean_jet_lengths_err / mean_jet_lengths


    # fit voor het Webergetal
    def f(B, x):
        return B[0] + B[1] * x
    model = Model(f)
    data = RealData(log_We, log_L_over_D, sx=log_we_err, sy=log_L_over_D_err)
    odr = ODR(data, model, beta0=[-1, 0.5])
    out = odr.run()

    N_We = len(log_We)
    p_We = len(out.beta)
    chi2_red_We = out.res_var * N_We / (N_We - p_We)

    C_We = math.exp(out.beta[0])
    C_We_err = C_We * out.sd_beta[0]
    n_We = out.beta[1]
    n_We_err = out.sd_beta[1]

    # fit voor het Renolds getal:
    def f(B, x):
        return B[0] + B[1] * x
    model = Model(f)
    data = RealData(log_Re, log_L_over_D, sx=log_Re_err, sy=log_L_over_D_err)
    odr = ODR(data, model, beta0=[-1, 0.5])
    out = odr.run()

    N_Re = len(log_Re)
    p_Re = len(out.beta)
    chi2_red_Re = out.res_var * N_Re / (N_Re - p_Re)

    C_Re = math.exp(out.beta[0])
    C_Re_err = C_Re * out.sd_beta[0]
    n_Re = out.beta[1]
    n_Re_err = out.sd_beta[1]

    # fit voor de snelheid:
    def f(B, x):
        return B[0] + B[1] * x
    model = Model(f)
    data = RealData(log_u, log_L, sx=log_u_err, sy=log_L_err)
    odr = ODR(data, model, beta0=[-1, 0.5])
    out = odr.run()

    N_u = len(log_u)
    p_u = len(out.beta)
    chi2_red_u = out.res_var * N_u / (N_u - p_u)

    C_u = math.exp(out.beta[0])
    C_u_err = C_u * out.sd_beta[0]
    n_u = out.beta[1]
    n_u_err = out.sd_beta[1]

    return (
        We,
        Re,
        L_over_D,
        C_We,
        C_Re,
        n_We,
        n_Re,
        chi2_red_We,
        chi2_red_Re,
        mean_jet_lengths,
        mean_jet_lengths_err,
        diameter,
        We_err,
        mean_jet_lengths_err,
        Re_err,
        L_over_D_err,
        u,
        u_err,
        C_u,
        n_u,
        chi2_red_u
    )


class Data:
    def __init__(self, path, liquid):
        (
            self.We,
            self.Re,
            self.L_over_D,
            self.C_We,
            self.C_Re,
            self.n_We,
            self.n_Re,
            self.chi_We,
            self.chi_Re,
            self.L,
            self.L_err,
            self.D,
            self.We_err,
            self.L_err,
            self.Re_err,
            self.L_over_D_err,
            self.u,
            self.u_err,
            self.C_u,
            self.n_u,
            self.chi_u
        ) = get_data(path)
        self.liquid = liquid

data_03 = Data(
    r"C:\Users\bosel\OneDrive\Documenten\GitHub\Breakuplength-Jet\resultaten\03-06-2026\all_data.txt",
    "Distilled water"
)
data_05_and_12 = Data(
    r"C:\Users\bosel\OneDrive\Documenten\GitHub\Breakuplength-Jet\resultaten\05-06-2026_and_12-06-2026\all_data.txt",
    "Distilled water"
)
data_08 = Data(
    r"C:\Users\bosel\OneDrive\Documenten\GitHub\Breakuplength-Jet\resultaten\08-06-2026\all_data.txt",
    "80% water, 20% ethanol"
)
data_09 = Data(
    r"C:\Users\bosel\OneDrive\Documenten\GitHub\Breakuplength-Jet\resultaten\09-06-2026\all_data.txt",
    "80% water, 20% ethanol"
)
data_15 = Data(
    r"C:\Users\bosel\OneDrive\Documenten\GitHub\Breakuplength-Jet\resultaten\15-06-2026(final)\all_data.txt",
    "80% water, 20% glycerol"
)
data_16 = Data(
    r"C:\Users\bosel\OneDrive\Documenten\GitHub\Breakuplength-Jet\resultaten\16-06-2026\all_data.txt",
    "80% water, 20% glycerol"
)

plot_We = [
    data_03,
    data_05_and_12,
    data_08,
    data_09
]
plot_Re = [
    data_03,
    data_05_and_12,
    data_15,
    data_16
]
plot = [
    data_03,
    data_05_and_12,
    data_08,
    data_09,
    data_15,
    data_16
]
plot_small = [
    data_03,
    data_09,
    data_16
]
plot_big = [
    data_05_and_12,
    data_08,
    data_15
]


"""
# plot small needle
fig = plt.figure()
ax = fig.add_axes([0.15, 0.1, 0.8, 0.8])
colors = [
    "red",
    "blue",
    "green"
]
shapes = ["o", "*", "^"]
for data, color, shape in zip(plot_small, colors, shapes):
    ax.plot(data.u, data.L * 1000, marker = shape, color = color, ls = "None", label = f"{data.liquid}; $D = {data.D * 10 ** 6} \mu m$")
    x = np.linspace(0, 5, 1000)
    y = data.C_u * x ** data.n_u * 1000
    ax.plot(x, y, color = color, label = "_nolegend_")
ax.legend()
ax.set_xlabel("u (m/s)")
ax.set_ylabel("L (mm)")

plt.show()

fig.savefig("plot_small_needle")

# plot big needle
fig = plt.figure()
ax = fig.add_axes([0.15, 0.1, 0.8, 0.8])
colors = [
    "purple",
    "orange",
    "black"
]
shapes = ["o", "*", "^"]
for data, color, shape in zip(plot_big, colors, shapes):
    ax.plot(data.u, data.L * 1000, marker = shape, color = color, ls = "None", label = f"{data.liquid}; $D = {data.D * 10 ** 6}\mu m$")
    x = np.linspace(0, 2.5, 1000)
    y = data.C_u * x ** data.n_u * 1000
    ax.plot(x, y, color = color, label = "_nolabel_")
ax.legend()
ax.set_xlabel("u (m/s)")
ax.set_ylabel("L (mm)")

plt.show()

fig.savefig("plot_big_needle")
"""





"""
# speed against length, all
fig = plt.figure()
ax = fig.add_axes([0.15, 0.1, 0.8, 0.8])
colors = [
    "red",
    "blue",
    "green",
    "purple",
    "orange",
    "black"
]
shapes = ["o", "*", "^", "x", "D", "+"]
for data, color, shape in zip(plot, colors, shapes):
    ax.plot(data.u, data.L, marker = shape, color = color, ls = "None", label = f"{data.liquid}; D = {data.D}")
ax.legend()
ax.set_xlabel("u")
ax.set_ylabel("L")

plt.show()

fig.savefig("final_plot(I hope)")
"""






# Renold plot met water en glycerol
fig = plt.figure()
ax = fig.add_axes([0.15, 0.1, 0.8, 0.8])
colors = ["red", "blue", "green", "purple"]
shapes = ["o", "*", "^", "x"]
for data, color, shape in zip(plot_Re, colors, shapes):
    ax.errorbar(data.Re, data.L * 1000, xerr = data.Re_err, yerr = data.L_err * 1000, marker = shape, color = color, ls = "None")
    x = np.linspace(0, 1200, 10000)
    y = data.C_Re * x ** data.n_Re * data.D * 1000
    ax.plot(x, y, color = color, label = "_nolegend_")

ax.legend((
        "water; $D = 231.6\mu m$",
        "water; $D = 460\mu m$",
        "80% water, 20% glycerol; $D = 460\mu m$",
        "80% water, 20% glycerol; $D = 231.6\mu m$"
    ))
ax.set_xlabel("$Re$")
ax.set_ylabel("$L (mm)$")

plt.show()

fig.savefig("fits_Re")




"""
# weber plot met water en alcohol
fig = plt.figure()
ax = fig.add_axes([0.15, 0.1, 0.8, 0.8])
colors = ["red", "blue", "green", "purple"]
shapes = ["o", "*", "^", "x"]
for data, color, shape in zip(plot_We, colors, shapes):
    ax.plot(data.We, data.L * 1000, marker = shape, color = color, ls = "None")
    x = np.linspace(0, 60, 1000)
    y = data.C_We * x ** data.n_We * data.D * 1000
    ax.plot(x, y, color = color, label = "_nolegend_")

ax.legend((
        "water; $D = 231.6\mu m$",
        "water; $D = 460\mu m$",
        "80% water, 20% ethanol; $D = 460\mu m$",
        "80% water, 20% ethanol; $D = 231.6\mu m$"
    ))
ax.set_xlabel("$We$")
ax.set_ylabel("$L (mm)$")

plt.show()

fig.savefig("fits_We")
"""





"""
print(f"chi_We 03 = {data_03.chi_We}")
print(f"chi_Re 03 = {data_03.chi_Re}")
print(f"chi_We 05 and 12 = {data_05_and_12.chi_We}")
print(f"chi_Re 05 and 12 = {data_05_and_12.chi_Re}")
print(f"chi_We 08 = {data_08.chi_We}")
print(f"chi_Re 08 = {data_08.chi_Re}")
print(f"chi_We 09 = {data_09.chi_We}")
print(f"chi_Re 09 = {data_09.chi_Re}")
print(f"chi_We 15 = {data_15.chi_We}")
print(f"chi_Re 15 = {data_15.chi_Re}")
print(f"chi_We 16 = {data_16.chi_We}")
print(f"chi_Re 16 = {data_16.chi_Re}")
"""

"""
fig = plt.figure()
ax = fig.add_axes([0.15, 0.1, 0.8, 0.8])
ax.errorbar(We, L_over_D, xerr = We_err, yerr = L_over_D_err, marker = "o", markersize = None, ls = "None", color = "purple")
ax.set_title("Jet Breakup")
ax.set_xlabel("$We$")
ax.set_ylabel(r"$\frac{h_0}{D}$")
plt.show()
"""