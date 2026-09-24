import numpy as np
import matplotlib.pyplot as plt

V0 = 16.13 # mV
EN = 0.0 # mV
g_bar = 1.0 # normalized maximum conductance
V = np.linspace(-90, 0, 1000) # v range from -90 to 0 mV

# Magnesium block function
def B(V, Mg):
    return 1 / (1 + (Mg / 3.57) * np.exp(-V / V0))

# NMDA current
def I_N(V, Mg):
    return g_bar * B(V, Mg) * (V - EN)

Mg_values = [0, 0.1, 1.0] # Mg2+ concentrations
for Mg in Mg_values:
    plt.plot(V, I_N(V, Mg), label=f"[Mg2+] = {Mg} mM")
plt.xlabel("Voltage V (mV)")
plt.ylabel("NMDA Current (normalized)")
plt.title("NMDA Current-Voltage Relationship")
plt.legend()
plt.grid()
plt.show()