import numpy as np
import matplotlib.pyplot as plt

a = np.linspace(0.001, 0.999, 1000)
D = 1

# active front speed
c_active = np.sqrt(D) * (1 - 2*a) / np.sqrt(a * (1 - a))
# passive speed
c_passive = 2 * np.sqrt(D)
# active/passive crossing
a_cross = (2 - np.sqrt(2)) / 4
# stall threshold
a_stall = 0.5

plt.figure(figsize=(8, 6))
plt.plot(a, c_active, label="Active front speed")
plt.axhline(c_passive, linestyle="--", label="Passive apparent speed")
plt.scatter(a_cross, c_passive, label=f"Crossing: a = {a_cross:.3f}")
plt.scatter(a_stall,0,label="Stall: a = 0.5")
plt.axhline(0, linewidth=0.8)
plt.xlabel("Excitation threshold a")
plt.ylabel("Propagation speed c")
plt.title("Propagation Speed vs. Excitation Threshold (at D = 1)")
plt.ylim(-6, 6)
plt.grid(alpha=0.3)
plt.legend()
plt.show()