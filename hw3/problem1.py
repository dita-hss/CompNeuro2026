import numpy as np
import matplotlib.pyplot as plt

eps = 0.5
dt = 1e-3
T_transient = 100
T_measure = 200
T_total = T_transient + T_measure
I_values = np.linspace(0.9, 3.0, 500)

firing_number = []
for I in I_values:
    v = 0.0
    spikes = []
    t = 0.0

    while t < T_total:
        input_current = I + eps * np.sin(2 * np.pi * t)
        v += dt * (-v + input_current)
        t += dt

        # threshold crossing
        if v >= 1.0:
            spikes.append(t)
            v = 0.0

    spikes = np.array(spikes)
    measured_spikes = spikes[spikes >= T_transient]
    # spikes per unit time = spikes per forcing period
    N = len(measured_spikes) / T_measure
    firing_number.append(N)
firing_number = np.array(firing_number)


# unforced firing number
unforced = np.zeros_like(I_values)
mask = I_values > 1
unforced[mask] = 1 / np.log(I_values[mask] / (I_values[mask] - 1))


# part a -- 1:1 tongue boundaries
I_star = 1 / (1 - np.exp(-1))
lower = I_star - eps / np.sqrt(1 + 4 * np.pi**2)
upper = I_star + eps / np.sqrt(1 + 4 * np.pi**2)


# plots of firing number vs I
plt.figure(figsize=(9, 6))
plt.plot(I_values, firing_number, label="Forced LIF")
plt.plot(I_values, unforced, "--", label="Unforced")
plt.axvline(lower, linestyle=":", label="1:1 tongue boundaries")
plt.axvline(upper, linestyle=":", alpha=0.5)
plt.axhline(1, linestyle=":", alpha=0.5)
plt.xlabel("I")
plt.ylabel("Firing number (spikes per forcing period)")
plt.title(r"Periodically Forced LIF ($\epsilon=0.5$)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# part b -- spike phases at I = 1.6
I = 1.6
v = 0.0
spikes = []
t = 0.0

while t < T_total:
    input_current = I + eps * np.sin(2 * np.pi * t)
    v += dt * (-v + input_current)
    t += dt

    # threshold crossing
    if v >= 1.0:
        spikes.append(t)
        v = 0.0

spikes = np.array(spikes)
spikes = spikes[spikes >= T_transient]
phases = spikes % 1

print("1:1 tongue boundaries:")
print("lower =", lower)
print("upper =", upper)
print("\nLast 10 spike phases for I = 1.6:")
print(phases[-10:])
print("\nMean of last 10 phases:")
print(np.mean(phases[-10:]))