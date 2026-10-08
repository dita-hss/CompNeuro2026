import numpy as np
import matplotlib.pyplot as plt

L = 200.0
phi0 = 0.4

def simulate(eps, T=300.0, dt=1e-5):
    v1 = -L
    theta2 = 1.0 - phi0
    v2 = -1.0 / np.tan(np.pi * theta2)

    spikes1 = []
    spikes2 = []
    n_steps = int(T / dt)

    for n in range(n_steps):
        t = n * dt
        dv1 = 1.0 + v1**2 + eps * (v2 - v1)
        dv2 = 1.0 + v2**2 + eps * (v1 - v2)
        v1 += dt * dv1
        v2 += dt * dv2

        if v1 >= L:
            spikes1.append(t)
            v1 = -L

        if v2 >= L:
            spikes2.append(t)
            v2 = -L
    return np.array(spikes1), np.array(spikes2)

def measure_lags(spikes1, spikes2):
    times = []
    phis = []

    for k in range(1, len(spikes1) - 1):
        t1 = spikes1[k]
        previous2 = spikes2[spikes2 < t1]
        if len(previous2) == 0:
            continue

        t2 = previous2[-1]
        period = spikes1[k + 1] - t1
        phi = (t2 - t1) / period
        phi = phi % 1.0
        times.append(t1)
        phis.append(phi)

    return np.array(times), np.array(phis)


eps = 0.02
spikes1, spikes2 = simulate(eps)
times, phis = measure_lags(spikes1, spikes2)

prediction = np.arctan(np.tan(np.pi * phi0) * np.exp(-2 * eps * times)) / np.pi # phase-model prediction

plt.plot(times, phis, "o", markersize=3, label="Voltage simulation")
plt.plot(times, prediction, label="Phase prediction")
plt.xlabel("Time")
plt.ylabel(r"Phase lag $\phi$")
plt.title("Gap-Junction Coupled QIF Neurons")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

eps_values = [0.005, 0.02, 0.05]

print("\nDecay-rate comparison")

for eps in eps_values:
    spikes1, spikes2 = simulate(eps)
    times, phis = measure_lags(spikes1, spikes2)
    mask = (phis > 1e-4) & (phis < 0.49)
    t_fit = times[mask]
    phi_fit = phis[mask]
    y = np.log(np.abs(np.tan(np.pi * phi_fit)))

    slope, intercept = np.polyfit(t_fit, y, 1)
    fitted_rate = -slope
    predicted_rate = 2 * eps

    print(
        f"eps = {eps:.3f}: "
        f"fitted rate = {fitted_rate:.5f}, "
        f"2*epsilon = {predicted_rate:.5f}"
    )