import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

eta, tau_E, tau_I = 4.0, 5.0, 10.0
w_EI = w_IE = 10.0
E_eq = I_eq = 0.1
w_hopf = (1 + tau_E / tau_I) / (eta * E_eq * (1 - E_eq))
logit_eq = np.log(0.1 / 0.9) / eta

def rhs(t, y, w_EE, p_E, p_I):
    E, I = y
    f_E = 1 / (1 + np.exp(-eta * (w_EE * E - w_EI * I + p_E)))
    f_I = 1 / (1 + np.exp(-eta * (w_IE * E + p_I)))
    return [(-E + f_E) / tau_E, (-I + f_I) / tau_I]

def simulate(epsilon, total_ms=20000, sample_ms=1200, dt=0.1):
    w_EE = (1 + epsilon) * w_hopf
    p_E = logit_eq - w_EE * E_eq + w_EI * I_eq
    p_I = logit_eq - w_IE * E_eq

    solution = solve_ivp(rhs, (0, total_ms), [0.101, 0.1], args=(w_EE, p_E, p_I), rtol=1e-9, atol=1e-11, max_step=1.5, dense_output=True)
    t = np.arange(total_ms - sample_ms, total_ms, dt)
    E, I = solution.sol(t)
    e_peaks, _ = find_peaks(E, prominence=0.001)
    i_peaks, _ = find_peaks(I, prominence=0.001)
    period = np.mean(np.diff(t[e_peaks]))
    delays = []
    for idx in e_peaks:
        later = i_peaks[(i_peaks > idx) & (t[i_peaks] < t[idx] + period)]
        if len(later):
            delays.append(t[later[0]] - t[idx])
    lag_deg = 360 * np.mean(delays) / period
    return t, E, I, period, np.ptp(E), lag_deg

epsilons = np.array([0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
results = [simulate(eps) for eps in epsilons]
amplitudes = np.array([r[4] for r in results])
slope, intercept = np.polyfit(np.log(epsilons), np.log(amplitudes), 1)

print("epsilon period (ms)  max(E)-min(E)  I lag (degrees)")
for eps, (_, _, _, period, amplitude, lag) in zip(epsilons, results):
    print(f"{eps:7.3f} {period:10.3f} {amplitude:13.5f} {lag:12.2f}")
print(f"\nLog-log fitted slope: {slope:.4f}")

plt.figure(figsize=(6.5, 4.5))
plt.loglog(epsilons, amplitudes, "o", label="Simulated amplitude")
plt.loglog(epsilons, np.exp(intercept) * epsilons**slope, label=f"Fit: slope = {slope:.3f}")
plt.xlabel("epsilon")
plt.ylabel("Amplitude: max(E) - min(E)")
plt.title("Wilson-Cowan oscillation amplitude near Hopf")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.show()
