
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

eta = 4.0
tau_E = 5.0
w_EI = 10.0
w_IE = 10.0
E_eq = 0.1
I_eq = 0.1
epsilon = 0.02
tau_values = [2, 3, 5, 7.5, 10, 15, 20]

# pb
B = (eta**2 * w_EI * w_IE * E_eq * (1 - E_eq) * I_eq * (1 - I_eq)) / tau_E

def sigmoid(x):
    return 1 / (1 + np.exp(-eta * x))

def equations(t, y, w_EE, p_E, p_I, tau_I):
    E, I = y
    dE = (-E + sigmoid(w_EE * E - w_EI * I + p_E)) / tau_E
    dI = (-I + sigmoid(w_IE * E + p_I)) / tau_I
    return [dE, dI]

def simulate(tau_I):
    w_hopf = (1 + tau_E / tau_I) / (eta * E_eq * (1 - E_eq))
    w_EE = (1 + epsilon) * w_hopf
    logit = np.log(E_eq / (1 - E_eq)) / eta
    p_E = logit - w_EE * E_eq + w_EI * I_eq
    p_I = logit - w_IE * E_eq
    total_time = 20000  # ms

    sol = solve_ivp(equations, (0, total_time), [0.101, 0.1], args=(w_EE, p_E, p_I, tau_I), rtol=1e-9, atol=1e-11, max_step=1.5, dense_output=True)
    t = np.arange(total_time - 1200, total_time, 0.1)
    E, I = sol.sol(t)
    peaks, _ = find_peaks(E, prominence=0.001)
    periods = np.diff(t[peaks])
    period = np.mean(periods)
    frequency = 1000 / period
    return frequency

simulated_frequencies = []
theoretical_frequencies = []

print(f"{'tau_I (ms)':<12} {'Simulated (Hz)':<18} {'Theoretical (Hz)':<18}")
print("-" * 48)

for tau_I in tau_values:
    # Simulated frequency
    f_sim = simulate(tau_I)
    # Theoretical frequency
    omega = np.sqrt(B / tau_I - 1 / tau_I**2)
    f_theory = 1000 * omega / (2 * np.pi)
    simulated_frequencies.append(f_sim)
    theoretical_frequencies.append(f_theory)
    print(f"{tau_I:<12.1f} {f_sim:<18.2f} {f_theory:<18.2f}")

plt.figure(figsize=(8, 5))
plt.plot(tau_values, simulated_frequencies, "o-", label="Simulated frequency")
plt.plot(tau_values, theoretical_frequencies, "s--", label="Theoretical frequency")
plt.xlabel("Inhibitory time constant (ms)")
plt.ylabel("Frequency (Hz)")
plt.title("Simulated vs. Theoretical Oscillation Frequency")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
