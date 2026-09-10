import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ohms, milliseconds, milliseconds, millivolts, millivolts
R = 1.0 
tau_m = 10.0
tau_s = 2.0
u_rest = 0.0
u_th = 1.0


# Part b, theoretical values
r = tau_s / tau_m
t_peak = (tau_m * tau_s / (tau_s - tau_m)) * np.log(tau_s / tau_m)
I_crit = ((u_th - u_rest) / R) * r**(1 / (r - 1))
print(f"\n")
print(f"\n")
print(f"t_peak = {t_peak:.4f} ms")
print(f"I_crit = {I_crit:.4f} mA")


# LIF differential equation

def lif(t, u, I0):
    I_t = I0 * np.exp(-t / tau_s)
    du_dt = (-(u[0] - u_rest) + R * I_t) / tau_m
    return [du_dt]


# Test currents below, at, and above I_crit
I_values = [
    0.8 * I_crit,
    I_crit,
    1.2 * I_crit] # mA
labels = ["Below", "At", "Above"]
t_eval = np.linspace(0, 50, 5000)
results = []


# integrate for each current
plt.figure(figsize=(8, 5))

for I0, label in zip(I_values, labels):
    solution = solve_ivp(
        lif,
        [0, 50],
        [u_rest],
        args=(I0,),
        t_eval=t_eval,
        rtol=1e-9,
        atol=1e-11
    )

    t = solution.t
    u = solution.y[0]

    # Find numerical peak
    peak_index = np.argmax(u)

    numerical_peak = u[peak_index]
    numerical_t_peak = t[peak_index]

    results.append([
        label,
        I0,
        numerical_t_peak,
        numerical_peak
    ])

    plt.plot(t, u, label=f"{label}: I0 = {I0:.4f}")


# threshold
plt.axhline(u_th, linestyle="--", label="Threshold")
plt.xlabel("Time (ms)")
plt.ylabel("Voltage u(t) (mV)")
plt.title("LIF neuron driven by synaptic transient (P4 / HW1)")
plt.legend()
plt.grid()
plt.show()


# Table
print("\nNumerical results")
print("-" * 58)
print(f"{'Current':<10} {'I0 (mA)':<12} {'t_peak (ms)':<15} {'u_peak (mV)':<12}")

for row in results:
    label, I0, tp, up = row

    print(
        f"{label:<10} "
        f"{I0:<12.4f} "
        f"{tp:<15.4f} "
        f"{up:<12.4f}"
    )


# Plot I_crit versus tau_s
tau_s_values = np.logspace(
    np.log10(0.5),
    np.log10(100),
    500
)

r_values = tau_s_values / tau_m


# Need to handle r = 1 separately because
# r^(1/(r-1)) has a removable singularity there.
# Its limit as r -> 1 is e

factor = np.empty_like(r_values)
near_one = np.isclose(r_values, 1.0)
factor[near_one] = np.e
factor[~near_one] = np.exp(np.log(r_values[~near_one]) / (r_values[~near_one] - 1))
I_crit_values = ((u_th - u_rest) / R) * factor

plt.figure(figsize=(8, 5))
plt.loglog(tau_s_values, I_crit_values)
plt.xlabel(r"$\tau_s$ (ms)")
plt.ylabel(r"$I_0^{crit}$ (mA)")
plt.title(r"Critical current vs. synaptic time constant (P4 / HW1)")
plt.grid(True, which="both")
plt.show()
print(f"\n")
print(f"\n")