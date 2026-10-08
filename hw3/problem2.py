import numpy as np
import matplotlib.pyplot as plt

tau = 1.0
v_reset = 0.0
v_threshold = 1.0
I = 1.5
delta = 1e-3

# natural period
Delta = tau * np.log((I * tau - v_reset) / (I * tau - v_threshold))
print(f"Natural period Delta = {Delta:.6f}")

# unperturbed LIF trajectory
def voltage(t):
    return ( I * tau - (I * tau - v_reset) * np.exp(-t / tau) )

# prc
phis = np.linspace(0.01, 0.99, 100)
R_numeric = np.zeros_like(phis)

for i, phi in enumerate(phis):
    t_kick = phi * Delta # time at which the perturbation is applied
    v_before = voltage(t_kick) # voltage immediately before perturbation
    v_after = v_before + delta
    remaining_time = tau * np.log((I * tau - v_after) / (I * tau - v_threshold)) # time remaining after the perturbation until threshold is reached
    T_prime = t_kick + remaining_time # perturbed next-spike time
    R_numeric[i] = (Delta - T_prime) / (Delta * delta) # numerical prc

R_analytic = tau * np.exp(phis * Delta / tau) / (Delta * (I * tau - v_reset)) # analytical prc

# error between numerical and analytical results
error = np.abs(R_numeric - R_analytic)
print(f"Maximum |R_numeric - R_analytic| = {np.max(error):.6f}")
print(f"Minimum numerical R = {np.min(R_numeric):.6f}")
print(f"Maximum numerical R = {np.max(R_numeric):.6f}")

# plots of prcs
plt.plot(phis, R_analytic, linewidth=2, label="Formula from part (a)")
plt.scatter(phis, R_numeric, s=18, label="Numerical perturbation")
plt.xlabel(r"Phase $\phi$")
plt.ylabel(r"$R(\phi)$")
plt.title("LIF Phase Response Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()