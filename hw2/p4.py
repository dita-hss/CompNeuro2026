import numpy as np
import matplotlib.pyplot as plt

# Given 
tau = 0.020 # membrane time constant (seconds)
mean_V = 10.0 # desired mean voltage (mV)
nu_tau_values = [2, 20, 200]

# Simulation parameters
dt = 0.0001 # time step = 0.1 ms
T = 100.0 # simulate for 100 seconds

nt = int(T / dt)
rng = np.random.default_rng(1)
results = {}


for nu_tau in nu_tau_values:
    # nu*tau is known, so:
    nu = nu_tau / tau
    # mean V = nu*w*tau = 10 mV
    w = mean_V / nu_tau

    V = np.zeros(nt)
    for j in range(nt - 1):
        # number of Poisson events occurring during dt
        n_events = rng.poisson(nu * dt)
        # voltage leaks toward zero and every event causes a jump of size w
        V[j + 1] = (V[j] - (dt / tau) * V[j] + w * n_events)

    # throw away the first second so V has time to reach its stationary distribution
    burn_in = int(1.0 / dt)
    V_stationary = V[burn_in:]

    # mean, variance, and skewness
    measured_mean = np.mean(V_stationary)
    measured_variance = np.var(V_stationary)
    measured_std = np.std(V_stationary)
    measured_skewness = np.mean((V_stationary - measured_mean)**3) / measured_std**3

    theoretical_mean = nu * w * tau
    theoretical_variance = nu * w**2 * tau / 2
    theoretical_skewness = (2 * np.sqrt(2) / (3 * np.sqrt(nu_tau)))

    results[nu_tau] = {
        "V": V_stationary,
        "nu": nu,
        "w": w,
        "mean_measured": measured_mean,
        "mean_theory": theoretical_mean,
        "variance_measured": measured_variance,
        "variance_theory": theoretical_variance,
        "skewness_measured": measured_skewness,
        "skewness_theory": theoretical_skewness
    }

    # results 
    print(f"\nnu*tau = {nu_tau}")
    print(f"nu = {nu:.1f} Hz")
    print(f"w = {w:.3f} mV")
    print(
        f"Mean:     measured = {measured_mean:.4f}, "
        f"theory = {theoretical_mean:.4f}"
    )
    print(
        f"Variance: measured = {measured_variance:.4f}, "
        f"theory = {theoretical_variance:.4f}"
    )
    print(
        f"Skewness: measured = {measured_skewness:.4f}, "
        f"theory = {theoretical_skewness:.4f}"
    )



#### histograms
for nu_tau in [2, 200]:
    V = results[nu_tau]["V"]
    theoretical_mean = results[nu_tau]["mean_theory"]
    theoretical_variance = results[nu_tau]["variance_theory"]
    theoretical_std = np.sqrt(theoretical_variance)

    plt.figure(figsize=(7, 5))
    plt.hist( V, bins=80, density=True, alpha=0.7, label="Simulation")

    x = np.linspace(V.min(), V.max(), 500)
    gaussian = 1 / (theoretical_std * np.sqrt(2 * np.pi)) * np.exp(-(x - theoretical_mean)**2 / (2 * theoretical_std**2))

    plt.plot(x, gaussian, linewidth=2, label="Matched Gaussian")
    plt.xlabel("Voltage V (mV)")
    plt.ylabel("Probability Density")
    plt.title("Shot Noise Distribution")
    plt.legend()
    plt.show()