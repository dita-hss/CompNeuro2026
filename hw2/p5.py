import numpy as np
import matplotlib.pyplot as plt


mu = 1.0
theta = 1.0
dt = 0.001
T = 5000.0
transient_time = 100.0
sigma_squared_values = [0.25, 1.0, 4.0]

def simulate_noisy_integrator(mu, theta, sigma_squared, T, dt, seed=0):
    rng = np.random.default_rng(seed)
    num_steps = int(T / dt)
    voltage = np.zeros(num_steps)
    u = 0.0
    spike_times = []

    for step in range(num_steps):

        # Euler-Maruyama update
        random_normal = rng.standard_normal()
        u += mu * dt + np.sqrt(2.0 * sigma_squared * dt) * rng.standard_normal()

        if u >= theta:
            spike_times.append(step * dt)
            u = 0.0
        voltage[step] = u

    return voltage, np.array(spike_times)

def theoretical_density(u_values, mu, theta, sigma_squared):
    # theoretical firing rate from part (b)
    nu = mu / theta 
    density = np.zeros_like(u_values)
    below_reset = u_values < 0
    density[below_reset] = (nu / mu) * (1.0 - np.exp(-mu * theta / sigma_squared)) * np.exp(mu * u_values[below_reset] / sigma_squared)

    # 0 <= u <= theta
    between_reset_and_threshold = (u_values >= 0) & (u_values <= theta)
    density[between_reset_and_threshold] = (nu / mu) * (1.0 - np.exp(mu * (u_values[between_reset_and_threshold] - theta) / sigma_squared))
    return density

for sigma_squared in sigma_squared_values:

    print("\n" + "=" * 50)
    print(f"sigma^2 = {sigma_squared}")
    print("=" * 50)

    voltage, spike_times = simulate_noisy_integrator(mu=mu, theta=theta, sigma_squared=sigma_squared, T=T, dt=dt, seed=0)
    transient_steps = int(transient_time / dt)
    voltage_after_transient = voltage[transient_steps:]
    spike_times_after_transient = spike_times[spike_times >= transient_time]
    measurement_time = T - transient_time
    measured_rate = len(spike_times_after_transient) / measurement_time
    measured_fraction_below = np.mean(voltage_after_transient < 0)
    predicted_rate = mu / theta
    predicted_fraction_below = sigma_squared / (mu * theta) * (1.0 - np.exp(-mu * theta / sigma_squared))

    print(f"Measured firing rate:             {measured_rate:.4f}")
    print(f"Predicted firing rate:            {predicted_rate:.4f}")
    print(f"Measured fraction below reset:    {measured_fraction_below:.4f}")
    print(f"Predicted fraction below reset:   {predicted_fraction_below:.4f}")

    u_min = np.percentile(voltage_after_transient, 0.1)
    u_values = np.linspace(u_min, theta, 1000)
    theoretical_p = theoretical_density(u_values, mu, theta, sigma_squared)

    plt.figure(figsize=(7, 5))
    plt.hist(voltage_after_transient, bins=100, density=True, alpha=0.6, label="Simulation")
    plt.plot(u_values, theoretical_p, linewidth=2, label="Theory")
    plt.xlabel("Voltage u")
    plt.ylabel("Probability density")
    plt.title(rf"Stationary density, $\sigma^2={sigma_squared}$")
    plt.legend()
    plt.tight_layout()
    plt.show()