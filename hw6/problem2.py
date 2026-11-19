
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

eta = 0.02
x_star = 0.40 / 0.85
y_star = 0.45 / 0.85

def mean_flow(t, state):
    x, y = state
    dx = 0.85 * eta * (x * (1 - x))**2 * (y - y_star)
    dy = -0.85 * eta * (y * (1 - y))**2 * (x - x_star)

    return [dx, dy]

initial = [0.2, 0.3]
times = np.linspace(0, 400000, 4001)
solution = solve_ivp(mean_flow, [0, 400000], initial, t_eval=times, rtol=1e-10, atol=1e-12, max_step=100)
x = solution.y[0]
y = solution.y[1]

def F(s, a):
    return a / s + (1 - a) / (1 - s) + (1 - 2*a) * np.log(s) + (2*a - 1) * np.log1p(-s)

def K(x, y):
    return F(x, x_star) - F(x_star, x_star) + F(y, y_star) - F(y_star, y_star)

K_values = K(x, y)

print(f"x range: {x.min():.6f} to {x.max():.6f}")
print(f"y range: {y.min():.6f} to {y.max():.6f}")
print(f"K initial: {K_values[0]:.8f}")
print(f"K variation: {np.ptp(K_values):.10f}")

plt.figure(figsize=(7, 6))
plt.plot(x, y, label="Mean-flow orbit")
plt.scatter(x[0], y[0], color="green", s=70, label="Starting point", zorder=5)
plt.scatter(x_star, y_star, color="red", marker="x", s=100, label="Equilibrium", zorder=5)
plt.xlabel("x (Probability of shooting 3)")
plt.ylabel("y (Probability of guarding paint)")
plt.title("Mean-Flow Orbit Around Equilibrium")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.show()
