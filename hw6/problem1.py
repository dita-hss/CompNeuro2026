
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

p_H = 0.5
rho = 0.5
tau = 5
n_visits = 400000

def R(T): # overall food intake rate
    food = p_H * (1 - np.exp(-rho * T))
    time = (
        p_H * (1 - np.exp(-rho * T)) / rho
        + (1 - p_H) * T
        + tau
    )
    return food / time

def lambda_tilde(T): # expected current food-finding rate
    return (rho * p_H * np.exp(-rho * T)) / (
        p_H * np.exp(-rho * T) + 1 - p_H
    )

R_known = p_H / (p_H / rho + tau) # intake rate if patch type is known

def equation(T):
    return R(T) - lambda_tilde(T)

T_star = brentq(equation, 0, 20)
R_star = R(T_star)

print("Optimal values:")
print(f"T* = {T_star:.6f}")
print(f"R* = {R_star:.6f}")
print(f"R_known = {R_known:.6f}")


def simulate(T, rng): # simulate 400,000 patch visits
    high = rng.random(n_visits) < p_H
    discovery = rng.exponential(1 / rho, n_visits)
    found = high & (discovery <= T)
    total_food = np.sum(found)
    search_time = np.where(found, discovery, T)
    total_time = np.sum(search_time + tau)

    return total_food / total_time

rng = np.random.default_rng(42)

print(f"{'T':<12}{'Simulated R':<16}{'Theoretical R'}")
for T in [2, T_star, 10]:
    simulated = simulate(T, rng)
    theoretical = R(T)
    print(f"{T:<12.4f}{simulated:<16.6f}{theoretical:.6f}")


# ------------------------------------------------------------------------------------------------

T_values = np.linspace(0, 15, 500) # generate waiting times

plt.figure(figsize=(10, 6)) # plot overall food intake rate
plt.plot(T_values, R(T_values), label="R(T): Overall intake rate", linewidth=2)
plt.plot(T_values, lambda_tilde(T_values), label="Expected current rate", linewidth=2)
plt.axhline(R_known, color="green", linestyle="--", label="R_known")
plt.scatter(T_star, R_star, color="red", s=80, zorder=5, label="Optimal point")
plt.axvline(T_star, color="red", linestyle=":", alpha=0.6)
plt.annotate(f"T* = {T_star:.3f}\nR* = {R_star:.4f}", xy=(T_star, R_star), xytext=(T_star + 1.5, R_star + 0.025), arrowprops=dict(arrowstyle="->", color="red"), fontsize=11)
plt.xlabel("Waiting time T")
plt.ylabel("Food intake rate")
plt.title("Optimal Patch-Leaving Time")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.show()
