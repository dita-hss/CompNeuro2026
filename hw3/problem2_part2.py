import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def fhn(t, y):
    v, w = y
    dv = v - v**3 / 3 - w + 0.5
    dw = 0.08 * (v + 0.7 - 0.8 * w)
    return [dv, dw]

def upward_crossing(t, y):
    return y[0]

upward_crossing.direction = 1
upward_crossing.terminal = False

# solve for limit cycle
sol = solve_ivp(fhn, [0, 1000], [-1.0, 1.0], events=upward_crossing, rtol=1e-10, atol=1e-12, max_step=0.02)
crossings = sol.t_events[0]
late_crossings = crossings[-10:]
periods = np.diff(late_crossings)
Delta = np.mean(periods)
print("Period Delta =", Delta)


# get a state at phase zero
t0 = crossings[-2]
# integrate again to obtain the state at that crossing
sol0 = solve_ivp(fhn, [0, t0], [-1.0, 1.0], dense_output=True, rtol=1e-10, atol=1e-12, max_step=0.02)
y0 = sol0.sol(t0)
delta = 1e-3
phis = np.linspace(0.02, 0.98, 60)

R_numeric = []
for phi in phis:    
    t_kick = phi * Delta
    before_kick = solve_ivp(fhn,[0, t_kick], y0, rtol=1e-10, atol=1e-12, max_step=0.02)
    y_kick = before_kick.y[:, -1].copy()
    y_kick[0] += delta
    after_kick = solve_ivp(fhn, [t_kick, t_kick + 6 * Delta], y_kick, events=upward_crossing, rtol=1e-10, atol=1e-12, max_step=0.02)
    perturbed_crossings = after_kick.t_events[0]
    target_cycle = 4
    T_perturbed = perturbed_crossings[target_cycle - 1]
    T_unperturbed = target_cycle * Delta

    R = (T_unperturbed - T_perturbed) / (Delta * delta) # numerical prc
    R_numeric.append(R)


R_numeric = np.array(R_numeric)

# results
print("Minimum R =", np.min(R_numeric))
print("Maximum R =", np.max(R_numeric))

negative_phases = phis[R_numeric < 0]

if len(negative_phases) > 0:
    print(f"R < 0 approximately for phi = {negative_phases[0]} to {negative_phases[-1]}")
else:
    print("No negative region found.")

# plots
plt.plot(phis, R_numeric, marker="o", markersize=3, label="Numerical perturbation")
plt.axhline(0, linewidth=1)
plt.xlabel(r"Phase $\phi$")
plt.ylabel(r"$R(\phi)$")
plt.title("FitzHugh-Nagumo Phase Response Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()