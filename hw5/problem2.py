import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.special import expit

tau_a = 120.0
beta = 1.0
g = 2.0
eta = 1000.0

def equations(t, y, I_L, I_R, w):
    uL, uR, aL, aR = y
    input_L = w*uL - beta*uR - g*aL + I_L
    input_R = w*uR - beta*uL - g*aR + I_R
    duL = -uL + expit(eta * input_L)
    duR = -uR + expit(eta * input_R)
    daL = (-aL + uL) / tau_a
    daR = (-aR + uR) / tau_a
    return [duL, duR, daL, daR]

def simulate(I_L=1.3, I_R=1.3, w=0.8):
    # integrate to t = 30*tau_a
    total_time = 30 * tau_a

    solution = solve_ivp(
        equations,
        (0, total_time),
        [1, 0, 0, 0],
        args=(I_L, I_R, w),
        method="Radau",
        rtol=1e-8,
        atol=1e-10,
        max_step=2,
        dense_output=True
    )

    t = np.arange(0, total_time, 0.1)
    uL, uR, aL, aR = solution.sol(t)
    difference = uL - uR
    indices = np.where(np.diff(np.signbit(difference)))[0]
    crossings = t[indices] - difference[indices] * (t[indices+1] - t[indices]) / (difference[indices+1] - difference[indices])
    durations = np.diff(crossings) # time between consecutive handovers
    dominant_left = difference[indices[:-1]+1] > 0 # identify which population dominates
    left_durations = durations[dominant_left][3:][-5:]
    right_durations = durations[~dominant_left][3:][-5:]
    T_L = np.mean(left_durations) # average duration of left dominance
    T_R = np.mean(right_durations) # average duration of right dominance
    return t, uL, uR, aL, aR, T_L, T_R


# symmetric case
t, uL, uR, aL, aR, TL, TR = simulate()

B = (1.3 - beta) / g # boundary between left and right dominance

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True) # create figure and axes
mask = t >= 2400 # show final portion after transients
axes[0].plot(t[mask], uL[mask], label="u_L")
axes[0].plot(t[mask], uR[mask], label="u_R")
axes[0].set_ylabel("Activity")
axes[0].set_title("Binocular Rivalry: Population Activity")
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[1].plot(t[mask], aL[mask], label="a_L")
axes[1].plot(t[mask], aR[mask], label="a_R")
axes[1].axhline(B, linestyle="--", color="gray", label="B = 0.15")
axes[1].axhline(1-B, linestyle=":", color="black", label="1-B = 0.85")
axes[1].set_xlabel("Time")
axes[1].set_ylabel("Adaptation")
axes[1].set_title("Adaptation Variables")
axes[1].legend()
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# asymmetric stimuli
print("I_L   Sim T_L   Theory T_L   Sim T_R   Theory T_R")
for I_L in [1.3, 1.45, 1.6]:
    t, uL, uR, aL, aR, TL, TR = simulate(I_L=I_L, I_R=1.3)

    B_L = (I_L - beta) / g
    B_R = (1.3 - beta) / g

    TL_theory = tau_a * np.log((1-B_L)/B_R)
    TR_theory = tau_a * np.log((1-B_R)/B_L)
    print(
        f"{I_L:.2f}   "
        f"{TL:.3f}    "
        f"{TL_theory:.3f}      "
        f"{TR:.3f}    "
        f"{TR_theory:.3f}"
    )

# w= 0.3
w = 0.3
I = 1.3
t, uL, uR, aL, aR, TL, TR = simulate(I_L=I, I_R=I, w=w)
T_sim = (TL + TR) / 2
T_theory = tau_a * np.log((w + I) / (g - w - I))
print(f"Simulated T_dom = {T_sim:.3f}")
print(f"Theoretical T_dom = {T_theory:.3f}")
print(f"Difference = {T_sim - T_theory:.3f}")