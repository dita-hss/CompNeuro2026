
import numpy as np
import matplotlib.pyplot as plt

eta = 0.02
n_possessions = 400000
x_star = 0.40 / 0.85
y_star = 0.45 / 0.85
shot_probs = np.array([
    [0.40, 0.60],
    [0.40, 0.25]
])

def sigmoid(theta):
    return 1 / (1 + np.exp(-theta))

def simulate(tau, seed=42):
    rng = np.random.default_rng(seed)
    theta1 = 0.3
    theta2 = -0.3
    baseline1 = 0.0
    baseline2 = 0.0
    xs = np.zeros(n_possessions)
    ys = np.zeros(n_possessions)

    for t in range(n_possessions):
        x = sigmoid(theta1)
        y = sigmoid(theta2)
        xs[t] = x
        ys[t] = y

        # a1=1: shoot 3
        # a1=0: shoot 2
        # a2=1: guard paint
        # a2=0: guard arc
        a1 = int(rng.random() < x)
        a2 = int(rng.random() < y)

        col = 0 if a2 == 1 else 1
        prob_make = shot_probs[a1, col]
        made = rng.random() < prob_make
        points = (2 + a1) if made else 0
        reward1 = points
        reward2 = -points

        theta1 += eta * ((reward1 - baseline1) * (a1 - x) - tau * theta1 * x * (1 - x))
        theta2 += eta * ((reward2 - baseline2) * (a2 - y) - tau * theta2 * y * (1 - y))
        baseline1 += (reward1 - baseline1) / (t + 1)
        baseline2 = -baseline1

    return xs, ys


#run
x0, y0 = simulate(tau=0)
x1, y1 = simulate(tau=0.05)


def report(x, y, label):
    quarter = n_possessions // 4
    distance = np.sqrt((x - x_star)**2 + (y - y_star)**2)
    avg_x = np.mean(x[-quarter:])
    avg_y = np.mean(y[-quarter:])
    early = np.mean(distance[:quarter])
    late = np.mean(distance[-quarter:])

    print(f"\n{label}")
    print(f"Last-quarter mean x: {avg_x:.6f}")
    print(f"Last-quarter mean y: {avg_y:.6f}")
    print(f"Early mean distance: {early:.6f}")
    print(f"Late mean distance:  {late:.6f}")

report(x0, y0, "Tau = 0")
report(x1, y1, "Tau = 0.05")


#plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, x, y, title in [(axes[0], x0, y0, "Tau = 0"), (axes[1], x1, y1, "Tau = 0.05")]:
    ax.plot(x, y, linewidth=0.5, alpha=0.65)
    ax.scatter(x[0], y[0], color="green", s=65, label="Start", zorder=5)
    ax.scatter(x_star, y_star, color="red", marker="x", s=90, label="Equilibrium", zorder=5)
    ax.set_title(title)
    ax.set_xlabel("x (Probability of shooting 3)")
    ax.set_ylabel("y (Probability of guarding paint)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")

plt.suptitle("Policy-Gradient Learning: Entropy Comparison")
plt.tight_layout()

plt.show()
