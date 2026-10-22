import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.linalg import solve_banded
from scipy.special import expit, ndtr

# ws = wc = (1 + lambda) / 2 and start at the same rate, x = r1 - r2,dx = (lambda * x + mu) dt + sigma * sqrt(2) dW
MU, SIGMA = 0.2, 0.5
DT, TRIALS = 0.002, 10_000
THETAS = np.linspace(0.5, 4.0, 8)
LAMBDAS = (0.0, 0.2)
SEED = 20261021
MAX_TIME = 200.0


def run_trials(lam, theta, rng):
    x = np.zeros(TRIALS)
    done = np.zeros(TRIALS, dtype=bool)
    correct = np.zeros(TRIALS, dtype=bool)
    times = np.full(TRIALS, np.nan)
    scale = SIGMA * np.sqrt(2 * DT)
    for step in range(int(MAX_TIME / DT)):
        active = ~done
        if not active.any():
            break
        old = x[active]
        new = old + DT * (lam * old + MU) + scale * rng.standard_normal(old.size)
        x[active] = new
        hit = np.abs(new) >= theta
        if not hit.any():
            continue
        who = np.flatnonzero(active)[hit]
        step_size = np.maximum(np.abs(new[hit] - old[hit]), 1e-12)
        fraction = np.clip((theta - np.abs(old[hit])) / step_size, 0, 1)
        correct[who] = new[hit] > 0
        times[who] = (step + fraction) * DT
        done[who] = True
    return correct.mean(), np.mean(times)


def theory_accuracy(theta, lam):
    if lam == 0:
        return expit(MU * theta / SIGMA**2)
    # mass on (-theta, 0) is P(hit +theta first)
    weight = lambda x: np.exp(-(lam * x * x + 2 * MU * x) / (2 * SIGMA**2))
    correct, _ = quad(weight, -theta, 0)
    total, _ = quad(weight, -theta, theta)
    return correct / total


def theory_time(theta, lam):
    if lam == 0:
        return (theta / MU) * np.tanh(MU * theta / (2 * SIGMA**2))
    # sigma^2 T'' + (lambda * x + mu) T' = -1, with T(-theta) = T(theta) = 0
    n = 3001
    h = 2 * theta / (n + 1)
    x = np.linspace(-theta + h, theta - h, n)
    drift = lam * x + MU
    sub = SIGMA**2 / h**2 - drift[1:] / (2 * h)
    sup = SIGMA**2 / h**2 + drift[:-1] / (2 * h)
    ab = np.zeros((3, n))
    ab[0, 1:] = sup
    ab[1] = -2 * SIGMA**2 / h**2
    ab[2, :-1] = sub
    t = solve_banded((1, 1), ab, -np.ones(n))
    return np.interp(0.0, x, t)


rng = np.random.default_rng(SEED)
records = []
for lam in LAMBDAS:
    for theta in THETAS:
        accuracy, mean_time = run_trials(lam, theta, rng)
        row = dict(
            lam=lam,
            theta=theta,
            accuracy=accuracy,
            mean_time=mean_time,
            theory_accuracy=theory_accuracy(theta, lam),
            theory_time=theory_time(theta, lam),
        )
        records.append(row)
        print(
            f"lambda={lam:.1f}, theta={theta:.1f}: "
            f"accuracy={accuracy:.4f} (theory {row['theory_accuracy']:.4f}), "
            f"mean time={mean_time:.3f} (theory {row['theory_time']:.3f})",
            flush=True,
        )

with open("results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
for lam in LAMBDAS:
    rows = [r for r in records if r["lam"] == lam]
    xs = [r["theta"] for r in rows]
    axes[0].plot(xs, [r["accuracy"] for r in rows], "o-", label=f"Simulation λ={lam:g}")
    axes[0].plot(xs, [r["theory_accuracy"] for r in rows], "--", label=f"Theory λ={lam:g}")
    axes[1].plot(xs, [r["mean_time"] for r in rows], "o-", label=f"Simulation λ={lam:g}")
    axes[1].plot(xs, [r["theory_time"] for r in rows], "--", label=f"Theory λ={lam:g}")
axes[0].set(xlabel="Decision threshold θ", ylabel="P(correct)", title="Decision accuracy", ylim=(0.45, 1.02))
axes[1].set(xlabel="Decision threshold θ", ylabel="Mean decision time", title="Decision speed")
for ax in axes:
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
fig.savefig("problem2d.png", dpi=300)
plt.close(fig)

result = next(r for r in records if r["lam"] == 0.2 and np.isclose(r["theta"], 4))
p = result["accuracy"]
matched_theta = (SIGMA**2 / MU) * np.log(p / (1 - p))
print(f"\nlambda=0.2, theta=4: P(correct)={p:.4f}, mean time={result['mean_time']:.3f}")
print(f"lambda=0 threshold with the same accuracy: theta={matched_theta:.3f}")
print(f"perfect integrator mean time at that threshold: {theory_time(matched_theta, 0):.3f}")
print(f"positive-feedback accuracy ceiling: {ndtr(MU / (SIGMA * np.sqrt(0.2))):.4f}")