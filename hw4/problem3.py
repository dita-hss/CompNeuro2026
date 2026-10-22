import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtr

N = 500
ALPHAS = np.linspace(0.02, 0.62, 16)
CUES = 10
MAX_SWEEPS = 60
rng = np.random.default_rng(42)


def recall(W, state):
    # asynchronous updates, field is adjusted in place when a bit flips,
    # which is the same as recomputing W @ state after that flip
    field = W @ state
    for _ in range(MAX_SWEEPS):
        flips = 0
        for i in rng.permutation(N):
            updated = 1.0 if field[i] >= 0 else -1.0
            if updated == state[i]:
                continue
            field += W[:, i] * (updated - state[i])
            state[i] = updated
            flips += 1
        if flips == 0:
            break
    return state


rows = []
for alpha in ALPHAS:
    P = int(round(alpha * N))
    X = rng.choice((-1.0, 1.0), size=(N, P))
    W = X @ X.T / N
    np.fill_diagonal(W, 0)

    unstable = np.mean(X * (W @ X) < 0)
    overlaps = []
    for _ in range(CUES):
        target = X[:, int(rng.integers(P))].copy()
        cue = target.copy()
        cue[rng.choice(N, size=N // 10, replace=False)] *= -1
        final = recall(W, cue)
        overlaps.append(np.dot(final, target) / N)
    row = dict(alpha=alpha, unstable=unstable, overlap=np.mean(overlaps), spread=np.std(overlaps, ddof=1))
    rows.append(row)
    print(
        f"alpha={alpha:.2f}, P={P:3d}, unstable={unstable:.4f}, "
        f"prediction={ndtr(-1 / np.sqrt(alpha)):.4f}, recall m={row['overlap']:.4f}",
        flush=True,
    )

alphas = np.array([r["alpha"] for r in rows])
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
axes[0].plot(alphas, [r["unstable"] for r in rows], "o-", label="Simulation")
axes[0].plot(alphas, ndtr(-1 / np.sqrt(alphas)), "--", label=r"Theory $\Phi(-1/\sqrt{\alpha})$")
axes[0].set(xlabel=r"Loading $\alpha=P/N$", ylabel="Fraction unstable bits", title="Pattern stability")
axes[0].legend()
axes[1].errorbar(alphas, [r["overlap"] for r in rows], yerr=[r["spread"] for r in rows], fmt="o-", capsize=2)
axes[1].set(xlabel=r"Loading $\alpha=P/N$", ylabel="Mean final overlap", title="Recall from 10% corrupted cues", ylim=(-0.05, 1.05))
fig.savefig("problem3c.png", dpi=220)
plt.close(fig)
