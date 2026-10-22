import numpy as np
import matplotlib.pyplot as plt

N = 200
dt = 0.02
steps = int(round(40.0 / dt))
sigma_e = 1.0
sigma_i_values = [2.0, 3.0]
n_trials = 20
rng = np.random.default_rng(42)

# shortest periodic distance between sites on the ring
sites = np.arange(N)
distance = np.minimum(sites, N - sites)

def kernel(sigma_i):
    return (np.exp(-distance**2 / (2 * sigma_e**2))
            - (sigma_e / sigma_i) * np.exp(-distance**2 / (2 * sigma_i**2)))

def simulate(K_fft, w):
    # w is one profile, shape (N,), or a batch, shape (n_trials, N)
    for _ in range(steps):
        conv = np.fft.ifft(K_fft * np.fft.fft(w, axis=-1), axis=-1).real
        w = np.clip(w + dt * conv, -1.0, 1.0)
    return w

def stripe_pairs(w):
    # sign changes around the ring, including the last site back to the first
    signs = np.where(w >= 0, 1, -1)
    flips = np.count_nonzero(signs != np.roll(signs, 1, axis=-1), axis=-1)
    return flips // 2

fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, constrained_layout=True)
for ax, sigma_i in zip(axes, sigma_i_values):
    K_fft = np.fft.fft(kernel(sigma_i))
    k_star = np.sqrt(4 * np.log(sigma_i / sigma_e) / (sigma_i**2 - sigma_e**2))
    m = int(np.rint(N * k_star / (2 * np.pi)))

    w = simulate(K_fft, rng.normal(0.0, 0.01, (n_trials, N)))
    pairs = stripe_pairs(w)
    periods = np.divide(N, pairs, where=pairs > 0, out=np.full(n_trials, np.nan))

    ax.plot(sites, w[0], lw=1.5)
    ax.axhline(0, color='gray', lw=.8)
    ax.set_ylabel(r'$w_R - w_L$')
    ax.set_title(rf'$\sigma_i={sigma_i:g}$, sample trial: {pairs[0]} stripe pairs, period {periods[0]:.2f}')

    print(f'sigma_i={sigma_i:g}')
    print(f'  k*={k_star:.6f}, theoretical continuum period={2 * np.pi / k_star:.6f}')
    print(f'  nearest ring Fourier mode m={m}, ring mode period={N / m:.6f}')
    print(f'  stripe pairs for 20 trials: {pairs.tolist()}')
    print(f'  simulated periods: {np.round(periods, 3).tolist()}')
    print(f'  mean observed period={np.nanmean(periods):.6f}, sample SD={np.nanstd(periods, ddof=1):.6f}')

axes[-1].set_xlabel('Cortical site a')
fig.savefig('problem1c.png', dpi=300, bbox_inches='tight')
plt.show()
