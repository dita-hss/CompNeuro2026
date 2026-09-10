"""Classical Hodgkin-Huxley model under constant current injection.

Python port of hh_sim.m. Same parameters, same shifted potential
convention of the 1952 papers, so rest is near V = 0 and the spike peak
near V = 100.

Call it and capture what you need:

    t, V, spk, rate = hh_sim()

Running the file directly (python hh_sim.py) prints the spike count and
rate at the default Id = 7. There is no MATLAB-style ans here, but t is
30001 elements of 0 to 300 in steps of 0.01, so avoid echoing the whole
return value in a notebook cell.

Regression checks with default parameters:
    Id = 7 gives sustained firing at 58.3 Hz
    Id = 5 gives no sustained firing
The 58.3 is solver independent: BDF, LSODA, Radau and RK45 all agree to
three digits, so a port that lands more than a Hz away has a real bug,
most likely a sign in an exponent. The removable singularities in
alpha_m and alpha_n at V = 25 and V = 10 are patched below; if you
rewrite them, check that alpha_m(25) and alpha_n(10) come out finite.

The reported rheobase depends on how you define repetitive firing: both
skipfrac and the spike count you demand of it will move the answer,
because the underlying bifurcation is subcritical. State the criterion
you used rather than treating the number as absolute.
"""

import numpy as np
from scipy.integrate import solve_ivp

__all__ = ["hh_sim"]


# ---- gating rate functions (ms^-1), with removable singularities patched ----
def _am(V):
    return 1.0 if abs(V - 25) < 1e-6 else 0.1 * (25 - V) / (np.exp((25 - V) / 10) - 1)


def _bm(V):
    return 4 * np.exp(-V / 18)


def _ah(V):
    return 0.07 * np.exp(-V / 20)


def _bh(V):
    return 1 / (np.exp((30 - V) / 10) + 1)


def _an(V):
    return 0.1 if abs(V - 10) < 1e-6 else 0.01 * (10 - V) / (np.exp((10 - V) / 10) - 1)


def _bn(V):
    return 0.125 * np.exp(-V / 80)


def hh_sim(Id=7.0, tmax=300.0, gK=36.0, gNa=120.0, skipfrac=0.4):
    """Integrate the HH model under constant current.

    Parameters
    ----------
    Id : injected current density (uA/cm^2). Defaults to 7, above
         rheobase, so hh_sim() with no arguments gives repetitive
         firing.
    tmax : simulation duration (ms).
    gK, gNa : maximal conductances (mS/cm^2).
    skipfrac : fraction of the trace discarded before counting spikes, so
        that the onset transient is not mistaken for repetitive firing.

    Returns
    -------
    t, V : time vector (ms) and voltage trace (mV).
    spk : spike times, upward crossings of 40 mV.
    rate : mean firing rate in Hz over the counted window, 0 if fewer
        than two spikes.
    """
    C, gL, ENa, EK, EL = 1.0, 0.3, 115.0, -12.0, 10.6

    def rhs(_, y):
        V, m, h, n = y
        INa = gNa * m**3 * h * (V - ENa)
        IK = gK * n**4 * (V - EK)
        IL = gL * (V - EL)
        return [
            (Id - INa - IK - IL) / C,
            _am(V) * (1 - m) - _bm(V) * m,
            _ah(V) * (1 - h) - _bh(V) * h,
            _an(V) * (1 - n) - _bn(V) * n,
        ]

    V0 = 0.0  # steady state gating at rest
    y0 = [
        V0,
        _am(V0) / (_am(V0) + _bm(V0)),
        _ah(V0) / (_ah(V0) + _bh(V0)),
        _an(V0) / (_an(V0) + _bn(V0)),
    ]

    t = np.arange(0, tmax + 1e-9, 0.01)
    sol = solve_ivp(rhs, [0, tmax], y0, method="BDF", t_eval=t,
                    rtol=1e-8, atol=1e-10, max_step=0.05)
    V = sol.y[0]

    i0 = max(1, int(round(skipfrac * len(t))))
    up = np.flatnonzero((V[i0:-1] < 40) & (V[i0 + 1:] >= 40)) + i0
    spk = t[up]

    rate = 1000 * (len(spk) - 1) / (spk[-1] - spk[0]) if len(spk) >= 2 else 0.0
    return t, V, spk, rate


# if __name__ == "__main__":
#     t, V, spk, rate = hh_sim(Id=6.25)
#     print(f"Id = 6.5: {len(spk)} spikes, rate = {rate:.1f} Hz")



#### Additions to professors code

import matplotlib.pyplot as plt

def part_a():
    # Bisection
    low = 5.0
    high = 7.0

    while high - low > 1e-5:
        mid = (low + high) / 2
        t, V, spk, rate = hh_sim(Id=mid, skipfrac=0.8)
        if len(spk) >= 2:
            high = mid
        else:
            low = mid

    rheobase = high

    print("Below:", low)
    print("Above:", high)
    print(f"Rheobase = {rheobase:.3g} uA/cm^2")


    # Voltage traces
    I_below = low
    I_above = high

    t1, V1, spk1, rate1 = hh_sim(Id=I_below)
    t2, V2, spk2, rate2 = hh_sim(Id=I_above)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(t1, V1)
    axes[0].set_title(f"Just below rheobase: I = {I_below:.6f}")
    axes[0].set_xlabel("Time (ms)")
    axes[0].set_ylabel("Voltage (mV)")
    axes[0].set_xlim(0, 300)
    axes[0].set_ylim(-20, 110)

    axes[1].plot(t2, V2)
    axes[1].set_title(f"Just above rheobase: I = {I_above:.6f}")
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("Voltage (mV)")
    axes[1].set_xlim(0, 300)
    axes[1].set_ylim(-20, 110)

    plt.tight_layout()
    plt.show()


def part_b():
    # Rheobase from part (a)
    rheobase = 6.2614086

    # Start slightly above rheobase and go to about twice rheobase
    currents = np.linspace(rheobase + 0.005, 2 * rheobase, 20)

    rates = []

    for I in currents:
        t, V, spk, rate = hh_sim(Id=I, skipfrac=0.8)
        rates.append(rate)

    # values
    for I, rate in zip(currents, rates):
        print(f"I = {I:.3f} uA/cm^2, rate = {rate:.3f} Hz")

    # f-I curve
    plt.figure(figsize=(7, 5))
    plt.plot(currents, rates, marker="o")
    plt.xlabel("Injected current, I (uA/cm^2)")
    plt.ylabel("Firing rate, f (Hz)")
    plt.title("Hodgkin-Huxley f-I Curve")
    plt.grid(True)
    plt.show()

    # Onset firing rate
    print(f"Onset rate = {rates[0]:.3g} Hz")

def part_c():
    # original gK = 36
    low_36 = 5.0
    high_36 = 7.0

    while high_36 - low_36 > 1e-5:
        mid = (low_36 + high_36) / 2
        t, V, spk, rate = hh_sim(Id=mid, gK=36, skipfrac=0.8)
        if len(spk) >= 2:
            high_36 = mid
        else:
            low_36 = mid

    rheobase_36 = high_36


    # rheobase for reduced gK = 30
    low_30 = 2.0
    high_30 = 4.0

    while high_30 - low_30 > 1e-5:
        mid = (low_30 + high_30) / 2
        t, V, spk, rate = hh_sim(Id=mid, gK=30, skipfrac=0.8)

        if len(spk) >= 2:
            high_30 = mid
        else:
            low_30 = mid

    rheobase_30 = high_30


    # Rheobase table
    print("gK (mS/cm^2)    Rheobase (uA/cm^2)")
    print("----------------------------------")
    print(f"36               {rheobase_36:.3g}")
    print(f"30               {rheobase_30:.3g}")


    # f-I curves over the same current range
    currents = np.linspace(2.0, 2 * rheobase_36, 35)

    rates_36 = []
    rates_30 = []

    for I in currents:
        # gK = 36
        t, V, spk, rate = hh_sim(Id=I, gK=36, skipfrac=0.8)
        rates_36.append(rate)

        # gK = 30
        t, V, spk, rate = hh_sim(Id=I, gK=30, skipfrac=0.8)
        rates_30.append(rate)


    # Overlay f-I curves
    plt.figure(figsize=(8, 5))
    plt.plot(currents, rates_36, marker="o", label="gK = 36 mS/cm²")
    plt.plot(currents, rates_30, marker="s", label="gK = 30 mS/cm²")
    # Mark rheobase values
    plt.axvline(rheobase_36,linestyle="--",label=f"Rheobase gK=36: {rheobase_36:.2f}")
    plt.axvline(rheobase_30,linestyle="--",label=f"Rheobase gK=30: {rheobase_30:.2f}")
    plt.xlabel("Injected current, I (uA/cm^2)")
    plt.ylabel("Firing rate, f (Hz)")
    plt.title("Effect of Potassium Conductance on the f-I Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()