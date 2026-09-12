"""
Market-making strategies.

A strategy is a callable: (i, mid, state, steps_left) -> (bid_price, ask_price).
It reads the current situation and returns where to quote. The simulator handles
everything else. Keeping strategies as pure functions of state + parameters is
what makes them swappable and (for A-S) portable to C++ later.
"""
import numpy as np


def fixed_spread_strategy(half_spread):
    """Baseline (Step 3): symmetric quotes at a fixed distance, no inventory control."""
    def strategy(i, mid, state, steps_left):
        return mid - half_spread, mid + half_spread
    return strategy


def avellaneda_stoikov_strategy(gamma, sigma, kappa, dt, horizon=None):
    """
    Avellaneda-Stoikov optimal quotes (Step 4).

      gamma   : risk aversion (units: 1/price). YOUR choice — sweep it.
      sigma   : mid volatility per unit time (same time unit as dt). From Step 1.
      kappa   : fill-intensity decay (units: 1/price). From calibration.
      dt      : length of one step (time units).
      horizon : if None, uses the finite-horizon (T - t) = steps_left * dt, so the
                inventory term shrinks toward the end of the run (textbook A-S).
                If set to a constant (e.g. 60.0 seconds), uses the stationary
                variant — better for a continuous market like crypto.

    Note: the base intensity A does NOT enter the optimal quotes; it lives only in
    the simulator's fill model.
    """
    def strategy(i, mid, state, steps_left):
        q = state.inventory
        time_left = horizon if horizon is not None else steps_left * dt

        # inventory-adjusted reservation price: shifts quotes to shed inventory
        r = mid - q * gamma * sigma**2 * time_left

        # optimal total spread: risk premium + order-flow component
        spread = gamma * sigma**2 * time_left + (2.0 / gamma) * np.log(1 + gamma / kappa)

        return r - spread / 2, r + spread / 2
    return strategy


def as_spread(gamma, sigma, kappa, time_left):
    """The A-S total spread at zero inventory — use it to sanity-check parameters
    against the median market spread you measured in Step 1."""
    return gamma * sigma**2 * time_left + (2.0 / gamma) * np.log(1 + gamma / kappa)