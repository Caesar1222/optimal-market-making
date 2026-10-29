"""
Scenario generation and multi-path evaluation (Step 6).

Step 4 showed A-S beating the baseline on ONE historical price path. To turn that
into a robust finding we:
  (a) resample many synthetic paths from the real data via block bootstrap, so the
      result no longer depends on a single afternoon of market data;
  (b) construct adversarial scenarios (a trend, a volatility spike) to find where
      the model breaks.
"""
from dataclasses import replace
import numpy as np
import pandas as pd

from evaluation import run_metrics
from simulator import MarketMakingSimulator


def block_bootstrap_paths(mids, n_paths, block=50, seed=0):
    """
    Generate synthetic mid-price paths by resampling contiguous BLOCKS of returns
    from the real series and stitching them together. Blocks (rather than single
    returns) preserve short-term autocorrelation, so the synthetic paths keep a
    realistic microstructure texture.
    """
    mids = np.asarray(mids, float)
    rets = np.diff(mids)
    n = len(rets)
    rng = np.random.default_rng(seed)
    paths = []
    for _ in range(n_paths):
        out = []
        while len(out) < n:
            start = rng.integers(0, n - block)
            out.extend(rets[start:start + block])
        out = np.array(out[:n])
        paths.append(mids[0] + np.concatenate([[0.0], np.cumsum(out)]))
    return paths


def trending_path(mids, drift_per_step, seed=0):
    """
    A bootstrapped path with a deterministic drift added. This violates A-S's core
    no-drift assumption, so it tests the model against adverse selection.
    Tune `drift_per_step` to your price scale (e.g. a total move of a few %).
    """
    base = block_bootstrap_paths(mids, 1, seed=seed)[0]
    return base + drift_per_step * np.arange(len(base))


def vol_spike_path(mids, spike_mult=3.0, spike_frac=(0.4, 0.6), seed=0):
    """A path with a burst of elevated volatility in the middle section."""
    mids = np.asarray(mids, float)
    rets = np.diff(mids)
    rng = np.random.default_rng(seed)
    sample = rng.choice(rets, size=len(rets), replace=True)
    a, b = int(spike_frac[0] * len(sample)), int(spike_frac[1] * len(sample))
    sample[a:b] *= spike_mult
    return mids[0] + np.concatenate([[0.0], np.cumsum(sample)])


def evaluate_on_paths(paths, strategy, base_config, seeds_per_path=20):
    """
    Run a strategy across many paths (and several fill-seeds per path) and return a
    DataFrame of per-run metrics. This captures BOTH price-path and fill randomness,
    unlike the single-path monte_carlo in evaluation.py.
    """
    records = []
    for p_idx, path in enumerate(paths):
        sim_base = None
        for s in range(seeds_per_path):
            cfg = replace(base_config, seed=p_idx * 1000 + s)
            hist, final_pnl = MarketMakingSimulator(path, cfg).run(strategy)
            records.append(run_metrics(hist, final_pnl))
    return pd.DataFrame(records)
