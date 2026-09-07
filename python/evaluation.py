"""
Evaluation utilities (Step 3 onward).

Turns raw simulation output into the metrics that matter for market making, and
runs a strategy many times across random seeds to produce a distribution of
outcomes rather than a single noisy number.

Reused in Step 4 (comparing A-S to the baseline) and Step 6 (stress testing).
"""
from dataclasses import replace
import numpy as np
import pandas as pd

from simulator import MarketMakingSimulator, SimConfig


def run_metrics(hist, final_pnl):
    """Summarize a single simulation run into the metrics that matter."""
    inv = hist["inventory"]
    return {
        "final_pnl": final_pnl,
        "n_fills": int(hist["buy_fill"].sum() + hist["sell_fill"].sum()),
        "max_abs_inventory": float(np.abs(inv).max()),   # worst position held
        "terminal_inventory": float(inv[-1]),            # position before liquidation
        "inventory_std": float(inv.std()),               # how much inventory swung
    }


def monte_carlo(mids, strategy, base_config, n_runs=200):
    """
    Run `strategy` across `n_runs` random seeds and return a DataFrame of
    per-run metrics. Only the fill randomness varies run to run; the mid-price
    path is held fixed (see the note in the Step 3 guide about this limitation).

    `strategy` must be stateless across runs (ours are).
    """
    records = []
    for seed in range(n_runs):
        cfg = replace(base_config, seed=seed)
        sim = MarketMakingSimulator(mids, cfg)
        hist, final_pnl = sim.run(strategy)
        records.append(run_metrics(hist, final_pnl))
    return pd.DataFrame(records)


def summarize(df_metrics):
    """Collapse a Monte-Carlo DataFrame into headline numbers."""
    pnl = df_metrics["final_pnl"]
    return {
        "mean_pnl": pnl.mean(),
        "std_pnl": pnl.std(),
        "pnl_p05": pnl.quantile(0.05),                   # 5th-percentile (bad case)
        "sharpe_like": pnl.mean() / (pnl.std() + 1e-9),  # risk-adjusted consistency
        "mean_max_abs_inventory": df_metrics["max_abs_inventory"].mean(),
        "mean_n_fills": df_metrics["n_fills"].mean(),
    }