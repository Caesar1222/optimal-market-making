"""
Market-making simulator (Step 2).

A minimal order-book environment for testing market-making strategies. It steps
through a mid-price path and simulates fills of the strategy's posted bid/ask
quotes using an Avellaneda-Stoikov-style fill model:

    market orders arrive with intensity  lambda(delta) = A * exp(-kappa * delta)

where `delta` is the distance of a quote from the mid. Quote close to the mid ->
filled often, earn little; quote far out -> filled rarely, earn more. That
trade-off is the whole market-making problem.

DESIGN NOTE: the simulator knows nothing about any specific strategy. It only
takes quotes and returns fills. Strategies are separate callables. This keeps the
hot loop (the environment) easy to port to C++ later, while strategy logic stays
in Python for fast iteration.

ASSUMPTIONS (be able to defend these):
  - Aggregated fill intensity, not real queue position.
  - Each side's fill is drawn independently per step.
  - The mid-price path is exogenous (our own trades don't move it).
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class SimConfig:
    A: float = 1.0          # base market-order arrival intensity
    kappa: float = 1.5      # how fast fill probability decays with distance
    dt: float = 1.0         # length of one time step (seconds)
    order_size: float = 1.0  # size filled per market order
    seed: int = 0


@dataclass
class SimState:
    inventory: float = 0.0
    cash: float = 0.0


class MarketMakingSimulator:
    def __init__(self, mids, config: SimConfig = SimConfig()):
        self.mids = np.asarray(mids, dtype=float)
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)

    def fill_prob(self, delta):
        """Probability a quote at distance `delta` from the mid fills in one step."""
        delta = max(delta, 0.0)                      # quoting through mid -> treat as at-mid
        intensity = self.cfg.A * np.exp(-self.cfg.kappa * delta)
        return 1.0 - np.exp(-intensity * self.cfg.dt)

    def run(self, strategy):
        """
        strategy: callable(i, mid, state, steps_left) -> (bid_price, ask_price)
                  Return None for a side to skip quoting it.

        Returns (history_dict, final_pnl), where final_pnl liquidates any leftover
        inventory at the last mid.
        """
        state = SimState()
        n = len(self.mids)
        keys = ("mid", "inventory", "cash", "pnl", "bid", "ask", "buy_fill", "sell_fill")
        hist = {k: np.zeros(n) for k in keys}

        for i, mid in enumerate(self.mids):
            steps_left = n - i
            bid, ask = strategy(i, mid, state, steps_left)

            buy_fill = sell_fill = 0

            # our BID (resting buy) is hit by an incoming SELL market order -> we BUY
            if bid is not None:
                if self.rng.random() < self.fill_prob(mid - bid):
                    state.inventory += self.cfg.order_size
                    state.cash -= bid * self.cfg.order_size
                    buy_fill = 1

            # our ASK (resting sell) is lifted by an incoming BUY market order -> we SELL
            if ask is not None:
                if self.rng.random() < self.fill_prob(ask - mid):
                    state.inventory -= self.cfg.order_size
                    state.cash += ask * self.cfg.order_size
                    sell_fill = 1

            pnl = state.cash + state.inventory * mid     # mark-to-market wealth
            hist["mid"][i] = mid
            hist["inventory"][i] = state.inventory
            hist["cash"][i] = state.cash
            hist["pnl"][i] = pnl
            hist["bid"][i] = bid if bid is not None else np.nan
            hist["ask"][i] = ask if ask is not None else np.nan
            hist["buy_fill"][i] = buy_fill
            hist["sell_fill"][i] = sell_fill

        final_pnl = state.cash + state.inventory * self.mids[-1]
        return hist, final_pnl


# --- a trivial strategy to test the simulator (this doubles as the Step 3 baseline) ---
def fixed_spread_strategy(half_spread):
    """Symmetric quoting at a fixed distance from the mid, no inventory management."""
    def strategy(i, mid, state, steps_left):
        return mid - half_spread, mid + half_spread
    return strategy