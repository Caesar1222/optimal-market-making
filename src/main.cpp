#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
#include <tuple>

namespace py = pybind11;

// Compute (bid, ask) for the chosen strategy.
//   strategy 0 = fixed spread, strategy 1 = Avellaneda-Stoikov
static inline void quotes(int strategy, double mid, double q, double time_left,
                          double half_spread, double gamma, double sigma, double kappa,
                          double& bid, double& ask) {
    if (strategy == 0) {                              // fixed spread
        bid = mid - half_spread;
        ask = mid + half_spread;
    } else {                                          // Avellaneda-Stoikov
        double r = mid - q * gamma * sigma * sigma * time_left;
        double spread = gamma * sigma * sigma * time_left
                        + (2.0 / gamma) * std::log(1.0 + gamma / kappa);
        bid = r - spread / 2.0;
        ask = r + spread / 2.0;
    }
}

// Run the full market-making loop in C++.
// u_bid / u_ask are pre-generated uniform(0,1) draws (one per step, per side), so
// that the Python reference and this C++ version consume identical randomness and
// can be checked for an exact match.
// Returns (inventory_path, buy_fills, sell_fills, final_pnl).
std::tuple<py::array_t<double>, long, long, double>
run_market_making(py::array_t<double, py::array::c_style | py::array::forcecast> mids,
                  py::array_t<double, py::array::c_style | py::array::forcecast> u_bid,
                  py::array_t<double, py::array::c_style | py::array::forcecast> u_ask,
                  int strategy, double A, double kappa, double dt, double order_size,
                  double half_spread, double gamma, double sigma, double horizon) {
    auto m  = mids.unchecked<1>();
    auto ub = u_bid.unchecked<1>();
    auto ua = u_ask.unchecked<1>();
    py::ssize_t n = m.shape(0);

    auto inv_out = py::array_t<double>(n);
    auto inv = inv_out.mutable_unchecked<1>();

    double inventory = 0.0, cash = 0.0;
    long buy_fills = 0, sell_fills = 0;

    for (py::ssize_t i = 0; i < n; ++i) {
        double mid = m(i);
        double time_left = (horizon > 0.0) ? horizon : (double)(n - i) * dt;

        double bid, ask;
        quotes(strategy, mid, inventory, time_left, half_spread, gamma, sigma, kappa, bid, ask);

        // our BID (resting buy) hit by a SELL market order -> we BUY
        double delta_b = mid - bid; if (delta_b < 0.0) delta_b = 0.0;
        double p_b = 1.0 - std::exp(-A * std::exp(-kappa * delta_b) * dt);
        if (ub(i) < p_b) { inventory += order_size; cash -= bid * order_size; ++buy_fills; }

        // our ASK (resting sell) lifted by a BUY market order -> we SELL
        double delta_a = ask - mid; if (delta_a < 0.0) delta_a = 0.0;
        double p_a = 1.0 - std::exp(-A * std::exp(-kappa * delta_a) * dt);
        if (ua(i) < p_a) { inventory -= order_size; cash += ask * order_size; ++sell_fills; }

        inv(i) = inventory;
    }

    double final_pnl = cash + inventory * m(n - 1);
    return std::make_tuple(inv_out, buy_fills, sell_fills, final_pnl);
}

int add(int a, int b) { return a + b; }   // kept from the Step 0 smoke test

PYBIND11_MODULE(fastsim, m) {
    m.doc() = "C++ core for the market-making simulator";
    m.def("add", &add, "Add two integers");
    m.def("run_market_making", &run_market_making,
          "Run the market-making simulation loop in C++",
          py::arg("mids"), py::arg("u_bid"), py::arg("u_ask"),
          py::arg("strategy"), py::arg("A"), py::arg("kappa"), py::arg("dt"),
          py::arg("order_size"), py::arg("half_spread"), py::arg("gamma"),
          py::arg("sigma"), py::arg("horizon"));
}