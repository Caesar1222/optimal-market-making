#include <pybind11/pybind11.h>

int add(int a, int b) { return a + b; }

namespace py = pybind11;

PYBIND11_MODULE(fastsim, m) {
    m.doc() = "C++ core for the market-making simulator";
    m.def("add", &add, "Add two integers");
}
