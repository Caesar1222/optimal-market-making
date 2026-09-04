cmake_minimum_required(VERSION 3.15)
project(fastsim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(fastsim src/main.cpp)
install(TARGETS fastsim DESTINATION .)