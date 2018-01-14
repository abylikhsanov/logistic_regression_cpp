cmake_minimum_required(VERSION 3.9)
project(logistic_regression)

set(CMAKE_CXX_STANDARD 11)

add_executable(logistic_regression main.cpp Logistic.cpp Logistic.h HiddenLayer.cpp HiddenLayer.h Helper.h)