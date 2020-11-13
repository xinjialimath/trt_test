cmake_minimum_required(VERSION 2.8.3)
project(traffic_trt)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_CXX_FLAGS "-march=native -O3 -Wall")

add_executable(trt-ssd main.cpp trt_ssd.cpp convert_model.cpp)
