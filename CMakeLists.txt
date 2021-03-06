cmake_minimum_required(VERSION 3.0.2)

add_compile_options(-std=c++14)

project(traffic_light_ssd_fine_detector)

option(CUDA_VERBOSE "Verbose output of CUDA modules" OFF)

find_package(OpenCV REQUIRED)

# set flags for CUDA availability
option(CUDA_AVAIL "CUDA available" OFF)
find_package(CUDA)
if (CUDA_FOUND)
  find_library(CUBLAS_LIBRARIES cublas HINTS
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
    ${CUDA_TOOLKIT_ROOT_DIR}/lib
  )
  if (CUDA_VERBOSE)
    message("CUDA is available!")
    message("CUDA Libs: ${CUDA_LIBRARIES}")
    message("CUDA Headers: ${CUDA_INCLUDE_DIRS}")
  endif ()
  set(CUDA_AVAIL ON)
else()
  message("CUDA NOT FOUND")
  set(CUDA_AVAIL OFF)
endif (CUDA_FOUND)

# set flags for TensorRT availability
option(TRT_AVAIL "TensorRT available" OFF)
# try to find the tensorRT modules
find_library(NVINFER NAMES nvinfer)
find_library(NVONNXPARSER nvonnxparser)
find_library(NVINFER_PLUGIN NAMES nvinfer_plugin)
if(NVINFER AND NVONNXPARSER AND NVINFER_PLUGIN)
  if (CUDA_VERBOSE)
    message("TensorRT is available!")
    message("NVINFER: ${NVINFER}")
    message("NVPARSERS: ${NVPARSERS}")
    message("NVINFER_PLUGIN: ${NVINFER_PLUGIN}")
    message("NVONNXPARSER: ${NVONNXPARSER}")
  endif ()
  set(TRT_AVAIL ON)
else()
  message("TensorRT is NOT Available")
  set(TRT_AVAIL OFF)
endif()

# set flags for CUDNN availability
option(CUDNN_AVAIL "CUDNN available" OFF)
# try to find the CUDNN module
find_library(CUDNN_LIBRARY
NAMES libcudnn.so${__cudnn_ver_suffix} libcudnn${__cudnn_ver_suffix}.dylib ${__cudnn_lib_win_name}
PATHS $ENV{LD_LIBRARY_PATH} ${__libpath_cudart} ${CUDNN_ROOT_DIR} ${PC_CUDNN_LIBRARY_DIRS} ${CMAKE_INSTALL_PREFIX}
PATH_SUFFIXES lib lib64 bin
DOC "CUDNN library." )
if(CUDNN_LIBRARY)
  if (CUDA_VERBOSE)
    message("CUDNN is available!")
    message("CUDNN_LIBRARY: ${CUDNN_LIBRARY}")
  endif ()
  set(CUDNN_AVAIL ON)
else()
  message("CUDNN is NOT Available")
  set(CUDNN_AVAIL OFF)
endif()

# Download caffemodel and prototxt
find_program(GDOWN_AVAIL "gdown")
if (NOT GDOWN_AVAIL)
  message("gdown: command not found. External files could not be downloaded.")
endif()
set(PATH "${CMAKE_CURRENT_SOURCE_DIR}/data")
if (NOT EXISTS "${PATH}")
  execute_process(COMMAND mkdir -p ${PATH})
endif()
set(FILE "${PATH}/mb2-ssd-lite-tlr.onnx")
message(STATUS "Checking and downloading mb2-ssd-lite-tlr.onnx")
if (NOT EXISTS "${FILE}")
  message(STATUS "........${FILE}")
  message(STATUS "... file does not exist. Downloading now ...")
  execute_process(COMMAND gdown "https://drive.google.com/uc?id=1PGxVstVV_sVOLEEyl47m5lNXdLy_Iyob" -O ${PATH}/mb2-ssd-lite-tlr.onnx)
endif()
set(FILE "${PATH}/voc_labels_tl.txt")
message(STATUS "Checking and downloading voc_labels_tl.txt")
if (NOT EXISTS "${FILE}")
  message(STATUS "........${FILE}")
  message(STATUS "... file does not exist. Downloading now ...")
  execute_process(COMMAND gdown "https://drive.google.com/uc?id=1hPcKvKgKz0fqEo0cNAXH7roEletqZErL" -O ${PATH}/voc_labels_tl.txt)
endif()


if(TRT_AVAIL AND CUDA_AVAIL AND CUDNN_AVAIL)
  include_directories(
    include
    ${Opencv_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
  )


  set(CMAKE_CXX_FLAGS "-O2 -Wall -Wunused-variable ${CMAKE_CXX_FLAGS} -fpic -std=c++11 -pthread")

  ### ssd ###
  add_library(ssd
    lib/src/trt_ssd.cpp
  )

  target_include_directories(ssd PUBLIC
    lib/include
  )

  target_link_libraries(ssd
    ${NVINFER}
    ${NVONNXPARSER}
    ${NVINFER_PLUGIN}
    ${CUDA_LIBRARIES}
    ${CUBLAS_LIBRARIES}
    ${CUDA_curand_LIBRARY}
    ${CUDNN_LIBRARY}
  )

  add_library(covert SHARED
    src/convert_model.cpp
  )

  target_link_libraries(covert
    ${OpenCV_LIBS}
    ssd
  )


  add_executable(trt_ssd
    src/main.cpp
  )

  target_link_libraries(trt_ssd
    covert
    ${OpenCV_LIBS}
    ssd
  )
  install(TARGETS
    trt_ssd)

endif()
