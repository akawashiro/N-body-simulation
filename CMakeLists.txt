cmake_minimum_required(VERSION 3.16)

# You must use g++-9 to compile this project because g++-10 does not support
# cuda.

project(N-body-simulation)

find_package(OpenGL REQUIRED)
find_package(CUDA REQUIRED)

if(CUDA_FOUND)
    message("Cuda version: " ${CUDA_VERSION})
    CUDA_ADD_LIBRARY(cuda_simulation STATIC
        cuda_simulation.cu
        cuda_simulation.h
    )
endif()

add_executable(main main.cpp simulation.cpp)
target_link_libraries(main ${OPENGL_LIBRARY} cuda_simulation "-Wl,-lglfw -Wl,-lGLEW -Wl,-lglut")
