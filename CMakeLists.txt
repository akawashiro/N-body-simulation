cmake_minimum_required(VERSION 3.16)

project(N-body-simulation)

find_package(OpenGL REQUIRED)

add_executable(main main.cpp simulation.cpp)
target_link_libraries(main ${OPENGL_LIBRARY} "-Wl,-lglfw -Wl,-lGLEW -Wl,-lglut")
