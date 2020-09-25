#include <GL/glut.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <tuple>

#include "simulation.h"

int current_time = 0;

std::tuple<double, double, double> color_from_index(int index) {
    double r = (get_weight(index) / MAX_MASS - MINIMUM_WEIGTHT_RATIO) /
               MINIMUM_WEIGTHT_RATIO;
    return {r, 1.0 - r, 0.0};
}

// Note that the number of points is kept fairly small because a display
// callback should NEVER run for too long.
void display() {
    // std::cout << dump_pos(current_time) << std::endl;
    usleep(REFRESH_TIME);

    glClear(GL_COLOR_BUFFER_BIT);

    double x_max = 1, x_min = 0, y_max = 1, y_min = 0;

    if (EXPAND_WINDOW) {
        for (int t = current_time; std::max(0, current_time - TAIL_TIME) < t;
             t--) {
            for (int i = 0; i < N_PARTICLE; i++) {
                x_max = std::max(x_max, get_pos(t, i, 0));
                x_min = std::min(x_min, get_pos(t, i, 0));
                y_max = std::max(y_max, get_pos(t, i, 1));
                y_min = std::min(y_min, get_pos(t, i, 1));
            }
        }
    }

    // Draw current points
    glPointSize(POINT_SIZE);
    glEnable(GL_POINT_SMOOTH);
    glBegin(GL_POINTS);
    for (int i = 0; i < N_PARTICLE; i++) {
        auto c = color_from_index(i);
        glColor3f(std::get<0>(c), std::get<1>(c), std::get<2>(c));
        glVertex2f((get_pos(current_time, i, 0) - x_min) / (x_max - x_min) *
                       EDGE_LENGTH,
                   (get_pos(current_time, i, 1) - y_min) / (y_max - y_min) *
                       EDGE_LENGTH);
    }
    glEnd();

    // Draw tails
    glPointSize(TAIL_POINT_SIZE);
    glBegin(GL_POINTS);
    for (int t = current_time;
         std::max(0, current_time - TAIL_TIME * DISPLAY_RATIO) < t;
         t -= DISPLAY_RATIO) {
        for (int i = 0; i < N_PARTICLE; i++) {
            auto c = color_from_index(i);
            glColor3f(std::get<0>(c), std::get<1>(c), std::get<2>(c));
            glVertex2f(
                (get_pos(t, i, 0) - x_min) / (x_max - x_min) * EDGE_LENGTH,
                (get_pos(t, i, 1) - y_min) / (y_max - y_min) * EDGE_LENGTH);
        }
    }
    glEnd();

    glFlush();

    current_time = (current_time + DISPLAY_RATIO) % TIME_LENGTH;
}

// Performs application-specific initialization. Sets colors and sets up a
// simple orthographic projection.
void init() {
    // Set a black background and draw in a greenish yellow.
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glColor3f(0.6, 1.0, 0.0);

    // Set up the viewing volume: 500 x 500 x 1 window with origin lower left.
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 500.0, 0.0, 500.0, 0.0, 1.0);
}

// Initializes GLUT, the display mode, and main window; registers callbacks;
// does application initialization; enters the main event loop.
int main(int argc, char** argv) {
    init_sim();
    auto start = std::chrono::steady_clock::now();
    if (USE_CUDA) {
        do_cuda_sim();
    } else {
        do_sim();
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "simulation time: " << elapsed_seconds.count() << "s\n";

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    glutInitWindowSize(1000, 1000);
    glutInitWindowPosition(40, 40);
    glutCreateWindow("N body simulation");
    glutDisplayFunc(display);
    glutIdleFunc(glutPostRedisplay);
    init();

    glutMainLoop();

    finish_sim();
}
