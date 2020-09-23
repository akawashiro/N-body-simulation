#include <GL/glut.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>

#include "simulation.h"

const double EDGE_LENGTH = 500;
int time = 0;

// Note that the number of points is kept fairly small because a display
// callback should NEVER run for too long.
void display() {
    glClear(GL_COLOR_BUFFER_BIT);

    glPointSize(30.0f);
    glBegin(GL_POINTS);

    usleep(100000);
    // printf("time = %d =========================\n", time);
    for (int i = 0; i < N_PARTICLE; i++) {
        glColor3f(1.0 * (double)i / N_PARTICLE,
                  1.0 * (double)(N_PARTICLE - i) / N_PARTICLE, 1.0);
        glVertex2f(get_pos(time, i, 0) * EDGE_LENGTH,
                   get_pos(time, i, 1) * EDGE_LENGTH);
        // printf("%lf, %lf\n", get_pos(time, i, 0), get_pos(time, i, 1));
    }

    glEnd();
    glFlush();

    time = (time + 1) % TIME_LENGTH;
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
    simulate();

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
