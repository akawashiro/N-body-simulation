#include <GL/glut.h>

#include <cstdlib>

// Note that the number of points is kept fairly small because a display
// callback should NEVER run for too long.
void display() {
    glClear(GL_COLOR_BUFFER_BIT);

    glPointSize(100.0f);
    glBegin(GL_POINTS);
    glVertex2f(500, 500.0);
    glVertex2f(0, 0.0);
    glEnd();
    glFlush();
}

// Performs application-specific initialization. Sets colors and sets up a
// simple orthographic projection.
void init() {
    // Set a deep purple background and draw in a greenish yellow.
    glClearColor(0.25, 0.0, 0.2, 1.0);
    glColor3f(0.6, 1.0, 0.0);

    // Set up the viewing volume: 500 x 500 x 1 window with origin lower left.
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 500.0, 0.0, 500.0, 0.0, 1.0);
}

// Initializes GLUT, the display mode, and main window; registers callbacks;
// does application initialization; enters the main event loop.
int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    glutInitWindowSize(1000, 1000);
    glutInitWindowPosition(40, 40);
    glutCreateWindow("N body simulation");
    glutDisplayFunc(display);
    init();
    glutMainLoop();
}
