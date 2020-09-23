#include "simulation.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const double G = 6.67430e-11;
const double dt = 1.0;
const double MAX_MASS = 10000000.0;

double *pos, *vel, *acc, *mas;

double dist(int i, int j) {
    double dx = pos[i * 2 + 0] - pos[j * 2 + 0];
    double dy = pos[i * 2 + 1] - pos[j * 2 + 1];
    return sqrt(dx * dx + dy * dy);
}

double get_pos(int time, int index, int xy) {
    return pos[time * N_PARTICLE * DIMENSION + index * DIMENSION + xy];
}

void init_sim() {
    pos = (double *)malloc(sizeof(double) * N_PARTICLE * (TIME_LENGTH + 1) *
                           DIMENSION);
    memset(pos, 0, sizeof(double) * N_PARTICLE * TIME_LENGTH * DIMENSION);
    vel = (double *)malloc(sizeof(double) * N_PARTICLE * DIMENSION);
    memset(vel, 0, sizeof(double) * N_PARTICLE * DIMENSION);
    acc = (double *)malloc(sizeof(double) * N_PARTICLE * DIMENSION);
    memset(acc, 0, sizeof(double) * N_PARTICLE * DIMENSION);
    mas = (double *)malloc(sizeof(double) * N_PARTICLE);
    memset(mas, 0, sizeof(double) * N_PARTICLE);

    for (int i = 0; i < N_PARTICLE; i++) {
        mas[i] = (rand() % RAND_MAX) * MAX_MASS / RAND_MAX;
        pos[i * DIMENSION + 0] = (double)(rand() % RAND_MAX) / (double)RAND_MAX;
        pos[i * DIMENSION + 1] = (double)(rand() % RAND_MAX) / (double)RAND_MAX;
    }
}

void simulate() {
    for (int t = 0; t < TIME_LENGTH; t++) {
        memset(acc, 0, sizeof(double) * N_PARTICLE * DIMENSION);
        for (int i = 0; i < N_PARTICLE; i++) {
            pos[(t + 1) * N_PARTICLE * DIMENSION + i * DIMENSION + 0] =
                pos[t * N_PARTICLE * DIMENSION + i * DIMENSION + 0];
            pos[(t + 1) * N_PARTICLE * DIMENSION + i * DIMENSION + 1] =
                pos[t * N_PARTICLE * DIMENSION + i * DIMENSION + 1];
        }
        for (int i = 0; i < N_PARTICLE; i++) {
            for (int j = 0; j < N_PARTICLE; j++) {
                if (i == j) continue;
                double dx =
                    pos[t * N_PARTICLE * DIMENSION + i * DIMENSION + 0] -
                    pos[t * N_PARTICLE * DIMENSION + j * DIMENSION + 0];
                double dy =
                    pos[t * N_PARTICLE * DIMENSION + i * DIMENSION + 1] -
                    pos[t * N_PARTICLE * DIMENSION + j * DIMENSION + 1];
                double r = dist(i, j);
                acc[i * DIMENSION + 0] += -G * mas[j] * dx / (r * r * r);
                acc[i * DIMENSION + 1] += -G * mas[j] * dy / (r * r * r);
            }
        }
        for (int i = 0; i < N_PARTICLE; i++) {
            vel[i * DIMENSION + 0] += acc[i * DIMENSION + 0] * dt;
            vel[i * DIMENSION + 1] += acc[i * DIMENSION + 1] * dt;
        }
        for (int i = 0; i < N_PARTICLE; i++) {
            pos[(t + 1) * N_PARTICLE * DIMENSION + i * DIMENSION + 0] +=
                vel[i * DIMENSION + 0] * dt;
            pos[(t + 1) * N_PARTICLE * DIMENSION + i * DIMENSION + 1] +=
                vel[i * DIMENSION + 1] * dt;
        }
    }
}

void finish_sim() {
    free(pos);
    free(vel);
    free(acc);
    free(mas);
}
