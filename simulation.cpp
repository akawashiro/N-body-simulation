#include <math.h>
#include <stdlib.h>

const int N_PARTICLE = 100;
const int TIME_LENGTH = 10000;
const int DIMENSION = 2;

double *pos, *vel, *acc, *mas;

double dist(int i, int j) {
    double dx = pos[i][0] - pos[j][0];
    double dy = pos[i][1] - pos[j][1];
    return sqrt(dx * dx + dy * dy);
}

void init_sim() {
    pos = malloc(sizeof(double) * N_PARTICLE * TIME_LENGTH * DIMENSION);
    memset(pos, 0, sizeof(double) * N_PARTICLE * TIME_LENGTH * DIMENSION);
    vel = malloc(sizeof(double) * N_PARTICLE * DIMENSION);
    memset(vel, 0, sizeof(double) * N_PARTICLE * DIMENSION);
    acc = malloc(sizeof(double) * N_PARTICLE * DIMENSION);
    memset(acc, 0, sizeof(double) * N_PARTICLE * DIMENSION);
    mas = malloc(sizeof(double) * N_PARTICLE * DIMENSION);
    memset(mas, 0, sizeof(double) * N_PARTICLE * DIMENSION);

    for (int i = 0; i < N_PARTICLE; i++) {
        mas[i] = rand() % N_PARTICLE;
    }
}

void simulate() {
    for (int t = 0; t < TIME_LENGTH; t++) {
        for (int i = 0; i < N_PARTICLE; i++) {
            for (int j = 0; j < N_PARTICLE; j++) {
            }
        }
    }
}

void finish_sim() {
    free(pos);
    free(vel);
    free(acc);
    free(mas);
}
