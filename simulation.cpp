#include "simulation.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <sstream>

#include "cuda_simulation.h"

double *pos, *vel, *acc, *mas;

double dist(int time, int i, int j) {
    double dx = pos[time * N_PARTICLE * DIMENSION + i * 2 + 0] -
                pos[time * N_PARTICLE * DIMENSION + j * 2 + 0];
    double dy = pos[time * N_PARTICLE * DIMENSION + i * 2 + 1] -
                pos[time * N_PARTICLE * DIMENSION + j * 2 + 1];
    return sqrt(dx * dx + dy * dy);
}

double get_pos(int time, int index, int xy) {
    return pos[time * N_PARTICLE * DIMENSION + index * DIMENSION + xy];
}

double get_weight(int index) { return mas[index]; }

std::string dump_pos(int time) {
    std::stringstream ss;
    for (int i = 0; i < N_PARTICLE; i++) {
        ss << "(" << get_pos(time, i, 0) << "," << get_pos(time, i, 1) << ") ";
    }
    return ss.str();
}

void init_sim() {
    pos =
        (double *)malloc(sizeof(double) * N_PARTICLE * TIME_LENGTH * DIMENSION);
    memset(pos, 0, sizeof(double) * N_PARTICLE * TIME_LENGTH * DIMENSION);
    vel =
        (double *)malloc(sizeof(double) * N_PARTICLE * TIME_LENGTH * DIMENSION);
    memset(vel, 0, sizeof(double) * N_PARTICLE * DIMENSION);
    acc =
        (double *)malloc(sizeof(double) * N_PARTICLE * TIME_LENGTH * DIMENSION);
    memset(acc, 0, sizeof(double) * N_PARTICLE * DIMENSION);
    mas = (double *)malloc(sizeof(double) * N_PARTICLE);
    memset(mas, 0, sizeof(double) * N_PARTICLE);

    for (int i = 0; i < N_PARTICLE; i++) {
        mas[i] = (MINIMUM_WEIGTHT_RATIO + (1.0 - MINIMUM_WEIGTHT_RATIO) *
                                              (rand() % RAND_MAX) / RAND_MAX) *
                 MAX_MASS;

        // mas[i] = 0.5 * MAX_MASS;
        pos[i * DIMENSION + 0] =
            (double)(rand() % RAND_MAX) / (double)RAND_MAX * MAX_POS;
        pos[i * DIMENSION + 1] =
            (double)(rand() % RAND_MAX) / (double)RAND_MAX * MAX_POS;
    }
}

void do_sim() {
    for (int t = 0; t < TIME_LENGTH - 1; t++) {
        // Update pos[t+1]
        for (int i = 0; i < N_PARTICLE; i++) {
            pos[(t + 1) * N_PARTICLE * DIMENSION + i * DIMENSION + 0] =
                pos[t * N_PARTICLE * DIMENSION + i * DIMENSION + 0] +
                vel[t * N_PARTICLE * DIMENSION + i * DIMENSION + 0] * dt +
                0.5 * acc[t * N_PARTICLE * DIMENSION + i * DIMENSION + 0] * dt *
                    dt;
            pos[(t + 1) * N_PARTICLE * DIMENSION + i * DIMENSION + 1] =
                pos[t * N_PARTICLE * DIMENSION + i * DIMENSION + 1] +
                vel[t * N_PARTICLE * DIMENSION + i * DIMENSION + 1] * dt +
                0.5 * acc[t * N_PARTICLE * DIMENSION + i * DIMENSION + 1] * dt *
                    dt;
        }

        // Update acc[t+1]
        for (int i = 0; i < N_PARTICLE; i++) {
            for (int j = 0; j < N_PARTICLE; j++) {
                if (i == j) continue;
                double dx =
                    pos[(t + 1) * N_PARTICLE * DIMENSION + i * DIMENSION + 0] -
                    pos[(t + 1) * N_PARTICLE * DIMENSION + j * DIMENSION + 0];
                double dy =
                    pos[(t + 1) * N_PARTICLE * DIMENSION + i * DIMENSION + 1] -
                    pos[(t + 1) * N_PARTICLE * DIMENSION + j * DIMENSION + 1];
                double r = sqrt(dx * dx + dy * dy);
                if (r < POS_EPS) r = POS_EPS;
                acc[(t + 1) * N_PARTICLE * DIMENSION + i * DIMENSION + 0] +=
                    -G * mas[j] * dx / (r * r * r);
                acc[(t + 1) * N_PARTICLE * DIMENSION + i * DIMENSION + 1] +=
                    -G * mas[j] * dy / (r * r * r);
            }
        }

        // Update vel[t+1]
        for (int i = 0; i < N_PARTICLE; i++) {
            vel[(t + 1) * N_PARTICLE * DIMENSION + i * DIMENSION + 0] =
                vel[t * N_PARTICLE * DIMENSION + i * DIMENSION + 0] +
                0.5 *
                    (acc[t * N_PARTICLE * DIMENSION + i * DIMENSION + 0] +
                     acc[(t + 1) * N_PARTICLE * DIMENSION + i * DIMENSION +
                         0]) *
                    dt;
            vel[(t + 1) * N_PARTICLE * DIMENSION + i * DIMENSION + 1] =
                vel[t * N_PARTICLE * DIMENSION + i * DIMENSION + 1] +
                0.5 *
                    (acc[t * N_PARTICLE * DIMENSION + i * DIMENSION + 1] +
                     acc[(t + 1) * N_PARTICLE * DIMENSION + i * DIMENSION +
                         1]) *
                    dt;
        }
    }
}

void do_cuda_sim() { call_cuda_sim(pos, vel, acc, mas); }

void finish_sim() {
    free(pos);
    free(vel);
    free(acc);
    free(mas);
}
