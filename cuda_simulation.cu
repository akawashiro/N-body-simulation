#include "constants.h"

__global__ void sim_kernel(double *pos, double *vel, double *acc, double *mas){
    for (int t = 0; t < TIME_LENGTH; t++) {
        for(int i=0;i<N_PARTICLE * DIMENSION;i++){
            acc[i] = 0;
        }

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

                double r_dx = pos[i * 2 + 0] - pos[j * 2 + 0];
                double r_dy = pos[i * 2 + 1] - pos[j * 2 + 1];
                double r = sqrt(r_dx * r_dx + r_dy * r_dy);

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

void call_cuda_sim(double *pos_host, double *vel_host, double *acc_host, double *mas_host) {
    double *pos, *mas, *acc, *vel;
    size_t pos_size = sizeof(double) * N_PARTICLE * (TIME_LENGTH + 1) *
                           DIMENSION;
    size_t mas_size = sizeof(double) * N_PARTICLE;
    size_t vel_size = sizeof(double) * N_PARTICLE * DIMENSION;
    size_t acc_size = sizeof(double) * N_PARTICLE * DIMENSION;

    cudaMalloc(&pos, pos_size);
    cudaMalloc(&mas, mas_size);
    cudaMalloc(&vel, vel_size);
    cudaMalloc(&acc, acc_size);

    cudaMemcpy(pos, pos_host, pos_size, cudaMemcpyHostToDevice);
    cudaMemcpy(mas, mas_host, mas_size, cudaMemcpyHostToDevice);
    cudaMemcpy(vel, vel_host, vel_size, cudaMemcpyHostToDevice);
    cudaMemcpy(acc, acc_host, acc_size, cudaMemcpyHostToDevice);

    sim_kernel<<<1, 1>>>(pos, vel, acc, mas);

    cudaMemcpy(pos_host, pos, pos_size, cudaMemcpyDeviceToHost);

    cudaFree(pos);
    cudaFree(mas);
    cudaFree(acc);
    cudaFree(vel);
}
