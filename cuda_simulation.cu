#include <cooperative_groups.h>

#include "constants.h"

__global__ void sim_kernel(double *pos, double *vel, double *acc, double *mas){
    cooperative_groups::grid_group g = cooperative_groups::this_grid();
    int particle_id = blockIdx.x *blockDim.x + threadIdx.x;

    for (int t = 0; t < TIME_LENGTH - 1; t++) {
        // Update pos[t+1]
            pos[(t + 1) * N_PARTICLE * DIMENSION + particle_id * DIMENSION + 0] =
                pos[t * N_PARTICLE * DIMENSION + particle_id * DIMENSION + 0] +
                vel[(t % 2) * N_PARTICLE * DIMENSION + particle_id * DIMENSION + 0] * dt +
                0.5 * acc[(t % 2) * N_PARTICLE * DIMENSION + particle_id * DIMENSION + 0] * dt *
                    dt;
            pos[(t + 1) * N_PARTICLE * DIMENSION + particle_id * DIMENSION + 1] =
                pos[t * N_PARTICLE * DIMENSION + particle_id * DIMENSION + 1] +
                vel[(t % 2) * N_PARTICLE * DIMENSION + particle_id * DIMENSION + 1] * dt +
                0.5 * acc[(t % 2) * N_PARTICLE * DIMENSION + particle_id * DIMENSION + 1] * dt *
                    dt;

        g.sync();

        // Update acc[t+1]
        double acc_x = 0, acc_y = 0;
        for (int i = 0; i < N_PARTICLE; i++) {
            if (i == particle_id) continue;
            double dx =
                pos[(t + 1) * N_PARTICLE * DIMENSION + particle_id * DIMENSION + 0] -
                pos[(t + 1) * N_PARTICLE * DIMENSION + i * DIMENSION + 0];
            double dy =
                pos[(t + 1) * N_PARTICLE * DIMENSION + particle_id * DIMENSION + 1] -
                pos[(t + 1) * N_PARTICLE * DIMENSION + i * DIMENSION + 1];
            double r = sqrt(dx * dx + dy * dy);
            if (r < POS_EPS) r = POS_EPS;
            acc_x += -G * mas[i] * dx / (r * r * r);
            acc_y += -G * mas[i] * dy / (r * r * r);
        }
        acc[((t + 1) % 2) * N_PARTICLE * DIMENSION + particle_id  * DIMENSION + 0] = acc_x;
        acc[((t + 1) % 2) * N_PARTICLE * DIMENSION + particle_id  * DIMENSION + 1] = acc_y;

        g.sync();

        // Update vel[t+1]
        vel[((t + 1) % 2) * N_PARTICLE * DIMENSION + particle_id * DIMENSION + 0] =
            vel[(t % 2) * N_PARTICLE * DIMENSION + particle_id * DIMENSION + 0] +
            0.5 *
                (acc[(t % 2) * N_PARTICLE * DIMENSION + particle_id * DIMENSION + 0] +
                 acc[((t + 1) % 2) * N_PARTICLE * DIMENSION + particle_id * DIMENSION +
                     0]) *
                dt;
        vel[((t + 1) % 2) * N_PARTICLE * DIMENSION + particle_id * DIMENSION + 1] =
            vel[(t % 2) * N_PARTICLE * DIMENSION + particle_id * DIMENSION + 1] +
            0.5 *
                (acc[(t % 2) * N_PARTICLE * DIMENSION + particle_id * DIMENSION + 1] +
                 acc[((t + 1) % 2) * N_PARTICLE * DIMENSION + particle_id * DIMENSION +
                     1]) *
                dt;
    }
}

void call_cuda_sim(double *pos_host, double *vel_host, double *acc_host, double *mas_host) {
    double *pos, *mas, *acc, *vel;
    size_t pos_size = sizeof(double) * N_PARTICLE * TIME_LENGTH *
                           DIMENSION;
    size_t mas_size = sizeof(double) * N_PARTICLE;
    size_t vel_size = sizeof(double) * N_PARTICLE * 2 * DIMENSION;
    size_t acc_size = sizeof(double) * N_PARTICLE * 2 * DIMENSION;

    cudaMalloc(&pos, pos_size);
    cudaMalloc(&mas, mas_size);
    cudaMalloc(&vel, vel_size);
    cudaMalloc(&acc, acc_size);

    cudaMemcpy(pos, pos_host, pos_size, cudaMemcpyHostToDevice);
    cudaMemcpy(mas, mas_host, mas_size, cudaMemcpyHostToDevice);
    cudaMemcpy(vel, vel_host, vel_size, cudaMemcpyHostToDevice);
    cudaMemcpy(acc, acc_host, acc_size, cudaMemcpyHostToDevice);


    const void* args[]= {&pos, &vel, &acc, &mas};
    dim3 grid(N_PARTICLE / N_THREAD_PER_BLOCK, 1, 1);
	dim3 block(N_THREAD_PER_BLOCK, 1, 1);
	cudaLaunchCooperativeKernel((void*)&sim_kernel, grid, block, (void**)args);

    cudaMemcpy(pos_host, pos, pos_size, cudaMemcpyDeviceToHost);

    cudaFree(pos);
    cudaFree(mas);
    cudaFree(acc);
    cudaFree(vel);
}
