#pragma once

// const int N_PARTICLE = 1000;
// const int TIME_LENGTH = 1000;
// const int DIMENSION = 2;
// const double G = 6.67430e-11;
// const double dt = 0.1;
// const double MAX_MASS = 10000000.0;
// const double MAX_POS = 100.0;
// const double EDGE_LENGTH = 500;
// const int TAIL_TIME = 100;
// const int REFRESH_TIME = 50000;
// const bool EXPAND_WINDOW = true;
// const double POINT_SIZE = 5;
// const double TAIL_POINT_SIZE = 1;

// This setting works
const int N_THREAD_PER_BLOCK = 64;
const int N_PARTICLE = 64 * 16;
const int CUDA_TIME_LENGTH = 1024;
const int TIME_LENGTH = 131072;
const int DIMENSION = 2;
const double G = 6.67430e-11;
const double dt = 0.01;
const double MAX_MASS = 100000;
const double MAX_POS = 10;
const double EDGE_LENGTH = 500;
const double MINIMUM_WEIGTHT_RATIO = 0.5;
const double POS_EPS = MAX_POS / 100;
const int TAIL_TIME = 100;
const int REFRESH_TIME = 50000;
const bool EXPAND_WINDOW = true;
const double POINT_SIZE = 10;
const double TAIL_POINT_SIZE = 1;
const int DISPLAY_RATIO = 300;
const bool USE_CUDA = true;
