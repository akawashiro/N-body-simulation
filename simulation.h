#include <string>

#include "constants.h"

double get_pos(int time, int index, int xy);
std::string dump_pos(int time);
double get_weight(int index);
void init_sim();
void do_sim();
void do_cuda_sim();
void finish_sim();
