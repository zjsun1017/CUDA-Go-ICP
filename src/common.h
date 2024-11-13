#ifndef COMMOM_H
#define COMMOM_H

#define VISUALIZE 1
#define CUDA_NAIVE 0
#define CUDA_KDTREE	0
#define CPU_GOICP 0
#define SUBSAMPLE 1 // Subsample data point cloud

#if SUBSAMPLE
const float subsampleRate = 0.05;
#endif

#endif