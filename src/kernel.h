#pragma once
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <iomanip>
#include <glm/glm.hpp>
#include "utilityCore.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>

#include "kdTree.hpp"

#define blockSize 512
#define scene_scale 0.050f

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

void checkCUDAError(const char* msg, int line = -1);

namespace PointCloud {
    void initBuffers(std::vector<glm::vec3>& Ybuffer, std::vector<glm::vec3>& Xbuffer);
    void copyPointsToVBO(float* vbodptr_positions, float* vbodptr_colors);
    void cleanupBuffers();
    void copyTransCubesToVBO(float* vbodptr_positions, int* vbodptr_flags, float* vbodptr_sizes);
    void copyRotCubesToVBO(float* vbodptr_positions, int* vbodptr_flags, float* vbodptr_sizes);
}