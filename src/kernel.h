#pragma once
// Standard C/C++ Libraries
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <thread>
#include <mutex>

// OpenGL and GLFW Libraries
#include <GL/glew.h>
#include <GLFW/glfw3.h>

// CUDA Libraries
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// GLM Libraries
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// Thrust Libraries
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/random.h>
#include <thrust/copy.h>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>

#include "glslUtility.hpp"
#include "common.h"
#include "kdTree.hpp"

// GoICP
#include "goicp/jly_goicp.h"
#include "fgoicp/fgoicp.hpp"

#define blockSize 512
#define scene_scale 1.0f

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

void checkCUDAError(const char* msg, int line = -1);

namespace PointCloud {
    void initBuffers(std::vector<glm::vec3>& Ybuffer, std::vector<glm::vec3>& Xbuffer);
    void copyPointsToVBO(int N, glm::vec3* posBuffer, glm::vec3* colBuffer, float* vbodptr_positions, float* vbodptr_colors);
    void cleanupBuffers();
}