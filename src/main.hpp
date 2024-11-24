#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <cstdlib>
#include <string>
#include <sstream>
#include <fstream>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "utilityCore.hpp"
#include "glslUtility.hpp"
#include "icp_kernel.h"
#include "kernel.h"

#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

int numPoints = 0;
int numDataPoints = 0;
int numModelPoints = 0;

PointCloudAdaptor tree;
KDTree kdtree(3, tree, nanoflann::KDTreeSingleIndexAdaptorParams(10));
FlattenedKDTree* dev_fkdt;
float* dev_minDists;
size_t* dev_minIndices;

std::vector<glm::vec3> dataBuffer;
std::vector<glm::vec3> modelBuffer;

glm::vec3* dev_pos;
glm::vec3* dev_col;

glm::vec3* dev_dataBuffer;
glm::vec3* dev_modelBuffer;
glm::vec3* dev_corrBuffer;

glm::vec3* dev_centeredCorrBuffer;
glm::vec3* dev_centeredDataBuffer;
glm::mat3* dev_ABtBuffer;

int numCubes = 100;
glm::vec3* dev_transCubePosBuffer;
glm::vec3* dev_transCubeColBuffer;
float* dev_transCubeSizeBuffer;

glm::vec3* dev_rotCubePosBuffer;
glm::vec3* dev_rotCubeColBuffer;
float* dev_rotCubeSizeBuffer;

std::string deviceName;
GLFWwindow* window;
GLFWwindow* secondWindow;

const char *projectName;
int main(int argc, char* argv[]);

void mainLoop();
void runCUDA();

void initPointCloud(int argc, char** argv);
void initBufferAndkdTree();
