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

int mode = 0;
#define ICP_CPU 0
#define ICP_GPU	1
#define ICP_KDTREE_GPU	2
#define GOICP_CPU	3
#define GOICP_GPU	4

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


int maxCubeDivide = 3;
int numCubes = 0;

std::vector<glm::vec3> transCubePosBuffer;
std::vector<glm::vec3> transCubeColBuffer;

std::vector<glm::vec3> rotCubePosBuffer;
std::vector<glm::vec3> rotCubeColBuffer;

glm::vec3* dev_cubePosBuffer;
glm::vec3* dev_cubeColBuffer;

std::string deviceName;
GLFWwindow* window;
GLFWwindow* secondWindow;

const char *projectName;
int main(int argc, char* argv[]);

void mainLoop();
void runCUDA();

void initPointCloud(int argc, char** argv);
void initSearchSpace();
void initBufferAndkdTree();
