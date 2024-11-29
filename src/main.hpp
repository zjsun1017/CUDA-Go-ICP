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
#include "goicp_kernel.h"
#include "kernel.h"

#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

#include "goicp/jly_goicp.h"
#include <thread>
#include <mutex>

int mode = 0;
int numPoints = 0;
int numDataPoints = 0;
int numModelPoints = 0;

// k-d Tree
PointCloudAdaptor tree;
KDTree kdtree(3, tree, nanoflann::KDTreeSingleIndexAdaptorParams(10));
FlattenedKDTree* dev_fkdt;
float* dev_minDists;
size_t* dev_minIndices;

// ICP Buffers
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

// Search Space Buffers
int maxCubeDivide = 3;
int numCubes = 0;
std::vector<glm::vec3> transCubePosBuffer;
std::vector<glm::vec3> transCubeColBuffer;
std::vector<glm::vec3> rotCubePosBuffer;
std::vector<glm::vec3> rotCubeColBuffer;
glm::vec3* dev_cubePosBuffer;
glm::vec3* dev_cubeColBuffer;

// GOICP on CPU setup
GoICP goicp;
Matrix prev_optR = Matrix::eye(3);
Matrix prev_optT = Matrix(3, 1);
std::mutex mtx;
glm::vec3* dev_optDataBuffer;
glm::vec3* dev_curDataBuffer;

// Window settings
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
