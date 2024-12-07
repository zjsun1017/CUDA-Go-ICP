#pragma once
#include "glslUtility.hpp"
#include "kernel.h"
#include "icp_kernel.h"
#include "goicp_kernel.h"
#include "goicp/jly_goicp.h"
#include "fgoicp/fgoicp.hpp"

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

// GOICP on CPU setup
GoICP* goicp;
Matrix prev_optR = Matrix::eye(3);
Matrix prev_optT = Matrix(3, 1);
std::mutex mtx;				   // shared by GOICP GPU
glm::vec3* dev_optDataBuffer;  // shared by GOICP GPU
glm::vec3* dev_curDataBuffer;  // shared by GOICP GPU
bool goicp_finished = false;

// GOICP on GPU setup
icp::FastGoICP* fgoicp;
glm::mat3 prev_optR_fgoicp(1.0f);
glm::vec3 prev_optT_fgoicp(0.0f);
float sse_threshold = 0;
float mse_threshold = 0;

// Window settings
std::string deviceName;
GLFWwindow* window;
const char *projectName;

int main(int argc, char* argv[]);
void mainLoop();
void runCUDA();
void initPointCloud(int argc, char** argv);
void initBufferAndkdTree();
