#define GLM_FORCE_CUDA
#include "kernel.h"

extern int mode;
extern int numPoints;
extern int numDataPoints;
extern int numModelPoints;

extern glm::vec3* dev_pos;
extern glm::vec3* dev_col;
extern glm::vec3* dev_dataBuffer;
extern glm::vec3* dev_modelBuffer;
extern glm::vec3* dev_corrBuffer;
extern glm::vec3* dev_centeredCorrBuffer;
extern glm::vec3* dev_centeredDataBuffer;
extern glm::mat3* dev_ABtBuffer;

extern glm::vec3* dev_optDataBuffer;
extern glm::vec3* dev_curDataBuffer;

extern FlattenedKDTree* dev_fkdt;
extern float* dev_minDists;
extern size_t* dev_minIndices;

extern int numCubes;
extern std::vector<glm::vec3> transCubePosBuffer;
extern std::vector<glm::vec3> transCubeColBuffer;
extern std::vector<glm::vec3> rotCubePosBuffer;
extern std::vector<glm::vec3> rotCubeColBuffer;
extern glm::vec3* dev_cubePosBuffer;
extern glm::vec3* dev_cubeColBuffer;
extern glm::vec3* dev_rotCubePosBuffer;
extern glm::vec3* dev_rotCubeColBuffer;

extern int maxTNodes;
extern float* dev_errors;
extern float* dev_rot_ub_trans_ub;
extern float* dev_rot_ub_trans_lb;


// Helper Functions
void checkCUDAError(const char* msg, int line) {
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		if (line >= 0) {
			fprintf(stderr, "Line %d: ", line);
		}
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

// Helper kernel functions
__global__ void kernResetVec3Buffer(int N, glm::vec3* intBuffer, glm::vec3 value) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < N) {
		intBuffer[index] = value;
	}
}

__global__ void kernCopyPositionsToVBO(int N, glm::vec3* pos, float* vbo, float s_scale) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	float c_scale = -1.0f / s_scale;

	if (index < N) {
		vbo[4 * index + 0] = pos[index].x * c_scale;
		vbo[4 * index + 1] = pos[index].y * c_scale;
		vbo[4 * index + 2] = pos[index].z * c_scale;
		vbo[4 * index + 3] = 1.0f;
	}
}

__global__ void kernCopyColorsToVBO(int N, glm::vec3* col, float* vbo, float s_scale) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	if (index < N) {
		vbo[4 * index + 0] = col[index].x;
		vbo[4 * index + 1] = col[index].y;
		vbo[4 * index + 2] = col[index].z;
		vbo[4 * index + 3] = 1.0f;
	}
}

void PointCloud::initBuffers(std::vector<glm::vec3>& dataBuffer, std::vector<glm::vec3>& modelBuffer) {
	// Use unified memory
	cudaDeviceSynchronize();
	cudaMallocManaged((void**)&dev_pos, numPoints * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMallocManaged dev_pos failed!");
	cudaMallocManaged((void**)&dev_col, numPoints * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMallocManaged dev_col failed!");
	cudaMallocManaged((void**)&dev_dataBuffer, numDataPoints * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMallocManaged dev_dataBuffer failed!");
	cudaMallocManaged((void**)&dev_modelBuffer, numModelPoints * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMallocManaged dev_modelBuffer failed!");
	cudaMallocManaged((void**)&dev_corrBuffer, numDataPoints * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMallocManaged dev_corrBuffer failed!");
	cudaMallocManaged((void**)&dev_centeredDataBuffer, numDataPoints * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMallocManaged dev_centeredDataBuffer failed!");
	cudaMallocManaged((void**)&dev_centeredCorrBuffer, numDataPoints * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMallocManaged dev_centeredCorrBuffer failed!");
	cudaMallocManaged((void**)&dev_ABtBuffer, numDataPoints * sizeof(glm::mat3));
	checkCUDAErrorWithLine("cudaMallocManaged dev_ABtBuffer failed!");
	cudaMallocManaged((void**)&dev_minDists, numDataPoints * sizeof(float));
	checkCUDAErrorWithLine("cudaMallocManaged dev_dataBuffer failed!");
	cudaMallocManaged((void**)&dev_minIndices, numModelPoints * sizeof(size_t));
	checkCUDAErrorWithLine("cudaMallocManaged dev_modelBuffer failed!");

	if (mode == GOICP_CPU)
	{
		cudaMallocManaged((void**)&dev_optDataBuffer, numDataPoints * sizeof(glm::vec3));
		checkCUDAErrorWithLine("cudaMallocManaged dev_optDataBuffer failed!");
		cudaMallocManaged((void**)&dev_curDataBuffer, numDataPoints * sizeof(glm::vec3));
		checkCUDAErrorWithLine("cudaMallocManaged dev_curDataBuffer failed!");
	}

	if (mode == GOICP_GPU)
	{
		cudaMallocManaged((void**)&dev_errors, sizeof(float) * numDataPoints);
		checkCUDAErrorWithLine("cudaMallocManaged dev_errors failed!");
		cudaMallocManaged((void**)&dev_rot_ub_trans_ub, sizeof(float) * numDataPoints * maxTNodes);
		checkCUDAErrorWithLine("cudaMallocManaged dev_rot_ub_trans_ub failed!");
		cudaMallocManaged((void**)&dev_rot_ub_trans_lb, sizeof(float) * numDataPoints * maxTNodes);
		checkCUDAErrorWithLine("cudaMallocManaged dev_rot_ub_trans_lb failed!");

		cudaMallocManaged((void**)&dev_cubePosBuffer, 2 * numCubes * sizeof(glm::vec3));
		checkCUDAErrorWithLine("cudaMallocManaged dev_cubePosBuffer failed!");
		cudaMallocManaged((void**)&dev_cubeColBuffer, 2 * numCubes * sizeof(glm::vec3));
		checkCUDAErrorWithLine("cudaMallocManaged dev_transCubeSizeBuffer failed!");
	}

	// Set Posistion Buffer
	std::copy(dataBuffer.begin(), dataBuffer.end(), dev_dataBuffer);
	std::copy(modelBuffer.begin(), modelBuffer.end(), dev_modelBuffer);
	std::copy(modelBuffer.begin(), modelBuffer.end(), dev_pos);
	std::copy(dataBuffer.begin(), dataBuffer.end(), dev_pos + numModelPoints);
	if (mode == GOICP_CPU) 
		std::copy(dataBuffer.begin(), dataBuffer.end(), dev_pos + numModelPoints + numDataPoints);

	// Set color buffer
	dim3 dataBlocksPerGrid((numDataPoints + blockSize - 1) / blockSize);
	dim3 modelBlocksPerGrid((numModelPoints + blockSize - 1) / blockSize);
	kernResetVec3Buffer << <modelBlocksPerGrid, blockSize >> > (numModelPoints, dev_col, glm::vec3(0, 0, 1));
	kernResetVec3Buffer << < dataBlocksPerGrid, blockSize >> > (numDataPoints, &dev_col[numModelPoints], glm::vec3(1, 0, 0));
	kernResetVec3Buffer << <modelBlocksPerGrid, blockSize >> > (numModelPoints, dev_col, glm::vec3(0, 0, 1));
	if (mode == GOICP_CPU) 
		kernResetVec3Buffer <<< dataBlocksPerGrid, blockSize >> > (numDataPoints, &dev_col[numModelPoints + numDataPoints], glm::vec3(1.0));
	cudaDeviceSynchronize();

	// Set search buffer
	if (mode == GOICP_GPU)
	{
		std::copy(transCubePosBuffer.begin(), transCubePosBuffer.end(), dev_cubePosBuffer);
		std::copy(transCubeColBuffer.begin(), transCubeColBuffer.end(), dev_cubeColBuffer);
		std::copy(rotCubePosBuffer.begin(), rotCubePosBuffer.end(), dev_cubePosBuffer + numCubes);
		std::copy(rotCubeColBuffer.begin(), rotCubeColBuffer.end(), dev_cubeColBuffer + numCubes);
		cudaDeviceSynchronize();
	}
}

void PointCloud::copyPointsToVBO(int N, glm::vec3* posBuffer, glm::vec3* colBuffer, float* vbodptr_positions, float* vbodptr_colors) {
	dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);
	kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> > (N, posBuffer, vbodptr_positions, scene_scale);
	kernCopyColorsToVBO << <fullBlocksPerGrid, blockSize >> > (N, colBuffer, vbodptr_colors, scene_scale);
	checkCUDAErrorWithLine("copyPointsToVBO failed!");
	cudaDeviceSynchronize();
}

void PointCloud::cleanupBuffers() {
	cudaFree(dev_pos);
	cudaFree(dev_col);
	cudaFree(dev_modelBuffer);
	cudaFree(dev_dataBuffer);
	cudaFree(dev_corrBuffer);
	cudaFree(dev_centeredDataBuffer);
	cudaFree(dev_centeredCorrBuffer);
	cudaFree(dev_ABtBuffer);
	cudaFree(dev_fkdt);
	cudaFree(dev_minDists);
	cudaFree(dev_minIndices);
	cudaFree(dev_cubePosBuffer);
	cudaFree(dev_cubeColBuffer);
	if (mode == GOICP_CPU)
	{
		cudaFree(dev_optDataBuffer);
		cudaFree(dev_curDataBuffer);
	}
	if (mode == GOICP_GPU)
	{
		cudaFree(dev_errors);
		cudaFree(dev_rot_ub_trans_ub);
		cudaFree(dev_rot_ub_trans_lb);
		cudaFree(dev_cubePosBuffer);
		cudaFree(dev_cubeColBuffer);
	}

	checkCUDAErrorWithLine("cudaFree failed!");
}

