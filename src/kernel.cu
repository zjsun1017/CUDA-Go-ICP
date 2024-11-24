#define GLM_FORCE_CUDA
#include "kernel.h"


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

extern FlattenedKDTree* dev_fkdt;
extern float* dev_minDists;
extern size_t* dev_minIndices;

extern int numTransCubes;

extern std::vector<glm::vec3> transCubePosBuffer;
extern std::vector<int> transCubeFlagBuffer;
extern std::vector<float> transCubeSizeBuffer;

extern std::vector<glm::vec3> rotCubePosBuffer;
extern std::vector<int> rotCubeFlagBuffer;
extern std::vector<float> rotCubeSizeBuffer;

extern glm::vec3* dev_transCubePosBuffer;
extern int* dev_transCubeFlagBuffer;
extern float* dev_transCubeSizeBuffer;

extern glm::vec3* dev_rotCubePosBuffer;
extern int* dev_rotCubeFlagBuffer;
extern float* dev_rotCubeSizeBuffer;

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
		vbo[4 * index + 0] = col[index].x + 0.3f;
		vbo[4 * index + 1] = col[index].y + 0.3f;
		vbo[4 * index + 2] = col[index].z + 0.3f;
		vbo[4 * index + 3] = 1.0f;
	}
}

__global__ void kernCopyCubeDataToVBO(
	int numTransCubes, glm::vec3* posBuffer, int* flagBuffer, float* sizeBuffer,
	float* vbodptr_positions, int* vbodptr_flags) {

	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < numTransCubes) {
		glm::vec3 center = posBuffer[index];
		float size = sizeBuffer[index];
		int flag = flagBuffer[index];

		// Define face offsets for a cube
		const glm::vec3 faceOffsets[8] = {
			glm::vec3(-0.5f, -0.5f, -0.5f),
			glm::vec3(0.5f, -0.5f, -0.5f),
			glm::vec3(0.5f,  0.5f, -0.5f),
			glm::vec3(-0.5f,  0.5f, -0.5f),
			glm::vec3(-0.5f, -0.5f,  0.5f),
			glm::vec3(0.5f, -0.5f,  0.5f),
			glm::vec3(0.5f,  0.5f,  0.5f),
			glm::vec3(-0.5f,  0.5f,  0.5f)
		};

		for (int i = 0; i < 8; ++i) {
			int vIdx = index * 8 + i; // 8 vertices per cube

			glm::vec3 vertexPos = center + faceOffsets[i] * size;
			vbodptr_positions[3 * vIdx + 0] = vertexPos.x;
			vbodptr_positions[3 * vIdx + 1] = vertexPos.y;
			vbodptr_positions[3 * vIdx + 2] = vertexPos.z;	

		}

		vbodptr_flags[index] = flag;
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

	cudaMallocManaged((void**)&dev_transCubePosBuffer, numTransCubes * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMallocManaged dev_transCubePosBuffer failed!");
	cudaMallocManaged((void**)&dev_transCubeFlagBuffer, numTransCubes * sizeof(int));
	checkCUDAErrorWithLine("cudaMallocManaged dev_transCubeFlagBuffer failed!");
	cudaMallocManaged((void**)&dev_transCubeSizeBuffer, numTransCubes * sizeof(float));
	checkCUDAErrorWithLine("cudaMallocManaged dev_transCubeSizeBuffer failed!");
	cudaMallocManaged((void**)&dev_rotCubePosBuffer, numTransCubes * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMallocManaged dev_rotCubePosBuffer failed!");
	cudaMallocManaged((void**)&dev_rotCubeFlagBuffer, numTransCubes * sizeof(int));
	checkCUDAErrorWithLine("cudaMallocManaged dev_rotCubeFlagBuffer failed!");
	cudaMallocManaged((void**)&dev_rotCubeSizeBuffer, numTransCubes * sizeof(float));
	checkCUDAErrorWithLine("cudaMallocManaged dev_rotCubeSizeBuffer failed!");

	// Set Posistion Buffer
	std::copy(dataBuffer.begin(), dataBuffer.end(), dev_dataBuffer);
	std::copy(modelBuffer.begin(), modelBuffer.end(), dev_modelBuffer);
	std::copy(modelBuffer.begin(), modelBuffer.end(), dev_pos);
	std::copy(dataBuffer.begin(), dataBuffer.end(), dev_pos + numModelPoints);

	// Set color buffer
	dim3 dataBlocksPerGrid((numDataPoints + blockSize - 1) / blockSize);
	dim3 modelBlocksPerGrid((numModelPoints + blockSize - 1) / blockSize);
	kernResetVec3Buffer << <modelBlocksPerGrid, blockSize >> > (numModelPoints, dev_col, glm::vec3(0, 0, 1));
	kernResetVec3Buffer << < dataBlocksPerGrid, blockSize >> > (numDataPoints, &dev_col[numModelPoints], glm::vec3(1, 0, 0));
	kernResetVec3Buffer << <modelBlocksPerGrid, blockSize >> > (numModelPoints, dev_col, glm::vec3(0, 0, 1));
	cudaDeviceSynchronize();

	// Set search buffer
	std::copy(transCubePosBuffer.begin(), transCubePosBuffer.end(), dev_transCubePosBuffer);
	std::copy(transCubeFlagBuffer.begin(), transCubeFlagBuffer.end(), dev_transCubeFlagBuffer);
	std::copy(transCubeSizeBuffer.begin(), transCubeSizeBuffer.end(), dev_transCubeSizeBuffer);
	cudaDeviceSynchronize();
}

void PointCloud::copyPointsToVBO(float* vbodptr_positions, float* vbodptr_colors) {
	dim3 fullBlocksPerGrid((numPoints + blockSize - 1) / blockSize);
	kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> > (numPoints, dev_pos, vbodptr_positions, scene_scale);
	kernCopyColorsToVBO << <fullBlocksPerGrid, blockSize >> > (numPoints, dev_col, vbodptr_colors, scene_scale);
	checkCUDAErrorWithLine("copyPointsToVBO failed!");

	cudaDeviceSynchronize();
}

void PointCloud::copyTransCubesToVBO( float* vbodptr_positions, int* vbodptr_flags) {
	dim3 fullBlocksPerGrid((numTransCubes + blockSize - 1) / blockSize);
	kernCopyCubeDataToVBO << <fullBlocksPerGrid, blockSize >> > (
		numTransCubes, dev_transCubePosBuffer, dev_transCubeFlagBuffer, dev_transCubeSizeBuffer,
		vbodptr_positions, vbodptr_flags);
	checkCUDAErrorWithLine("copyCubesToVBO failed!");
	cudaDeviceSynchronize();
}

void PointCloud::copyRotCubesToVBO( float* vbodptr_positions, int* vbodptr_flags) {
	dim3 fullBlocksPerGrid((numTransCubes + blockSize - 1) / blockSize);
	kernCopyCubeDataToVBO << <fullBlocksPerGrid, blockSize >> > (
		numTransCubes, dev_rotCubePosBuffer, dev_rotCubeFlagBuffer, dev_rotCubeSizeBuffer,
		vbodptr_positions, vbodptr_flags);
	checkCUDAErrorWithLine("copyCubesToVBO failed!");
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

	checkCUDAErrorWithLine("cudaFree failed!");
}

