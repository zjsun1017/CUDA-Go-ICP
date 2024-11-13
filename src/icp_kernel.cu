#define GLM_FORCE_CUDA
#include "icp_kernel.h"
#include "svd3.h"

extern int numPoints;
extern int numDataPoints;
extern int numModelPoints;

extern glm::vec3* dev_pos;
extern glm::vec3* dev_col;

extern glm::vec3 * dev_dataBuffer;
extern glm::vec3* dev_modelBuffer;
#if CPU_GOICP
extern glm::vec3* dev_optDataBuffer;
extern glm::vec3* dev_curDataBuffer;
#else
extern glm::vec3* dev_corrBuffer;

extern glm::vec3* dev_centeredCorrBuffer;
extern glm::vec3* dev_centeredDataBuffer;
extern glm::mat3* dev_ABtBuffer;
#endif

//Helper functions
void matSVD(glm::mat3& ABt, glm::mat3& U, glm::mat3& S, glm::mat3& V)
{
	svd(ABt[0].x, ABt[0].y, ABt[0].z,
		ABt[1].x, ABt[1].y, ABt[1].z,
		ABt[2].x, ABt[2].y, ABt[2].z,

		U[0].x, U[0].y, U[0].z,
		U[1].x, U[1].y, U[1].z,
		U[2].x, U[2].y, U[2].z,

		S[0].x, S[0].y, S[0].z,
		S[1].x, S[1].y, S[1].z,
		S[2].x, S[2].y, S[2].z,

		V[0].x, V[0].y, V[0].z,
		V[1].x, V[1].y, V[1].z,
		V[2].x, V[2].y, V[2].z);
}

// CPU ICP pipeline
void ICP::CPUStep(std::vector<glm::vec3>& dataBuffer, std::vector<glm::vec3>& modelBuffer) {

	// Find nearest correspondences
	std::vector<glm::vec3> corrBuffer(numDataPoints);
	for (int i = 0; i < numDataPoints; i++) {
		float distMin = FLT_MAX;
		for (int j = 0; j < numModelPoints; j++) {
			float dist = glm::distance(dataBuffer[i], modelBuffer[j]);
			if (distMin > dist) {
				distMin = dist;
				corrBuffer[i] = modelBuffer[j];
			}
		}
	}

	// Centralize
	glm::vec3 meanData(0.0f);
	glm::vec3 meanCorr(0.0f);
	for (int i = 0; i < numDataPoints; i++) {
		meanData += dataBuffer[i];
		meanCorr += corrBuffer[i];
	}
	meanData /= static_cast<float>(numDataPoints);
	meanCorr /= static_cast<float>(numDataPoints);

	std::vector<glm::vec3> centeredDataBuffer(numDataPoints);
	std::vector<glm::vec3> centeredCorrBuffer(numDataPoints);
	for (int i = 0; i < numDataPoints; i++) {
		centeredDataBuffer[i] = dataBuffer[i] - meanData;
		centeredCorrBuffer[i] = corrBuffer[i] - meanCorr;
	}

	// Calculating rotation and translations
	// PnP algorithm: minimizing A-RB equals to minimizing R-AB^T
	// Kabsch algorthm: Orthogonalize the rotation matrix with SVD: AB^T = USV^T, R = UV^T
	glm::mat3 ABt(0.0f);
	for (int i = 0; i < numDataPoints; i++) {
		ABt += glm::outerProduct(centeredDataBuffer[i], centeredCorrBuffer[i]);
	}

	//compute SVD of ABt
	glm::mat3 R(0.0f), U(0.0f), S(0.0f), V(0.0f);
	glm::vec3 T(0.0f);

	matSVD(ABt, U, S, V);

	R = glm::transpose(U) * V; // Strange glm::mat column sequence >:(
	T = meanCorr - (R * meanData);

	// Update and draw
	for (int i = 0; i < numDataPoints; i++)
		dataBuffer[i] = R * dataBuffer[i] + T;
	std::copy(dataBuffer.begin(), dataBuffer.end(), dev_pos + numModelPoints);
	cudaDeviceSynchronize();
}

// Helper kernel functions
__global__ void kernSearchNearest(int numDataPoints, int numModelPoints,
	const glm::vec3* dataBuffer, const glm::vec3* modelBuffer, glm::vec3* corrBuffer) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < numDataPoints) {
		float distMin = FLT_MAX;
		for (int j = 0; j < numModelPoints; j++) {
			float dist = glm::distance(dataBuffer[index], modelBuffer[j]);
			if (distMin > dist) {
				distMin = dist;
				corrBuffer[index] = modelBuffer[j];
			}
		}
	}
}

__global__ void kernCentralize(int numDataPoints, glm::vec3* in, glm::vec3* out, glm::vec3 mean) {

	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= numDataPoints) return;

	out[index] = in[index] - mean;
}

__global__ void kernOuterProduct(int numDataPoints,
	glm::vec3* A, glm::vec3* B, glm::mat3* out) {

	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= numDataPoints) return;

	out[index] = glm::outerProduct(A[index], B[index]);
}

__global__ void kernTransform(int numDataPoints, glm::vec3* pos, glm::mat3 R, glm::vec3 T) {

	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= numDataPoints) return;

	pos[index] = R * pos[index] + T;
}

__global__ void kernTransform(int numDataPoints, const glm::vec3* in_pos, glm::vec3* out_pos, glm::mat3 R, glm::vec3 T) {

	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= numDataPoints) return;

	out_pos[index] = R * in_pos[index] + T;
}

#if CUDA_NAIVE
// GPU ICP Pipeline
void ICP::naiveGPUStep() {

	dim3 dataBlocksPerGrid((numDataPoints + blockSize - 1) / blockSize);
	// Find nearest correspondences
	kernSearchNearest << <dataBlocksPerGrid, blockSize >> > (numDataPoints, numModelPoints, dev_dataBuffer, dev_modelBuffer, dev_corrBuffer);
	cudaDeviceSynchronize();

	// Centralize
	glm::vec3 meanData = thrust::reduce(dev_dataBuffer, dev_dataBuffer + numDataPoints);
	glm::vec3 meanCorr = thrust::reduce(dev_corrBuffer, dev_corrBuffer + numDataPoints);
	meanData = meanData / static_cast<float>(numDataPoints);
	meanCorr = meanCorr / static_cast<float>(numDataPoints);

	kernCentralize << < dataBlocksPerGrid, blockSize >> > (numDataPoints, dev_dataBuffer, dev_centeredDataBuffer, meanData);
	kernCentralize << < dataBlocksPerGrid, blockSize >> > (numDataPoints, dev_corrBuffer, dev_centeredCorrBuffer, meanCorr);
	cudaDeviceSynchronize();

	// Calculating rotation and translations
	// PnP algorithm: minimizing A-RB equals to minimizing R-AB^T
	// Kabsch algorthm: Orthogonalize the rotation matrix with SVD: AB^T = USV^T, R = UV^T
	kernOuterProduct << <dataBlocksPerGrid, blockSize >> > (numDataPoints, dev_centeredDataBuffer, dev_centeredCorrBuffer, dev_ABtBuffer);
	cudaDeviceSynchronize();

	glm::mat3 ABt = thrust::reduce(dev_ABtBuffer, dev_ABtBuffer + numDataPoints);

	//compute SVD of ABt
	glm::mat3 R(0.0f), U(0.0f), S(0.0f), V(0.0f);
	glm::vec3 T(0.0f);

	matSVD(ABt, U, S, V);

	R = glm::transpose(U) * V; // Strange glm::mat column sequence >:(
	T = meanCorr - (R * meanData);

	// Update and draw
	kernTransform << < dataBlocksPerGrid, blockSize >> > (numDataPoints, dev_dataBuffer, R, T);
	cudaDeviceSynchronize();

	std::copy(&dev_dataBuffer[0], &dev_dataBuffer[0] + numDataPoints, &dev_pos[numModelPoints]);
	cudaDeviceSynchronize();
}

// KD-tree ICP step using the KDTree
void ICP::kdTreeGPUStep(KDTree& kdTree, Tree& tree) {
	dim3 dataBlocksPerGrid((numDataPoints + blockSize - 1) / blockSize);

	// Use kdTree to find nearest points for each data point
	for (int i = 0; i < numDataPoints; ++i) {
		glm::vec3 queryPoint = dev_dataBuffer[i];
		float query[3] = { queryPoint.x, queryPoint.y, queryPoint.z };

		size_t nearestIndex;
		float outDistSqr;
		nanoflann::KNNResultSet<float> resultSet(1);
		resultSet.init(&nearestIndex, &outDistSqr);

		kdTree.findNeighbors(resultSet, query, nanoflann::SearchParams(10));
		dev_corrBuffer[i] = tree.points[nearestIndex];
	}

	// Centralize
	glm::vec3 meanData = thrust::reduce(dev_dataBuffer, dev_dataBuffer + numDataPoints);
	glm::vec3 meanCorr = thrust::reduce(dev_corrBuffer, dev_corrBuffer + numDataPoints);
	meanData = meanData / static_cast<float>(numDataPoints);
	meanCorr = meanCorr / static_cast<float>(numDataPoints);

	kernCentralize << < dataBlocksPerGrid, blockSize >> > (numDataPoints, dev_dataBuffer, dev_centeredDataBuffer, meanData);
	kernCentralize << < dataBlocksPerGrid, blockSize >> > (numDataPoints, dev_corrBuffer, dev_centeredCorrBuffer, meanCorr);
	cudaDeviceSynchronize();

	// Calculating rotation and translations
	// PnP algorithm: minimizing A-RB equals to minimizing R-AB^T
	// Kabsch algorthm: Orthogonalize the rotation matrix with SVD: AB^T = USV^T, R = UV^T
	kernOuterProduct << <dataBlocksPerGrid, blockSize >> > (numDataPoints, dev_centeredDataBuffer, dev_centeredCorrBuffer, dev_ABtBuffer);
	cudaDeviceSynchronize();

	glm::mat3 ABt = thrust::reduce(dev_ABtBuffer, dev_ABtBuffer + numDataPoints);

	//compute SVD of ABt
	glm::mat3 R(0.0f), U(0.0f), S(0.0f), V(0.0f);
	glm::vec3 T(0.0f);

	matSVD(ABt, U, S, V);

	R = glm::transpose(U) * V; // Strange glm::mat column sequence >:(
	T = meanCorr - (R * meanData);

	// Update and draw
	kernTransform << < dataBlocksPerGrid, blockSize >> > (numDataPoints, dev_dataBuffer, R, T);
	cudaDeviceSynchronize();

	std::copy(&dev_dataBuffer[0], &dev_dataBuffer[0] + numDataPoints, &dev_pos[numModelPoints]);
	cudaDeviceSynchronize();
}
#endif

void ICP::goicpCPUStep(const GoICP &goicp, Matrix &prev_optR, Matrix &prev_optT, std::mutex &mtx) {
	dim3 dataBlocksPerGrid((numDataPoints + blockSize - 1) / blockSize);

	Matrix curR;
	Matrix curT;
	// main thread, simply check for update
	{
		// Lock mutex before accessing optR and optT
		std::lock_guard<std::mutex> lock(mtx);

		prev_optR = goicp.optR;
		prev_optT = goicp.optT;

		curR = goicp.curR;
		curT = goicp.curT;

	} // Unlock mutex (out of scope)

	if (!goicp.finished || prev_optR != goicp.optR || prev_optT != goicp.optT) {
		// Draw Optimal data cloud
		glm::mat3 R { prev_optR.val[0][0], prev_optR.val[0][1] ,prev_optR.val[0][2] ,
					  prev_optR.val[1][0] ,prev_optR.val[1][1] ,prev_optR.val[1][2] ,
					  prev_optR.val[2][0] ,prev_optR.val[2][1] ,prev_optR.val[2][2] };
		R = glm::transpose(R);

		glm::vec3 T { prev_optT.val[0][0], prev_optT.val[1][0], prev_optT.val[2][0] };

		kernTransform <<< dataBlocksPerGrid, blockSize >>> (numDataPoints, dev_dataBuffer, dev_optDataBuffer, R, T);
		cudaDeviceSynchronize();

		std::copy(&dev_optDataBuffer[0], &dev_optDataBuffer[0] + numDataPoints, &dev_pos[numModelPoints]);

		// Draw Current computing data cloud
		glm::mat3 Rc { curR.val[0][0], curR.val[0][1] ,curR.val[0][2] ,
					   curR.val[1][0] ,curR.val[1][1] ,curR.val[1][2] ,
					   curR.val[2][0] ,curR.val[2][1] ,curR.val[2][2] };
		Rc = glm::transpose(Rc);

		glm::vec3 Tc { curT.val[0][0], curT.val[1][0], curT.val[2][0] };

		kernTransform <<< dataBlocksPerGrid, blockSize >>> (numDataPoints, dev_dataBuffer, dev_curDataBuffer, Rc, Tc);
		cudaDeviceSynchronize();

		std::copy(&dev_curDataBuffer[0], &dev_curDataBuffer[0] + numDataPoints, &dev_pos[numModelPoints + numDataPoints]);
	} else {
		// clear 
		numPoints = numDataPoints + numModelPoints;
	}
}

