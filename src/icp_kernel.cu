#define GLM_FORCE_CUDA
#include "icp_kernel.h"
#include "goicp_kernel.h"
#include "svd3.h"
#include <device_atomic_functions.h>
#include <cuda_runtime.h>

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

extern float* dev_errors;
extern std::vector<float> sse_rot_ub_trans_ub;
extern std::vector<float> sse_rot_ub_trans_lb;
extern float* dev_rot_ub_trans_ub;
extern float* dev_rot_ub_trans_lb;

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

__global__ void kernKDSearchNearest(int numDataPoints, glm::vec3* dataBuffer,
	glm::vec3* modelBuffer, glm::vec3* corrBuffer,
	FlattenedKDTree* fkdt, float* minDists, size_t* minIndices)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= numDataPoints) { return; }

	glm::vec3 query = dataBuffer[index];
	fkdt->find_nearest_neighbor(query, minDists[index], minIndices[index]);

	corrBuffer[index] = modelBuffer[minIndices[index]];
}

__global__ void kernTestKDTreeLookUp(int N, glm::vec3 query, FlattenedKDTree* fkdt, float* min_dists, size_t* min_indices)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= N) { return; }

	fkdt->find_nearest_neighbor(query, *min_dists, *min_indices);
}

__global__ void kernResetFloatBuffer(int N, float* intBuffer, float value) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < N) {
		intBuffer[index] = value;
	}
}


// GPU ICP Pipeline
void ICP::naiveGPUStep() {

	dim3 dataBlocksPerGrid((numDataPoints + blockSize - 1) / blockSize);
	// Find nearest correspondences
	kernSearchNearest << <dataBlocksPerGrid, blockSize >> > (numDataPoints, numModelPoints, dev_dataBuffer, dev_modelBuffer, dev_corrBuffer);
	cudaDeviceSynchronize();

	// Centralize
	glm::vec3 meanData = thrust::reduce(thrust::device, dev_dataBuffer, dev_dataBuffer + numDataPoints);
	glm::vec3 meanCorr = thrust::reduce(thrust::device, dev_corrBuffer, dev_corrBuffer + numDataPoints);
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
void ICP::kdTreeGPUStep(KDTree& kdTree, PointCloudAdaptor& tree, FlattenedKDTree* fkdt) {
	dim3 dataBlocksPerGrid((numDataPoints + blockSize - 1) / blockSize);

	/*Use kdTree to find nearest points for each data point*/
	//for (int i = 0; i < numDataPoints; ++i) {
	//	glm::vec3 queryPoint = dev_dataBuffer[i];
	//	float query[3] = { queryPoint.x, queryPoint.y, queryPoint.z };

	//	size_t nearestIndex;
	//	float outDistSqr;
	//	nanoflann::KNNResultSet<float> resultSet(1);
	//	resultSet.init(&nearestIndex, &outDistSqr);

	//	kdTree.findNeighbors(resultSet, query, nanoflann::SearchParameters(10, true));
	//	dev_corrBuffer[i] = tree.points[nearestIndex];
	//}

	kernResetFloatBuffer << < dataBlocksPerGrid, blockSize >> > (numDataPoints, dev_minDists, FLT_MAX);
	cudaDeviceSynchronize();

	kernKDSearchNearest << < dataBlocksPerGrid, blockSize >> > (numDataPoints, dev_dataBuffer,
		dev_modelBuffer, dev_corrBuffer, dev_fkdt, dev_minDists, dev_minIndices);
	cudaDeviceSynchronize();

	/*kernTestKDTreeLookUp << < dataBlocksPerGrid, blockSize >> > (numDataPoints, glm::vec3(1.0f), dev_fkdt, dev_minDists, dev_minIndices);*/
	//cudaDeviceSynchronize();

	// Centralize
	glm::vec3 meanData = thrust::reduce(thrust::device, dev_dataBuffer, dev_dataBuffer + numDataPoints);
	glm::vec3 meanCorr = thrust::reduce(thrust::device, dev_corrBuffer, dev_corrBuffer + numDataPoints);
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

	glm::mat3 ABt = thrust::reduce(thrust::device, dev_ABtBuffer, dev_ABtBuffer + numDataPoints);

	//compute SVD of ABt
	glm::mat3 R(0.0f), U(0.0f), S(0.0f), V(0.0f);
	glm::vec3 T(0.0f);

	matSVD(ABt, U, S, V);

	R = glm::transpose(U) * V; // Strange glm::mat column sequence >:(
	T = meanCorr - (R * meanData);

	// Update and draw
	kernTransform <<< dataBlocksPerGrid, blockSize >> > (numDataPoints, dev_dataBuffer, R, T);
	cudaDeviceSynchronize();

	std::copy(&dev_dataBuffer[0], &dev_dataBuffer[0] + numDataPoints, &dev_pos[numModelPoints]);
	cudaDeviceSynchronize();
}

__global__ void kernFindNearestNeighbor(int N, glm::mat3 R, glm::vec3 t, const glm::vec3* dev_pcs, const FlattenedKDTree* d_fkdt, Correspondence* dev_corrs)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= N) { return; }

	glm::vec3 query_point = R * dev_pcs[index] + t;

	size_t nearest_index = 0;

	float distance_squared = 1e+10f;
	d_fkdt->find_nearest_neighbor(query_point, distance_squared, nearest_index);

	dev_corrs[index].dist_squared = distance_squared;
	dev_corrs[index].idx_s = index;
	dev_corrs[index].idx_t = nearest_index;
	dev_corrs[index].ps_transformed = query_point;
}

__global__ void kernSetRegistrationMatrices(int N, Rotation q, glm::vec3 t, const glm::vec3* dev_pcs, const glm::vec3* dev_pct, const Correspondence* dev_corrs, float* dev_mat_pcs, float* dev_mat_pct)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= N) { return; }

	Correspondence corr = dev_corrs[index];
	glm::vec3 pt = dev_pct[corr.idx_t];
	glm::vec3 ps = corr.ps_transformed;

	size_t mat_idx = index * 3;

	dev_mat_pct[mat_idx] = pt.x;
	dev_mat_pct[mat_idx + 1] = pt.y;
	dev_mat_pct[mat_idx + 2] = pt.z;

	dev_mat_pcs[mat_idx] = ps.x;
	dev_mat_pcs[mat_idx + 1] = ps.y;
	dev_mat_pcs[mat_idx + 2] = ps.z;
}


//============================================
//            Flattened k-d tree
//============================================

FlattenedKDTree::FlattenedKDTree(const KDTree& kdt, const std::vector<glm::vec3>& pct) {
	// Transfer point cloud and KDTree accumulator to device
	d_pct = pct; // Copy point cloud to device
	d_vAcc = kdt.vAcc_;

	// Flatten the KDTree and transfer to device
	size_t currentIndex = 0;
	std::vector<ArrayNode> tempArray; // Temporary host vector for tree flattening
	flatten_KDTree(kdt.root_node_, tempArray, currentIndex);

	// Copy flattened KDTree to device
	d_array = tempArray;
}

void FlattenedKDTree::flatten_KDTree(const KDTree::Node* root, std::vector<ArrayNode>& array, size_t& currentIndex) {
	if (root == nullptr) return;

	size_t index = currentIndex++;
	if (index >= array.size()) {
		array.resize(index + 1);
	}

	if (root->child1 == nullptr && root->child2 == nullptr) {
		// Leaf node
		array[index].is_leaf = true;
		array[index].data.leaf.left = root->node_type.lr.left;
		array[index].data.leaf.right = root->node_type.lr.right;
	}
	else {
		// Non-leaf node
		array[index].is_leaf = false;
		array[index].data.nonleaf.divfeat = root->node_type.sub.divfeat;
		array[index].data.nonleaf.divlow = root->node_type.sub.divlow;
		array[index].data.nonleaf.divhigh = root->node_type.sub.divhigh;

		// Recursively flatten left and right child nodes
		size_t child1Index = currentIndex;
		flatten_KDTree(root->child1, array, currentIndex);
		array[index].data.nonleaf.child1 = child1Index;

		size_t child2Index = currentIndex;
		flatten_KDTree(root->child2, array, currentIndex);
		array[index].data.nonleaf.child2 = child2Index;
	}
}

__device__ float distance_squared(const glm::vec3 p1, const glm::vec3 p2)
{
	float dx = p1.x - p2.x;
	float dy = p1.y - p2.y;
	float dz = p1.z - p2.z;
	return dx * dx + dy * dy + dz * dz;
}

__device__ void FlattenedKDTree::find_nearest_neighbor(const glm::vec3 query, float& best_dist, size_t& best_idx) const {
	size_t stack[64]; // Stack for non-recursive traversal
	int stack_ptr = 0;

	stack[stack_ptr++] = 0; // Root node index

	while (stack_ptr > 0) {
		size_t index = stack[--stack_ptr];

		if (index >= d_array.size()) continue;
		const ArrayNode& node = d_array[index];

		if (node.is_leaf) {
			// Process all points in the leaf node
			size_t left = node.data.leaf.left;
			size_t right = node.data.leaf.right;

			if (left >= d_pct.size() || right >= d_pct.size()) continue;

			for (size_t i = left; i <= right; i++) {
				float dist = distance_squared(query, d_pct[d_vAcc[i]]);
				if (dist < best_dist) {
					/*printf("%f\n", best_dist);*/
					best_dist = dist;
					best_idx = d_vAcc[i];
				}
			}
		}
		else {
			// Process non-leaf node: determine near and far child
			int axis = node.data.nonleaf.divfeat;
			float diff = query[axis] - node.data.nonleaf.divlow;

			size_t nearChild = diff < 0 ? node.data.nonleaf.child1 : node.data.nonleaf.child2;
			size_t farChild = diff < 0 ? node.data.nonleaf.child2 : node.data.nonleaf.child1;

			stack[stack_ptr++] = nearChild;
			if (diff * diff < best_dist) {
				stack[stack_ptr++] = farChild;
			}
		}
	}
}

__global__ void kernComputeClosestError(int N, glm::mat3 R, glm::vec3 t, const glm::vec3* dataBuffer, const FlattenedKDTree* d_fkdt, float* d_errors)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= N) { return; }

	glm::vec3 source_point = dataBuffer[index];
	glm::vec3 query_point = R * source_point + t;

	size_t nearest_index = 0;
	float distance_squared = FLT_MAX;
	d_fkdt->find_nearest_neighbor(query_point, distance_squared, nearest_index);

	d_errors[index] = distance_squared;
}

__global__ void kernComputeBounds(int N, RotNode rnode, TransNode tnode, bool fix_rot, const glm::vec3* dev_dataBuffer, const FlattenedKDTree* d_fkdt, float* dev_rot_ub_trans_ub, float* dev_rot_ub_trans_lb)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= N) { return; }

	glm::vec3 source_point = dev_dataBuffer[index];
	float trans_uncertain_radius = M_SQRT3 * tnode.span;
	glm::vec3 query_point = rnode.q.R * source_point + tnode.t;

	float rot_uncertain_radius;
	if (!fix_rot)
	{
		float radius = source_point.x * source_point.x +
			source_point.y * source_point.y +
			source_point.z * source_point.z;
		float half_angle = rnode.span * M_SQRT3 * M_PI / 2.0f;  // TODO: Need examination, since we are using quaternions
		rot_uncertain_radius = 2.0f * radius * sin(half_angle);
	}

	size_t nearest_index = 0;
	float distance_squared = FLT_MAX;
	d_fkdt->find_nearest_neighbor(query_point, distance_squared, nearest_index);

	float distance = sqrt(distance_squared);
	if (!fix_rot)
	{
		distance -= rot_uncertain_radius;
	}

	dev_rot_ub_trans_ub[index] = distance > 0.0f ? distance * distance : 0.0f;


	float rot_ub_trans_lb = distance - trans_uncertain_radius;
	rot_ub_trans_lb = rot_ub_trans_lb > 0.0f ? rot_ub_trans_lb * rot_ub_trans_lb : 0.0f;
	dev_rot_ub_trans_lb[index] = rot_ub_trans_lb;
}

float compute_sse_error(glm::mat3 R, glm::vec3 t)
{
	size_t block_size = 32;
	dim3 dataBlocksPerGrid((numDataPoints + block_size - 1) / block_size);
	kernComputeClosestError << <dataBlocksPerGrid, blockSize >> > (numDataPoints, R, t, dev_dataBuffer, dev_fkdt, dev_errors);
	cudaDeviceSynchronize();

	float sse_error = thrust::reduce(thrust::device, dev_errors, dev_errors + numDataPoints);
	return sse_error;
}

BoundsResult_t compute_sse_error(RotNode& rnode, std::vector<TransNode>& tnodes, bool fix_rot, StreamPool& stream_pool)
{
	size_t num_transforms = tnodes.size();
	std::vector<float> sse_rot_ub_trans_ub(num_transforms);
	std::vector<float> sse_rot_ub_trans_lb(num_transforms);

	// Kernel launching parameters
	size_t block_size = 32;
	dim3 dataBlocksPerGrid((numDataPoints + block_size - 1) / block_size);

	// Launch kernel for each (R, t) pair on separate streams
	for (size_t i = 0; i < num_transforms; ++i) {
		// Get the appropriate stream from the stream pool
		cudaStream_t stream = stream_pool.getStream(i);

		// Launch the kernel with each (R, t) on a different stream
		kernComputeBounds << <dataBlocksPerGrid, block_size, 0, stream >> > (
			numDataPoints, rnode, tnodes[i], fix_rot, dev_dataBuffer, dev_fkdt, dev_rot_ub_trans_ub + i * numDataPoints, dev_rot_ub_trans_lb + i * numDataPoints);
	}

	// Reduce the lower/upper bounds for each pair
	for (size_t i = 0; i < num_transforms; ++i) {
		// Thrust reduce launching parameters
		auto thrust_policy = thrust::cuda::par.on(stream_pool.getStream(i));
		sse_rot_ub_trans_ub[i] = thrust::reduce(thrust_policy, dev_rot_ub_trans_ub + i * numDataPoints, dev_rot_ub_trans_ub + (i + 1) * numDataPoints);
		sse_rot_ub_trans_lb[i] = thrust::reduce(thrust_policy, dev_rot_ub_trans_ub + i * numDataPoints, dev_rot_ub_trans_ub + (i + 1) * numDataPoints);
	}

	cudaDeviceSynchronize();
	return { sse_rot_ub_trans_lb, sse_rot_ub_trans_ub };
}