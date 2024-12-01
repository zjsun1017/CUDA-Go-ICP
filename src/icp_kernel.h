#pragma once
#include "kernel.h"
#include "goicp_kernel.h"

void matSVD(glm::mat3& ABt, glm::mat3& U, glm::mat3& S, glm::mat3& V);
using BoundsResult_t = std::tuple<std::vector<float>, std::vector<float>>;
using Result_t = std::tuple<float, glm::mat3, glm::vec3>;

namespace ICP {
	void CPUStep(std::vector<glm::vec3>& dataBuffer, std::vector<glm::vec3>& modelBuffer);
	void naiveGPUStep();
	void kdTreeGPUStep(KDTree& kdTree, PointCloudAdaptor& tree, FlattenedKDTree* dev_fkdt);
}

Result_t computeICP(glm::mat3 R, glm::vec3 t);
float compute_sse_error(glm::mat3 R, glm::vec3 t);
BoundsResult_t compute_sse_error(RotNode& rnode, std::vector<TransNode>& tnodes, bool fix_rot, StreamPool& stream_pool);
