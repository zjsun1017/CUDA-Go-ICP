#pragma once
#include "kernel.h"
#include "goicp/jly_goicp.h"
#include <mutex>

void matSVD(glm::mat3& ABt, glm::mat3& U, glm::mat3& S, glm::mat3& V);

namespace ICP {
	void CPUStep(std::vector<glm::vec3>& dataBuffer, std::vector<glm::vec3>& modelBuffer);
	void naiveGPUStep();
	void kdTreeGPUStep(KDTree& kdTree, Tree& tree);
	void goicpCPUStep(const GoICP& goicp, Matrix& prev_optR, Matrix& prev_optT, std::mutex& mtx);
}
