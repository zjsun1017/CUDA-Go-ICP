#include "kernel.h"
#include "goicp/jly_goicp.h"
#include "fgoicp/fgoicp.hpp"
#include "icp_kernel.h"

namespace ICP {
	void goicpCPUStep(const GoICP* goicp, Matrix& prev_optR, Matrix& prev_optT, std::mutex& mtx);
	void sgoicpCPUStep(const GoICP* goicp, Matrix& prev_optR, Matrix& prev_optT, std::mutex& mtx);
	void goicpGPUStep(const icp::FastGoICP* fgoicp, glm::mat3& prev_optR, glm::vec3& prev_optT, std::mutex& mtx);
}


