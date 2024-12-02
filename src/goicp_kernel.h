#include "kernel.h"
#include "goicp/jly_goicp.h"
#include "fgoicp/fgoicp.hpp"
#include <mutex>

using ResultBnBR3 = std::tuple<float, glm::vec3>;

namespace ICP {
	void goicpCPUStep(const GoICP& goicp, Matrix& prev_optR, Matrix& prev_optT, std::mutex& mtx);
	void sgoicpCPUStep(const GoICP& goicp, Matrix& prev_optR, Matrix& prev_optT, std::mutex& mtx);
	void goicpGPUStep(const icp::FastGoICP* fgoicp, glm::mat3& prev_optR, glm::vec3& prev_optT, std::mutex& mtx);
	bool branchAndBoundSO3Step(StreamPool& stream_pool);
}

float branch_and_bound_SO3(StreamPool& stream_pool);
ResultBnBR3 branch_and_bound_R3(RotNode& rnode, bool fix_rot, StreamPool& stream_pool);


