#include "kernel.h"
#include "goicp/jly_goicp.h"
#include <mutex>

namespace ICP {
	void goicpCPUStep(const GoICP& goicp, Matrix& prev_optR, Matrix& prev_optT, std::mutex& mtx);
	void goicpGPUStep();
}