#include "goicp_kernel.h"

extern int numPoints;
extern int numDataPoints;
extern int numModelPoints;

extern glm::vec3* dev_pos;
extern glm::vec3* dev_col;
extern glm::vec3* dev_dataBuffer;
extern glm::vec3* dev_optDataBuffer;
extern glm::vec3* dev_curDataBuffer;


__global__ void kernTransform(int numDataPoints, const glm::vec3* in_pos, glm::vec3* out_pos, glm::mat3 R, glm::vec3 T) {

	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= numDataPoints) return;

	out_pos[index] = R * in_pos[index] + T;
}

void ICP::goicpCPUStep(const GoICP& goicp, Matrix& prev_optR, Matrix& prev_optT, std::mutex& mtx) {
	dim3 dataBlocksPerGrid((numDataPoints + blockSize - 1) / blockSize);

	Matrix curR;
	Matrix curT;

	bool finished;
	bool updated;
	// Main thread, simply check for update
	{
		// Lock mutex before accessing optR and optT
		std::lock_guard<std::mutex> lock(mtx);

		finished = goicp.finished;
		updated = (prev_optR != goicp.optR || prev_optT != goicp.optT);

		prev_optR = goicp.optR;
		prev_optT = goicp.optT;

		curR = goicp.curR;
		curT = goicp.curT;

	} // Unlock mutex (out of scope)

	if (updated) {
		// Draw Optimal data cloud
		glm::mat3 R{ prev_optR.val[0][0], prev_optR.val[0][1] ,prev_optR.val[0][2] ,
						prev_optR.val[1][0] ,prev_optR.val[1][1] ,prev_optR.val[1][2] ,
						prev_optR.val[2][0] ,prev_optR.val[2][1] ,prev_optR.val[2][2] };
		R = glm::transpose(R);

		glm::vec3 T{ prev_optT.val[0][0], prev_optT.val[1][0], prev_optT.val[2][0] };

		kernTransform << < dataBlocksPerGrid, blockSize >> > (numDataPoints, dev_dataBuffer, dev_optDataBuffer, R, T);
		cudaDeviceSynchronize();

		std::copy(&dev_optDataBuffer[0], &dev_optDataBuffer[0] + numDataPoints, &dev_pos[numModelPoints]);
	}

	if (!finished) {
		// Draw Current computing data cloud
		glm::mat3 Rc{ curR.val[0][0], curR.val[0][1] ,curR.val[0][2] ,
					   curR.val[1][0] ,curR.val[1][1] ,curR.val[1][2] ,
					   curR.val[2][0] ,curR.val[2][1] ,curR.val[2][2] };
		Rc = glm::transpose(Rc);

		glm::vec3 Tc{ curT.val[0][0], curT.val[1][0], curT.val[2][0] };

		kernTransform << < dataBlocksPerGrid, blockSize >> > (numDataPoints, dev_dataBuffer, dev_curDataBuffer, Rc, Tc);
		cudaDeviceSynchronize();
		std::copy(&dev_curDataBuffer[0], &dev_curDataBuffer[0] + numDataPoints, &dev_pos[numModelPoints + numDataPoints]);
	}
	else {
		// clear 
		numPoints = numDataPoints + numModelPoints;
	}
}