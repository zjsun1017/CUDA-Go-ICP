#include "goicp_kernel.h"

extern int numPoints;
extern int numDataPoints;
extern int numModelPoints;

extern glm::vec3* dev_pos;
extern glm::vec3* dev_col;
extern glm::vec3* dev_dataBuffer;
extern glm::vec3* dev_optDataBuffer;
extern glm::vec3* dev_curDataBuffer;

extern float sse_threshold;
extern bool goicp_finished;

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

void ICP::sgoicpCPUStep(const GoICP& goicp, Matrix& prev_optR, Matrix& prev_optT, std::mutex& mtx) {
    dim3 dataBlocksPerGrid((numDataPoints + blockSize - 1) / blockSize);

    Matrix curR;
    Matrix curT;

    bool updated;
    float currentError = FLT_MAX;

    // Main thread, simply check for update
    {
        // Lock mutex before accessing optR and optT
        std::lock_guard<std::mutex> lock(mtx);

        //finished = goicp.finished;
        updated = (prev_optR != goicp.optR || prev_optT != goicp.optT);
        
        goicp_finished = goicp.finished;
        currentError = goicp.optError;

        prev_optR = goicp.optR;
        prev_optT = goicp.optT;

        curR = goicp.curR;
        curT = goicp.curT;

    } // Unlock mutex (out of scope)

    if (updated || currentError <= sse_threshold) {
        // Draw Optimal data cloud
        glm::mat3 R{ prev_optR.val[0][0], prev_optR.val[0][1] ,prev_optR.val[0][2] ,
                        prev_optR.val[1][0] ,prev_optR.val[1][1] ,prev_optR.val[1][2] ,
                        prev_optR.val[2][0] ,prev_optR.val[2][1] ,prev_optR.val[2][2] };
        R = glm::transpose(R);

        glm::vec3 T{ prev_optT.val[0][0], prev_optT.val[1][0], prev_optT.val[2][0] };

        kernTransform << < dataBlocksPerGrid, blockSize >> > (numDataPoints, dev_dataBuffer, dev_optDataBuffer, R, T);
        cudaDeviceSynchronize();

        if (currentError <= sse_threshold)
        {
            Logger(LogLevel::Info) <<"Optimal error " << currentError << " smaller than threshold " << sse_threshold <<   ", Terminate Go-ICP, switch back to ICP. ";
            goicp_finished = true;
        }

        std::copy(&dev_optDataBuffer[0], &dev_optDataBuffer[0] + numDataPoints, &dev_pos[numModelPoints]);
    }

    if (!goicp_finished) {
        // Draw Current computing data cloud
        glm::mat3 Rc{ curR.val[0][0], curR.val[0][1] ,curR.val[0][2] ,
                       curR.val[1][0] ,curR.val[1][1] ,curR.val[1][2] ,
                       curR.val[2][0] ,curR.val[2][1] ,curR.val[2][2] };
        Rc = glm::transpose(Rc);

        glm::vec3 Tc{ curT.val[0][0], curT.val[1][0], curT.val[2][0] };

        kernTransform << < dataBlocksPerGrid, blockSize >> > (numDataPoints, dev_dataBuffer, dev_curDataBuffer, Rc, Tc);
        cudaDeviceSynchronize();
        std::copy(dev_curDataBuffer, dev_curDataBuffer + numDataPoints, dev_pos + numModelPoints + numDataPoints);
    }
    else {
        // clear 
        std::copy(dev_optDataBuffer, dev_optDataBuffer + numDataPoints, dev_dataBuffer);
        numPoints = numDataPoints + numModelPoints - 1;
    }
}


void ICP::goicpGPUStep(const icp::FastGoICP* fgoicp, glm::mat3& prev_optR, glm::vec3& prev_optT, std::mutex& mtx) {
    dim3 dataBlocksPerGrid((numDataPoints + blockSize - 1) / blockSize);

    glm::mat3 curR;
    glm::vec3 curT;

    bool updated;
    float currentError = FLT_MAX;

    // Main thread, simply check for update
    {
        // Lock mutex before accessing optR and optT
        std::lock_guard<std::mutex> lock(mtx);

        goicp_finished = fgoicp->finished;
        updated = (prev_optR != fgoicp->optR || prev_optT != fgoicp->optT);

        currentError = fgoicp->get_best_error();

        prev_optR = fgoicp->optR;
        prev_optT = fgoicp->optT;

        curR = fgoicp->curR;
        curT = fgoicp->curT;

    } // Unlock mutex (out of scope)

    if (updated || currentError <= sse_threshold) {
        // Draw Optimal data cloud
        kernTransform << < dataBlocksPerGrid, blockSize >> > (numDataPoints, dev_dataBuffer, dev_optDataBuffer, prev_optR, prev_optT);
        cudaDeviceSynchronize();

        //std::copy(dev_optDataBuffer, dev_optDataBuffer + numDataPoints, dev_pos + numModelPoints);
        cudaMemcpy(dev_pos + numModelPoints, dev_optDataBuffer, sizeof(glm::vec3) * numDataPoints, cudaMemcpyDeviceToDevice);
    }

    if (!goicp_finished) {
        // Draw Current computing data cloud

        kernTransform << < dataBlocksPerGrid, blockSize >> > (numDataPoints, dev_dataBuffer, dev_curDataBuffer, curR, curT);
        cudaDeviceSynchronize();
        checkCUDAError("Kern Transform");
        cudaMemcpy(dev_pos + numModelPoints + numDataPoints, dev_curDataBuffer, sizeof(glm::vec3) * numDataPoints, cudaMemcpyDeviceToDevice);
        //std::copy(dev_curDataBuffer, dev_curDataBuffer + numDataPoints, dev_pos + numModelPoints + numDataPoints);
    }
    else {
        // clear 
        cudaMemcpy(dev_dataBuffer, dev_optDataBuffer, numDataPoints * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
        numPoints = numDataPoints + numModelPoints - 1;
    }
}
