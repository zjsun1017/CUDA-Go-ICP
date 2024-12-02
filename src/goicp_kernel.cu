#include "goicp_kernel.h"
#include "icp_kernel.h"

extern int numPoints;
extern int numDataPoints;
extern int numModelPoints;

extern glm::vec3* dev_pos;
extern glm::vec3* dev_col;
extern glm::vec3* dev_dataBuffer;
extern glm::vec3* dev_optDataBuffer;
extern glm::vec3* dev_curDataBuffer;

extern std::priority_queue<RotNode> rcandidates;
extern float bestSSE;
extern glm::mat3 bestR;
extern glm::vec3 bestT;
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
        numPoints = numDataPoints + numModelPoints;
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

        //finished = goicp.finished;
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

    if (true) {
        // Draw Current computing data cloud

        kernTransform << < dataBlocksPerGrid, blockSize >> > (numDataPoints, dev_dataBuffer, dev_curDataBuffer, curR, curT);
        cudaDeviceSynchronize();
        checkCUDAError("Kern Transform");
        cudaMemcpy(dev_pos + numModelPoints + numDataPoints, dev_curDataBuffer, sizeof(glm::vec3) * numDataPoints, cudaMemcpyDeviceToDevice);
        //std::copy(dev_curDataBuffer, dev_curDataBuffer + numDataPoints, dev_pos + numModelPoints + numDataPoints);
    }
    else {
        // clear 
        std::copy(dev_optDataBuffer, dev_optDataBuffer + numDataPoints, dev_dataBuffer);
        numPoints = numDataPoints + numModelPoints;
    }
}


float branch_and_bound_SO3(StreamPool& stream_pool)
{
    // Initialize Rotation Nodes
    std::priority_queue<RotNode> rcandidates;
    RotNode rnode = RotNode(0.0f, 0.0f, 0.0f, 1.0f, 0.0f, bestSSE);
    rcandidates.push(std::move(rnode));

    //#define GROUND_TRUTH
#ifdef GROUND_TRUTH
    RotNode gt_rnode = RotNode(0.0625f, /*-1.0f / sqrt(2.0f)*/ -0.85f, 0.0625f, 0.0625f, 0.0f, M_INF);
    //RotNode gt_rnode = RotNode(0.0f, -1.0f / sqrt(2.0f), 0.0f, 0.0625f, 0.0f, M_INF);
    Logger() << "Ground Truth Rot:\n" << gt_rnode.q.R;
    auto [cub, t] = branch_and_bound_R3(gt_rnode, true, stream_pool);
    auto [clb, _] = branch_and_bound_R3(gt_rnode, false, stream_pool);
    Logger() << "Correct, ub: " << cub << " lb: " << clb << " t:\n\t" << t;

    IterativeClosestPoint3D icp3d(registration, pct, pcs, 1000, sse_threshold, gt_rnode.q.R, t);
    auto [icp_sse, icp_R, icp_t] = icp3d.run();
    Logger() << "ICP error: " << icp_sse
        << "\n\tRotation\n" << icp_R
        << "\n\tTranslation\t" << icp_t;

    return bestSSE;
#endif

    while (!rcandidates.empty())
    {
        RotNode rnode = rcandidates.top();
        rcandidates.pop();

        if (bestSSE - rnode.lb <= sse_threshold)
        {
            break;
        }

        // Spawn children RotNodes
        float span = rnode.span / 2.0f;
        for (char j = 0; j < 8; ++j)
        {
            if (span < 0.02f) { continue; }
            RotNode child_rnode(
                rnode.q.x - span + (j >> 0 & 1) * rnode.span,
                rnode.q.y - span + (j >> 1 & 1) * rnode.span,
                rnode.q.z - span + (j >> 2 & 1) * rnode.span,
                span, rnode.lb, rnode.ub
            );

            if (!child_rnode.overlaps_SO3()) { continue; }
            if (!child_rnode.q.in_SO3())
            {
                rcandidates.push(std::move(child_rnode));
                continue;
            }

            // BnB in R3 
            auto [ub, best_t] = branch_and_bound_R3(child_rnode, true, stream_pool);

            if (ub < bestSSE)
            {
                auto [icp_sse, icp_R, icp_t] = computeICP(child_rnode.q.R, best_t);

                if (icp_sse < ub)
                {
                    bestSSE = icp_sse;
                    bestR = icp_R;
                    bestT = icp_t;
                }
                else
                {
                    bestSSE = ub;
                    bestR = child_rnode.q.R;
                    bestT = best_t;
                }
                Logger(LogLevel::Debug) << "New best error: " << bestSSE << "\n"
                    << "\tRotation:\n" << bestR << "\n"
                    << "\tTranslation: " << bestT;
            }

            auto [lb, _] = branch_and_bound_R3(child_rnode, false, stream_pool);
            Logger() << "ub: " << ub
                << "\tlb: " << lb;

            if (lb >= bestSSE) { continue; }
            child_rnode.lb = lb;
            child_rnode.ub = ub;

            rcandidates.push(std::move(child_rnode));
        }
    }
    return bestSSE;
}

ResultBnBR3 branch_and_bound_R3(RotNode& rnode, bool fix_rot, StreamPool& stream_pool)
{
    float best_error = bestSSE;
    glm::vec3 best_t{ 0.0f };

    size_t count = 0;

    // Initialize queue for TransNodes
    std::priority_queue<TransNode> tcandidates;

    TransNode init_tnode = TransNode(0.0f, 0.0f, 0.0f, 1.0f, 0.0f, rnode.ub);
    tcandidates.push(std::move(init_tnode));

    while (!tcandidates.empty())
    {
        std::vector<TransNode> tnodes;

        if (best_error - tcandidates.top().lb < sse_threshold) { break; }
        // Get a batch
        while (!tcandidates.empty() && tnodes.size() < 16)
        {
            auto tnode = tcandidates.top();
            tcandidates.pop();
            if (tnode.lb < best_error)
            {
                tnodes.push_back(std::move(tnode));
            }
        }

        count += tnodes.size();

        // Compute lower/upper bounds
        auto [lb, ub] = compute_sse_error(rnode, tnodes, fix_rot, stream_pool);

        // *Fix rotation* to compute rotation *lower bound*
        // Get min upper bound of this batch to update best SSE
        size_t idx_min = std::distance(std::begin(ub), std::min_element(std::begin(ub), std::end(ub)));
        if (ub[idx_min] < best_error)
        {
            best_error = ub[idx_min];
            best_t = tnodes[idx_min].t;
        }

        // Examine translation lower bounds
        for (size_t i = 0; i < tnodes.size(); ++i)
        {
            // Eliminate those with lower bound >= best SSE
            if (lb[i] >= best_error) { continue; }

            TransNode& tnode = tnodes[i];
            // Stop if the span is small enough
            if (tnode.span < 0.2f) { continue; }  // TODO: use config threshold

            float span = tnode.span / 2.0f;
            // Spawn 8 children
            for (char j = 0; j < 8; ++j)
            {
                TransNode child_tnode(
                    tnode.t.x - span + (j >> 0 & 1) * tnode.span,
                    tnode.t.y - span + (j >> 1 & 1) * tnode.span,
                    tnode.t.z - span + (j >> 2 & 1) * tnode.span,
                    span, lb[i], ub[i]
                );
                tcandidates.push(std::move(child_tnode));
            }
        }

    }

    Logger() << count << " TransNodes searched. Inner BnB finished";

    return { best_error, best_t };
}

// Get the top rotation node from the priority queue
bool getNextRotationNode(RotNode& currentNode) {
    if (rcandidates.empty()) return false;

    currentNode = rcandidates.top();
    rcandidates.pop();

    // Stop if no further improvement is possible
    if (bestSSE - currentNode.lb <= sse_threshold) {
        return false;
    }

    return true;
}

// Process the current rotation node and spawn children
void processRotationNode(RotNode& rnode, StreamPool& stream_pool) {
    float span = rnode.span / 2.0f;

    for (char j = 0; j < 8; ++j) {
        if (span < 0.02f) { continue; }
        RotNode child_rnode(
            rnode.q.x - span + (j >> 0 & 1) * rnode.span,
            rnode.q.y - span + (j >> 1 & 1) * rnode.span,
            rnode.q.z - span + (j >> 2 & 1) * rnode.span,
            span, rnode.lb, rnode.ub
        );

        if (!child_rnode.overlaps_SO3()) { continue; }
        if (!child_rnode.q.in_SO3()) {
            rcandidates.push(std::move(child_rnode));
            continue;
        }

        // Compute bounds for translation (R3 space)
        auto [ub, best_t] = branch_and_bound_R3(child_rnode, true, stream_pool);

        if (ub < bestSSE) {
            auto [icp_sse, icp_R, icp_t] = computeICP(child_rnode.q.R, best_t);

            if (icp_sse < ub) {
                bestSSE = icp_sse;
                bestR = icp_R;
                bestT = icp_t;
            }
            else {
                bestSSE = ub;
                bestR = child_rnode.q.R;
                bestT = best_t;
            }
        }

        auto [lb, _] = branch_and_bound_R3(child_rnode, false, stream_pool);

        if (lb >= bestSSE) { continue; }
        child_rnode.lb = lb;
        child_rnode.ub = ub;

        rcandidates.push(std::move(child_rnode));
    }
}

// Single step of branch and bound SO3
bool ICP::branchAndBoundSO3Step(StreamPool& stream_pool) {
    RotNode currentNode(0.0f, 0.0f, 0.0f, 1.0f, 0.0f, bestSSE);
    if (!getNextRotationNode(currentNode)) {
        return false; // Termination condition
    }

    processRotationNode(currentNode, stream_pool);
    return true;
}

