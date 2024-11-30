#include "kernel.h"
#include "goicp/jly_goicp.h"
#include <mutex>

using BoundsResult_t = std::tuple<std::vector<float>, std::vector<float>>;

namespace ICP {
	void goicpCPUStep(const GoICP& goicp, Matrix& prev_optR, Matrix& prev_optT, std::mutex& mtx);
	void goicpGPUStep();
}

//========================================================================================
    //                                    CUDA Stream Pool
    //========================================================================================
class StreamPool
{
public:
    explicit StreamPool(size_t size) : streams(size)
    {
        for (auto& stream : streams)
        {
            cudaStreamCreate(&stream);
        }
    }

    ~StreamPool()
    {
        for (auto& stream : streams)
        {
            cudaStreamDestroy(stream);
        }
    }

    cudaStream_t getStream(size_t index) const
    {
        return streams[index % streams.size()];
    }

private:
    std::vector<cudaStream_t> streams;
};

float compute_sse_error(glm::mat3 R, glm::vec3 t);
BoundsResult_t compute_sse_error(RotNode& rnode, std::vector<TransNode>& tnodes, bool fix_rot, StreamPool& stream_pool);
