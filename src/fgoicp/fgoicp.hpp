#ifndef FGOICP_HPP
#define FGOICP_HPP

#include "fgoicp_common.hpp"
#include "registration.hpp"
#include <iostream>
#include <mutex>

namespace icp
{
    class FastGoICP
    {
    public:
        FastGoICP(std::vector<glm::vec3>& pct, std::vector<glm::vec3>& pcs, float mse_threshold, std::mutex& mtx) : 
            mtx(mtx),
            pcs(pcs), pct(pct),
            ns(pcs.size()),
            nt(pct.size()),
            registration{pct, nt, pcs, ns},
            max_iter(10), best_sse(M_INF), 
            best_translation(0.0f),
            mse_threshold(mse_threshold), // init *mean* squared error threshold 
            sse_threshold(ns * mse_threshold),    // init *sum* of squared error threshold
            stream_pool(32),
            curR(1.0f), optR(1.0f),
            curT(0.0f), optT(0.0f),
            finished(false)
        {};

        ~FastGoICP() {}

        void run();

        float get_best_error() const { return best_sse; }

    private:
        std::mutex& mtx;
        // Data
        PointCloud pcs;  // source point cloud
        PointCloud pct;  // target point cloud
        size_t ns, nt; // number of source/target points

        // Registration object for ICP and error computing
        Registration registration;

        // Runtime variables
        size_t max_iter;
        float best_sse;
        glm::mat3 best_rotation;
        glm::vec3 best_translation;

        // MSE threshold depends on the source point cloud stats.
        // If we normalize the source point cloud into a standard cube,
        // The MSE threshold can be specified without considering 
        // the point cloud stats.
        float mse_threshold;
        // SSE threshold is the summed error threshold,
        // the registration is considered converged if SSE threshold is reached.
        // If no trimming, sse_threshold = ns * mse_threshold
        float sse_threshold;

        // CUDA stream pool
        StreamPool stream_pool;

    public:
        // For visualization
        glm::mat3 curR, optR;
        glm::vec3 curT, optT;
        bool finished;

    private:
        using ResultBnBR3 = std::tuple<float, glm::vec3>;

        /**
         * @brief Perform branch-and-bound algorithm in Rotation Space SO(3)
         *
         * @return float
         */
        float branch_and_bound_SO3();

        /**
         * @brief Perform branch-and-bound algorithm in Translation Space R(3)
         *
         * @param
         * @return ResultBnBR3
         */
        ResultBnBR3 branch_and_bound_R3(RotNode &rnode, bool fix_rot);
    };
}

#endif // FGOICP_HPP