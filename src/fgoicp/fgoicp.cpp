#include "fgoicp.hpp"
#include <iostream>
#include <queue>
#include <tuple>
#include "icp3d.hpp"

namespace icp
{
    void FastGoICP::run()
    {
        IterativeClosestPoint3D icp3d(registration, pct, pcs, 100, sse_threshold, glm::mat3(1.0f), glm::vec3(0.0f));
        auto [icp_sse, icp_R, icp_t] = icp3d.run();
        optR = icp_R;
        optT = icp_t;
        best_sse = icp_sse;
        Logger(LogLevel::Info) << "Initial ICP best error: " << icp_sse
                               << "\n\tRotation:\n" << icp_R
                               << "\n\tTranslation: " << icp_t;

        branch_and_bound_SO3();
        Logger(LogLevel::Info) << "Searching over! Best Error: " << best_sse
                               << "\n\tRotation:\n" << best_rotation
                               << "\n\tTranslation: " << best_translation;
        finished = true;
    }

    float FastGoICP::branch_and_bound_SO3()
    {
        // Initialize Rotation Nodes
        std::priority_queue<RotNode> rcandidates;
        RotNode rnode = RotNode(0.0f, 0.0f, 0.0f, 1.0f, 0.0f, this->best_sse);
        rcandidates.push(std::move(rnode));

        while (!rcandidates.empty())
        {
            RotNode rnode = rcandidates.top();
            rcandidates.pop();

            if (best_sse - rnode.lb <= sse_threshold)
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

                curR = child_rnode.q.R;
                curT = glm::vec3(0.0f);

                // BnB in R3 
                auto [ub, best_t] = branch_and_bound_R3(child_rnode, true);
                curT = best_t;

                if (ub < best_sse * 1.8)
                {
                    IterativeClosestPoint3D icp3d(registration, pct, pcs, 100, sse_threshold, child_rnode.q.R, best_t);
                    auto [icp_sse, icp_R, icp_t] = icp3d.run();

                    if (icp_sse < best_sse)
                    {
                        best_sse = icp_sse;
                        best_rotation = icp_R;
                        best_translation = icp_t;
                        optR = icp_R;
                        optT = icp_t;
                    } 
                    Logger(LogLevel::Debug) << "New best error: " << best_sse << "\n"
                        << "\tRotation:\n" << best_rotation << "\n"
                        << "\tTranslation: " << best_translation;
                }

                auto [lb, _] = branch_and_bound_R3(child_rnode, false);
                Logger() << "ub: " << ub
                         << "\tlb: " << lb;

                if (lb >= best_sse) { continue; }
                child_rnode.lb = lb;
                child_rnode.ub = ub;

                rcandidates.push(std::move(child_rnode));
            }
        }
        return best_sse;
    }

    FastGoICP::ResultBnBR3 FastGoICP::branch_and_bound_R3(RotNode &rnode, bool fix_rot)
    {
        float best_error = this->best_sse;
        glm::vec3 best_t{ 0.0f };
        float best_ub = M_INF;

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
            auto [lb, ub] = registration.compute_sse_error(rnode, tnodes, fix_rot, stream_pool);

            // *Fix rotation* to compute rotation *lower bound*
            // Get min upper bound of this batch to update best SSE
            size_t idx_min = std::distance(std::begin(ub), std::min_element(std::begin(ub), std::end(ub)));
            best_ub = best_ub < ub[idx_min] ? best_ub : ub[idx_min];
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
                if (tnode.span < 0.1f) { continue; }  // TODO: use config threshold

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

        return { best_ub, best_t };
    }
}