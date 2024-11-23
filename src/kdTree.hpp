#pragma once
#include "kernel.h"
#include "nanoflann.hpp"
#include "common.h"

// PointCloudAdaptor struct definition for nanoflann usage
struct PointCloudAdaptor {
    std::vector<glm::vec3> points;

    // Required methods for nanoflann
    inline size_t kdtree_get_point_count() const {
        return points.size();
    }

    inline float kdtree_get_pt(const size_t idx, const size_t dim) const {
        if (dim == 0) return points[idx].x;
        else if (dim == 1) return points[idx].y;
        else return points[idx].z;
    }

    // Distance computation for nanoflann
    template <class T>
    float kdtree_distance(const T* p1, const size_t idx_p2, size_t /*size*/) const {
        const glm::vec3& p2 = points[idx_p2];
        float d0 = p1[0] - p2.x;
        float d1 = p1[1] - p2.y;
        float d2 = p1[2] - p2.z;
        return d0 * d0 + d1 * d1 + d2 * d2;  // Return squared distance
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /*bb*/) const {
        return false; // Return false indicating no bounding box computation
    }
};

// Define the KDTree type
using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float, PointCloudAdaptor>,
        PointCloudAdaptor,
        3 /* Data dimensionality */
    >;

class FlattenedKDTree
{
public:
    struct ArrayNode
    {
        bool is_leaf;

        // Leaf or Non-leaf node data
        union {
            // Leaf node data
            struct {
                size_t left, right;  // Indices of points in the leaf node
            } leaf;

            // Non-leaf node data
            struct {
                int32_t divfeat;       // Dimension used for subdivision
                float divlow, divhigh; // Range values used for subdivision
                size_t child1, child2; // Indices of child nodes in the array
            } nonleaf;
        } data;
    };

    thrust::device_vector<ArrayNode> d_array;  // Flattened KD-tree on device
    thrust::device_vector<uint32_t> d_vAcc;    // Indices mapping
    thrust::device_vector<glm::vec3> d_pct;      // Point cloud on device
    
    
    FlattenedKDTree(const KDTree& kdt, const std::vector<glm::vec3>& pct);
    __device__ void find_nearest_neighbor(const glm::vec3 query, float& best_dist, size_t& best_idx) const;

private:
    void FlattenedKDTree::flatten_KDTree(const KDTree::Node* root, std::vector<ArrayNode>& array, size_t& currentIndex);
};

