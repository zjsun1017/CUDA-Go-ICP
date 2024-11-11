#pragma once
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <glm/glm.hpp>
#include "utilityCore.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>

#include "nanoflann.hpp"

#define blockSize 512
#define scene_scale 0.050f

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

void checkCUDAError(const char* msg, int line = -1);

// Tree struct definition for nanoflann usage
struct Tree {
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
typedef nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<float, Tree>,
    Tree,
    3 /* Data dimensionality */
> KDTree;
void buildKDTree(Tree& tree);

namespace PointCloud {
    void initBuffers(std::vector<glm::vec3>& Ybuffer, std::vector<glm::vec3>& Xbuffer);
    void copyPointsToVBO(float* vbodptr_positions, float* vbodptr_colors);
    void cleanupBuffers();
}