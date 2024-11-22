# pragma once
#include "kernel.h"

struct Correspondence
{
    size_t idx_s;
    size_t idx_t;
    float dist_squared;
    glm::vec3 ps_transformed;
};

struct Rotation
{
    float rr, x, y, z;
    glm::mat3 R;

    Rotation(float x, float y, float z) :
        x(x), y(y), z(z)
    {
        rr = x * x + y * y + z * z;
        if (rr > 1.0f) { return; } // Not a rotation

        float ww = 1.0f - rr;
        float w = sqrt(ww);
        float wx = w * x, xx = x * x;
        float wy = w * y, xy = x * y, yy = y * y;
        float wz = w * z, xz = x * z, yz = y * z, zz = z * z;

        R = glm::mat3(
            ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy),
            2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx),
            2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz
        );
    }
};