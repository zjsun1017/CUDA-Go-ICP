# pragma once
#include <iostream>
#include <cstdlib>
#include <string>
#include <sstream>
#include <fstream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "utilityCore.hpp"
#include "glslUtility.hpp"
#include "icp_kernel.h"
#include "kernel.h"

#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

//Cube Setup
const GLfloat cubeVertices[] = {
    // Positions           // Colors
    -0.5f, -0.5f, -0.5f,   1.0f, 0.0f, 0.0f, // Bottom-left
     0.5f, -0.5f, -0.5f,   0.0f, 1.0f, 0.0f, // Bottom-right
     0.5f,  0.5f, -0.5f,   0.0f, 0.0f, 1.0f, // Top-right
    -0.5f,  0.5f, -0.5f,   1.0f, 1.0f, 0.0f, // Top-left

    -0.5f, -0.5f,  0.5f,   1.0f, 0.0f, 1.0f, // Bottom-left (back)
     0.5f, -0.5f,  0.5f,   0.0f, 1.0f, 1.0f, // Bottom-right (back)
     0.5f,  0.5f,  0.5f,   1.0f, 1.0f, 1.0f, // Top-right (back)
    -0.5f,  0.5f,  0.5f,   0.5f, 0.5f, 0.5f  // Top-left (back)
};

const GLuint cubeIndices[] = {
    // Front face
    0, 1, 2,
    2, 3, 0,
    // Back face
    4, 5, 6,
    6, 7, 4,
    // Left face
    0, 4, 7,
    7, 3, 0,
    // Right face
    1, 5, 6,
    6, 2, 1,
    // Top face
    3, 7, 6,
    6, 2, 3,
    // Bottom face
    0, 1, 5,
    5, 4, 0
};

void initCubeVAO();
void initCubeShaders(GLuint* program);
