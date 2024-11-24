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

void initCubes(int gridSize, float cubeSize, const glm::vec3& sphereCenter, float sphereRadius);
void drawCubes(GLuint program);
void initSphere(const glm::vec3& center, float radius);
void drawSphere(GLuint program, const glm::vec3& center, float radius);