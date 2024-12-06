#pragma once
#include "kernel.h"

const unsigned int PROG_POINT = 0;
const unsigned int PROG_CUBE = 1;
const unsigned int PROG_SPHERE = 2;

//First window
bool initMainWindow();
void initPointVAO();
void initPointShaders(GLuint* program);
void updateCamera();
void drawMainWindow();

void errorCallback(int error, const char* description);
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
void mousePositionCallback(GLFWwindow* window, double xpos, double ypos);