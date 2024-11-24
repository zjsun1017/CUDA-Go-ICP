/**
* @file      main.cpp
* @brief     GPU Accelerated Go-ICP
* @authors   Zhaojin Sun, Mufeng Xu
* @copyright 2024 Zhaojin & Mufeng. All rights reserved.
* @note      This code framework is based on CIS5650 Project 1
*/
#include "main.hpp"
#include "window.h"

#define CUDA_NAIVE 1
#define CUDA_KDTREE	1

/*
*C main function.
*/
int main(int argc, char* argv[]) {
	projectName = "Fast Globally Optimal ICP";
	initPointCloud(argc, argv);

	if (initMainWindow()) {
		initBufferAndkdTree();
		mainLoop();
		PointCloud::cleanupBuffers();
		return 0;
	}
	else {
		return 1;
	}
}

void initPointCloud(int argc, char** argv)
{
	// Parse configuration
	Config config(argv[1]);
	load_cloud(config.io.source, config.subsample, dataBuffer);
	load_cloud(config.io.target, config.subsample, modelBuffer);

	// Initialize drawing state
	numDataPoints = dataBuffer.size();
	numModelPoints = modelBuffer.size();
	numPoints = dataBuffer.size() + modelBuffer.size();
	Logger(LogLevel::Info) << "Total " << numPoints << " points loaded!";
}

void initBufferAndkdTree()
{
	// Init k-d Tree on CPU
	tree.points = modelBuffer;
	kdtree.buildIndex();
	Logger(LogLevel::Info) << "k-d Tree on CPU built!";

	// Init GPU buffers
	PointCloud::initBuffers(dataBuffer, modelBuffer);
	Logger(LogLevel::Info) << "GPU buffers initialized!";

	// Init k-d Tree on GPU
	FlattenedKDTree fkdt(kdtree, modelBuffer);
	cudaMalloc((void**)&dev_fkdt, sizeof(FlattenedKDTree));
	cudaMemcpy(dev_fkdt, &fkdt, sizeof(FlattenedKDTree), cudaMemcpyHostToDevice);
	Logger(LogLevel::Info) << "Flattened k-d Tree built!";
}

void runCUDA() {
#if CUDA_KDTREE
	ICP::kdTreeGPUStep(kdtree, tree, dev_fkdt);
#elif CUDA_NAIVE
	ICP::naiveGPUStep();
#else
	ICP::CPUStep(dataBuffer, modelBuffer);
#endif
}

void mainLoop() {
	double fps = 0;
	double timebase = 0;
	int frame = 0;

	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();

		frame++;
		double time = glfwGetTime();

		if (time - timebase > 1.0) {
			fps = frame / (time - timebase);
			timebase = time;
			frame = 0;
		}

		std::ostringstream ss;
		ss << "[";
		ss.precision(1);
		ss << std::fixed << fps;
		ss << " fps] " << deviceName;
		glfwSetWindowTitle(window, ss.str().c_str());

		runCUDA();
		drawMainWindow();
		glfwSwapBuffers(window);
	}
	glfwDestroyWindow(window);
	glfwTerminate();
}

