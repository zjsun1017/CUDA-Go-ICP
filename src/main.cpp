/**
* @file      main.cpp
* @brief     GPU Accelerated Go-ICP
* @authors   Zhaojin Sun, Mufeng Xu
* @copyright 2024 Zhaojin & Mufeng. All rights reserved.
* @note      This code framework is based on CIS5650 Project 1
*/
#include "main.hpp"
#include "window.h"

/*
*C main function.
*/
int main(int argc, char* argv[]) {
	projectName = "Fast Globally Optimal ICP";
	initPointCloud(argc, argv);

	if (initMainWindow()) {
		//std::this_thread::sleep_for(std::chrono::seconds(10));
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
	load_cloud(config.io.source, config.subsample, config.resize, dataBuffer);
	load_cloud(config.io.target, config.subsample, config.resize, modelBuffer);
	mode = config.mode;

	// Initialize drawing state
	numDataPoints = dataBuffer.size();
	numModelPoints = modelBuffer.size();
	mse_threshold = config.mse_threshold;
	sse_threshold = config.mse_threshold * numDataPoints;

	if (mode == GOICP_CPU)
	{
		goicp = new GoICP(mse_threshold);
		goicp->pModel = modelBuffer.data();
		goicp->Nm = numModelPoints;
		goicp->pData = dataBuffer.data();
		goicp->Nd = numDataPoints;

		// Build Distance Transform
		Logger(LogLevel::Info) << "Building Distance Transform...";
		goicp->BuildDT();
		Logger(LogLevel::Info) << "Done!";
	}
	if (mode == GOICP_CPU || mode == GOICP_GPU)
	{
		numPoints = 2 * numDataPoints + numModelPoints;
	}
	else 
		numPoints = numDataPoints + numModelPoints;
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

	// Init fgoicp
	if (mode == GOICP_GPU)
	{
		fgoicp = new icp::FastGoICP(modelBuffer, dataBuffer, 1e-3f, mtx);
		Logger(LogLevel::Info) << "FastGoICP instance created!";
	}
}

void runCUDA() {
	switch (mode) {
	case ICP_CPU:
		ICP::CPUStep(dataBuffer, modelBuffer);
		break;

	case ICP_GPU:
		ICP::naiveGPUStep();
		break;

	case ICP_KDTREE_GPU:
		ICP::kdTreeGPUStep(kdtree, tree, dev_fkdt);
		break;

	case GOICP_CPU:
		if (goicp_finished) ICP::naiveGPUStep();
		else ICP::sgoicpCPUStep(goicp, prev_optR, prev_optT, mtx);
		break;

	case GOICP_GPU:
		if (goicp_finished) ICP::naiveGPUStep();
		else ICP::goicpGPUStep(fgoicp, prev_optR_fgoicp, prev_optT_fgoicp, mtx);
		break;

	default:
		std::cerr << "Error: Invalid mode selected!" << std::endl;
		break;
	}
}

void mainLoop() {
	double fps = 0;
	double timebase = 0;
	int frame = 0;

	if (mode == GOICP_GPU)
	{
		std::thread fgoicp_thread(&icp::FastGoICP::run, fgoicp);
		fgoicp_thread.detach();
	}

	if (mode == GOICP_CPU)
	{
		Logger(LogLevel::Info) << "Initializing Go-ICP on CPU...";
		std::thread register_thread(&GoICP::Register, goicp);
		register_thread.detach();
	}

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

		glfwPollEvents();
	}
	glfwDestroyWindow(window);
	glfwTerminate();
}

