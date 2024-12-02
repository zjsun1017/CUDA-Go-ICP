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
	initSearchSpace();

	if (initMainWindow() && initSecondWindow()) {
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
		goicp.pModel = modelBuffer.data();
		goicp.Nm = numModelPoints;
		goicp.pData = dataBuffer.data();
		goicp.Nd = numDataPoints;

		// Build Distance Transform
		Logger(LogLevel::Info) << "Building Distance Transform...";
		goicp.BuildDT();
		Logger(LogLevel::Info) << "Done!";

		numPoints = 2 * numDataPoints + numModelPoints;
	}
	else 
		numPoints = numDataPoints + numModelPoints;
	Logger(LogLevel::Info) << "Total " << numPoints << " points loaded!";
}

void initSearchSpace() {
	if (mode != GOICP_GPU) return;

	float initialSize = 1.0f;

	transCubePosBuffer.push_back(glm::vec3(-1.0f, 0.0f, 0.0f));
	transCubeColBuffer.push_back(glm::vec3(1.0f));

	rotCubePosBuffer.push_back(glm::vec3(1.0f, 0.0f, 0.0f));
	rotCubeColBuffer.push_back(glm::vec3(1.0f));

	const glm::vec3 offsets[8] = {
		glm::vec3(-0.5f, -0.5f, -0.5f),
		glm::vec3(0.5f, -0.5f, -0.5f),
		glm::vec3(-0.5f,  0.5f, -0.5f),
		glm::vec3(0.5f,  0.5f, -0.5f),
		glm::vec3(-0.5f, -0.5f,  0.5f),
		glm::vec3(0.5f, -0.5f,  0.5f),
		glm::vec3(-0.5f,  0.5f,  0.5f),
		glm::vec3(0.5f,  0.5f,  0.5f)
	};

	int parentStartIndex = 0;
	int parentCount = 1;

	for (int divideLevel = 1; divideLevel <= maxCubeDivide; ++divideLevel) {
		float currentSize = initialSize / pow(2, divideLevel);

		for (int parentIndex = parentStartIndex; parentIndex < parentStartIndex + parentCount; ++parentIndex) {
			glm::vec3 parentPositionTrans = transCubePosBuffer[parentIndex];

			for (int childIndex = 0; childIndex < 8; ++childIndex) {
				glm::vec3 childPositionTrans = parentPositionTrans + currentSize * offsets[childIndex];

				transCubePosBuffer.push_back(childPositionTrans);
				transCubeColBuffer.push_back(glm::vec3(1.0f));
			}
		}

		for (int parentIndex = parentStartIndex; parentIndex < parentStartIndex + parentCount; ++parentIndex) {
			glm::vec3 parentPositionRot = rotCubePosBuffer[parentIndex];

			for (int childIndex = 0; childIndex < 8; ++childIndex) {
				glm::vec3 childPositionRot = parentPositionRot + currentSize * offsets[childIndex];

				rotCubePosBuffer.push_back(childPositionRot);

				float sphereRadius = sqrt(3) / 4;
				glm::vec3 sphereCenter(1.0f, 0.0f, 0.0f);

				glm::vec3 cubeMin = childPositionRot - glm::vec3(currentSize / 2.0f);
				glm::vec3 cubeMax = childPositionRot + glm::vec3(currentSize / 2.0f);

				glm::vec3 closestPointOnCube = glm::clamp(sphereCenter, cubeMin, cubeMax);

				float dist = glm::distance(closestPointOnCube, sphereCenter);

				if (dist > sphereRadius) {
					rotCubeColBuffer.push_back(glm::vec3(0.01f));
				}
				else {
					rotCubeColBuffer.push_back(glm::vec3(1.0f));
				}
			}
		}

		parentStartIndex += parentCount;
		parentCount *= 8;
	}

	numCubes = transCubeColBuffer.size();
	Logger(LogLevel::Info) << "Total " << numCubes << " search cubes initialized!";
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

void runCUDA(StreamPool& stream_pool) {
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
		if (!initialized) {
			// Initialize rotation candidates once
			RotNode rnode = RotNode(0.0f, 0.0f, 0.0f, 1.0f, 0.0f, bestSSE);
			rcandidates.push(std::move(rnode));
			initialized = true;
		}

		// Execute one step of branch and bound
		if (!ICP::branchAndBoundSO3Step(stream_pool)) {
			Logger() << "Branch and Bound completed. Final SSE: " << bestSSE;
			initialized = false; // Reset for next iteration
		}
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

	if (mode == GOICP_CPU)
	{
		Logger(LogLevel::Info) << "Initializing Go-ICP on CPU...";
		std::thread register_thread(&GoICP::Register, &goicp);
		register_thread.detach();
	}

	StreamPool stream_pool(32);
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

		runCUDA(stream_pool);
		drawMainWindow();
		drawSecondWindow();

		glfwPollEvents();
	}
	glfwDestroyWindow(window);
	glfwDestroyWindow(secondWindow);
	glfwTerminate();
}

