#include "window.h"
#include "scene.h"

//====================================
// GL Stuff for Main Window
//====================================

GLuint positionLocation = 0;   // Match results from glslUtility::createProgram.
GLuint colorsLocation = 1; // Also see attribtueLocations below.

GLuint pointVAO = 0;
GLuint pointVBO_positions = 0;
GLuint pointVBO_colors = 0;
GLuint pointIBO = 0;

GLuint program[3];

const float fovy = (float)(PI / 4);
const float zNear = 0.0001f;
const float zFar = 10000.0f;
int width = 1280;
int height = 720;
int pointSize = 2;

// For camera controls
bool leftMousePressed = false;
bool rightMousePressed = false;
double lastX;
double lastY;
float theta = 0.4f;
float phi = 0.0f;
float zoom = 0.1f;
glm::vec3 lookAt = glm::vec3(0.0f, 0.0f, 0.1f);
glm::vec3 cameraPosition;
glm::mat4 projection;

extern const char* projectName;
extern int numPoints;
extern int numDataPoints;
extern int numModelPoints;

extern int numCubes;

extern std::string deviceName;
extern GLFWwindow* window;
extern GLFWwindow* secondWindow;

extern GLuint cubeVAO;
extern GLuint cubeVBO_positions;
extern GLuint cubeVBO_colors;
extern GLuint cubeIBO;

extern glm::vec3* dev_pos;
extern glm::vec3* dev_col;
extern glm::vec3* dev_cubePosBuffer;
extern glm::vec3* dev_cubeColBuffer;

/**
* Initialization of main window
*/
bool initMainWindow() {
	cudaDeviceProp deviceProp;
	int gpuDevice = 0;
	int device_count = 0;

	cudaGetDeviceCount(&device_count);

	if (gpuDevice > device_count) {
		std::cout
			<< "Error: GPU device number is greater than the number of devices!"
			<< " Perhaps a CUDA-capable GPU is not installed?"
			<< std::endl;
		return false;
	}
	cudaGetDeviceProperties(&deviceProp, gpuDevice);
	int major = deviceProp.major;
	int minor = deviceProp.minor;

	std::ostringstream ss;
	ss << projectName << " [SM " << major << "." << minor << " " << deviceProp.name << "]";
	deviceName = ss.str();

	// Window setup stuff
	glfwSetErrorCallback(errorCallback);

	if (!glfwInit()) {
		std::cout
			<< "Error: Could not initialize GLFW!"
			<< " Perhaps OpenGL 3.3 isn't available?"
			<< std::endl;
		return false;
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	window = glfwCreateWindow(width, height, deviceName.c_str(), NULL, NULL);
	if (!window) {
		glfwTerminate();
		return false;
	}
	glfwMakeContextCurrent(window);
	glfwSetKeyCallback(window, keyCallback);
	glfwSetCursorPosCallback(window, mousePositionCallback);
	glfwSetMouseButtonCallback(window, mouseButtonCallback);

	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK) {
		return false;
	}

	initPointVAO();

	// Default to device ID 0. If you have more than one GPU and want to test a non-default one,
	// change the device ID.
	cudaGLSetGLDevice(0);

	cudaGLRegisterBufferObject(pointVBO_positions);
	cudaGLRegisterBufferObject(pointVBO_colors);

	updateCamera();
	initPointShaders(program);
	glEnable(GL_DEPTH_TEST);
	return true;
}

void initPointVAO() {

	std::unique_ptr<GLfloat[]> bodies{ new GLfloat[4 * (numPoints)] };
	std::unique_ptr<GLuint[]> bindices{ new GLuint[numPoints] };

	glm::vec4 ul(-1.0, -1.0, 1.0, 1.0);
	glm::vec4 lr(1.0, 1.0, 0.0, 0.0);

	for (int i = 0; i < numPoints; i++) {
		bodies[4 * i + 0] = 0.0f;
		bodies[4 * i + 1] = 0.0f;
		bodies[4 * i + 2] = 0.0f;
		bodies[4 * i + 3] = 1.0f;
		bindices[i] = i;
	}


	glGenVertexArrays(1, &pointVAO); // Attach everything needed to draw a particle to this
	glGenBuffers(1, &pointVBO_positions);
	glGenBuffers(1, &pointVBO_colors);
	glGenBuffers(1, &pointIBO);

	glBindVertexArray(pointVAO);

	// Bind the positions array to the pointVAO by way of the pointVBO_positions
	glBindBuffer(GL_ARRAY_BUFFER, pointVBO_positions); // bind the buffer
	glBufferData(GL_ARRAY_BUFFER, 4 * (numPoints) * sizeof(GLfloat), bodies.get(), GL_DYNAMIC_DRAW); // transfer data
	glEnableVertexAttribArray(positionLocation);
	glVertexAttribPointer((GLuint)positionLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);

	// Bind the colors array to the pointVAO by way of the pointVBO_colors
	glBindBuffer(GL_ARRAY_BUFFER, pointVBO_colors);
	glBufferData(GL_ARRAY_BUFFER, 4 * (numPoints) * sizeof(GLfloat), bodies.get(), GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(colorsLocation);
	glVertexAttribPointer((GLuint)colorsLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, pointIBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, (numPoints) * sizeof(GLuint), bindices.get(), GL_STATIC_DRAW);

	glBindVertexArray(0);
}

void initPointShaders(GLuint* program) {
	GLint location;
	const char* pointAttributes[] = { "Position", "Color" };

	program[PROG_POINT] = glslUtility::createProgram(
		"shaders/point.vert.glsl",
		"shaders/point.geom.glsl",
		"shaders/point.frag.glsl", 
		pointAttributes, 2);
	glUseProgram(program[PROG_POINT]);

	if ((location = glGetUniformLocation(program[PROG_POINT], "u_projMatrix")) != -1) {
		glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
	}
	if ((location = glGetUniformLocation(program[PROG_POINT], "u_cameraPos")) != -1) {
		glUniform3fv(location, 1, &cameraPosition[0]);
	}
}

void drawMainWindow()
{
	glfwMakeContextCurrent(window);
	// Map OpenGL buffer object for writing from CUDA on a single GPU
	// No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not
	// use this buffer
	float4* dptr = NULL;
	float* dptrVertPositions = NULL;
	float* dptrVertcolors = NULL;

	cudaGLMapBufferObject((void**)&dptrVertPositions, pointVBO_positions);
	cudaGLMapBufferObject((void**)&dptrVertcolors, pointVBO_colors);

	if (!dptrVertPositions || !dptrVertcolors) {
		printf("Error: Null pointer passed to point kernel!\n");
	}

	PointCloud::copyPointsToVBO(numPoints,dev_pos,dev_col,dptrVertPositions, dptrVertcolors);
	// unmap buffer object
	cudaGLUnmapBufferObject(pointVBO_positions);
	cudaGLUnmapBufferObject(pointVBO_colors);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(program[PROG_POINT]);
	glBindVertexArray(pointVAO);
	glPointSize((GLfloat)pointSize);
	glDrawElements(GL_POINTS, numPoints + 1, GL_UNSIGNED_INT, 0);
	glPointSize(1.0f);

	glUseProgram(0);
	glBindVertexArray(0);
	glfwSwapBuffers(window);
}

// Call back and camera functions
void errorCallback(int error, const char* description) {
	fprintf(stderr, "error %d: %s\n", error, description);
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, GL_TRUE);
	}
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
	leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
	rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
	if (leftMousePressed) {
		// compute new camera parameters
		phi += (xpos - lastX) / width;
		theta -= (ypos - lastY) / height;
		theta = std::fmax(0.01f, std::fmin(theta, 3.14f));
		updateCamera();
	}
	else if (rightMousePressed) {
		zoom += (ypos - lastY) / height;
		zoom = std::fmax(0.1f, std::fmin(zoom, 10000.0f));
		updateCamera();
	}

	lastX = xpos;
	lastY = ypos;
}

void updateCamera() {
	cameraPosition.x = zoom * sin(phi) * sin(theta);
	cameraPosition.z = zoom * cos(theta);
	cameraPosition.y = zoom * cos(phi) * sin(theta);
	cameraPosition += lookAt;

	projection = glm::perspective(fovy, float(width) / float(height), zNear, zFar);
	glm::mat4 view = glm::lookAt(cameraPosition, lookAt, glm::vec3(0, 0, 1));
	projection = projection * view;

	GLint location;

	glUseProgram(program[PROG_POINT]);
	if ((location = glGetUniformLocation(program[PROG_POINT], "u_projMatrix")) != -1) {
		glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
	}
}

bool initSecondWindow() {
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	secondWindow = glfwCreateWindow(1200, 600, "Second Window", NULL, window);
	if (!secondWindow) {
		std::cout << "Error: Could not create second GLFW window!" << std::endl;
		glfwTerminate();
		return false;
	}

	glfwMakeContextCurrent(secondWindow);

	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK) {
		std::cout << "Error: Could not initialize GLEW for second window!" << std::endl;
		return false;
	}

	initCubeVAO();

	cudaGLRegisterBufferObject(cubeVBO_positions);
	cudaGLRegisterBufferObject(cubeVBO_colors);

	glEnable(GL_DEPTH_TEST);

	initCubeShaders(program);
	return true;
}

void drawSecondWindow() {
	glfwMakeContextCurrent(secondWindow);
	glUseProgram(program[PROG_CUBE]);
	glBindVertexArray(cubeVAO);

	// Map OpenGL buffer objects for CUDA
	float* vbo_positions = nullptr;
	float* vbo_colors = nullptr;

	cudaGLMapBufferObject((void**)&vbo_positions, cubeVBO_positions);
	cudaGLMapBufferObject((void**)&vbo_colors, cubeVBO_colors);

	if (!vbo_positions || !vbo_colors) {
		printf("Error: Null pointer passed to cube kernel!\n");
		return;
	}

	// Update vertex positions and flags
	PointCloud::copyPointsToVBO(2 * numCubes,dev_cubePosBuffer, dev_cubeColBuffer, vbo_positions, vbo_colors);

	// Unmap the buffer objects
	cudaGLUnmapBufferObject(cubeVBO_positions);
	cudaGLUnmapBufferObject(cubeVBO_colors);

	// Set uniform matrices
	GLint modelLoc = glGetUniformLocation(program[PROG_CUBE], "u_modelMatrix");
	glm::mat4 modelMatrix = glm::rotate(glm::mat4(1.0f), (float)glfwGetTime(), glm::vec3(0.5f, 1.0f, 0.0f));
	glUniformMatrix4fv(modelLoc, 1, GL_FALSE, &modelMatrix[0][0]);

	GLint viewLoc = glGetUniformLocation(program[PROG_CUBE], "u_viewMatrix");
	glm::mat4 viewMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -5.0f));
	glUniformMatrix4fv(viewLoc, 1, GL_FALSE, &viewMatrix[0][0]);

	GLint projLoc = glGetUniformLocation(program[PROG_CUBE], "u_projMatrix");
	glm::mat4 projMatrix = glm::perspective(glm::radians(45.0f), 1200.0f / 600.0f, 0.1f, 100.0f);
	glUniformMatrix4fv(projLoc, 1, GL_FALSE, &projMatrix[0][0]);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glDrawElements(GL_POINTS, 2 * numCubes + 1, GL_UNSIGNED_INT, 0);
	glPointSize(2.0f);

	glBindVertexArray(0);
	glUseProgram(0);
	glfwSwapBuffers(secondWindow);
}