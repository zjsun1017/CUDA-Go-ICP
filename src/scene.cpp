#include "scene.h"
#include "window.h"

extern int numTransCubes;

extern std::vector<glm::vec3> transCubePosBuffer;
extern std::vector<int> transCubeColBuffer;
extern std::vector<float> transCubeSizeBuffer;

GLuint cubeVAO = 0;
GLuint cubeVBO_positions = 0;
GLuint cubeVBO_colors = 0;
GLuint cubeIBO = 0;

extern GLuint positionLocation;   
extern GLuint colorsLocation;

void initCubeVAO() {
	std::unique_ptr<GLfloat[]> bodies{ new GLfloat[4 * (numTransCubes)] };
	std::unique_ptr<GLuint[]> bindices{ new GLuint[numTransCubes] };

	glm::vec4 ul(-1.0, -1.0, 1.0, 1.0);
	glm::vec4 lr(1.0, 1.0, 0.0, 0.0);

	for (int i = 0; i < numTransCubes; i++) {
		bodies[4 * i + 0] = 0.0f;
		bodies[4 * i + 1] = 0.0f;
		bodies[4 * i + 2] = 0.0f;
		bodies[4 * i + 3] = 1.0f;
		bindices[i] = i;
	}

	glGenVertexArrays(1, &cubeVAO); 
	glGenBuffers(1, &cubeVBO_positions);
	glGenBuffers(1, &cubeVBO_colors);
	glGenBuffers(1, &cubeIBO);

	glBindVertexArray(cubeVAO);

	glBindBuffer(GL_ARRAY_BUFFER, cubeVBO_positions); // bind the buffer
	glBufferData(GL_ARRAY_BUFFER, 4 * (numTransCubes) * sizeof(GLfloat), bodies.get(), GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(positionLocation);
	glVertexAttribPointer((GLuint)positionLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);

	glBindBuffer(GL_ARRAY_BUFFER, cubeVBO_colors);
	glBufferData(GL_ARRAY_BUFFER, 4 * (numTransCubes) * sizeof(GLfloat), bodies.get(), GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(colorsLocation);
	glVertexAttribPointer((GLuint)colorsLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cubeIBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, (numTransCubes) * sizeof(GLuint), bindices.get(), GL_STATIC_DRAW);

	glBindVertexArray(0);
}

void initCubeShaders(GLuint* program) {
    GLint location;
    const char* cubeAttributeLocations[] = { "Position", "Color" };

    program[PROG_CUBE] = glslUtility::createProgram(
        "shaders/cube.vert.glsl",
        "shaders/cube.geom.glsl",
        "shaders/point.frag.glsl",
        cubeAttributeLocations,
        2
    );

    glUseProgram(program[PROG_CUBE]);

    if ((location = glGetUniformLocation(program[PROG_CUBE], "u_projMatrix")) != -1) {
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.00001f, 100000.0f);
        glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
    }

    if ((location = glGetUniformLocation(program[PROG_CUBE], "u_viewMatrix")) != -1) {
        glm::mat4 view = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -20.0f));
        glUniformMatrix4fv(location, 1, GL_FALSE, &view[0][0]);
    }

    if ((location = glGetUniformLocation(program[PROG_CUBE], "u_modelMatrix")) != -1) {
        glm::mat4 modelMatrix = glm::mat4(1.0f);
        glUniformMatrix4fv(location, 1, GL_FALSE, &modelMatrix[0][0]);
    }
}


