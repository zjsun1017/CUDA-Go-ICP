#include "scene.h"
#include "window.h"

extern int numTransCubes;

extern std::vector<glm::vec3> transCubePosBuffer;
extern std::vector<int> transCubeFlagBuffer;
extern std::vector<float> transCubeSizeBuffer;

GLuint cubeVAO = 0, cubeVBO_positions = 0, cubeVBO_flags = 0, cubeEBO = 0;

void initCubeVAO() {
    const GLuint cubeIndices[] = {
        // Front face
        0, 1, 2, 2, 3, 0,
        // Back face
        4, 5, 6, 6, 7, 4,
        // Left face
        0, 4, 7, 7, 3, 0,
        // Right face
        1, 5, 6, 6, 2, 1,
        // Top face
        3, 7, 6, 6, 2, 3,
        // Bottom face
        0, 1, 5, 5, 4, 0
    };

    // Initialize buffers
    glGenVertexArrays(1, &cubeVAO);
    glGenBuffers(1, &cubeVBO_positions);
    glGenBuffers(1, &cubeVBO_flags);
    glGenBuffers(1, &cubeEBO); // Generate the EBO

    glBindVertexArray(cubeVAO);

    // Upload positions
    glBindBuffer(GL_ARRAY_BUFFER, cubeVBO_positions);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * 8 * numTransCubes, nullptr, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

    // Upload flags
    glBindBuffer(GL_ARRAY_BUFFER, cubeVBO_flags);
    glBufferData(GL_ARRAY_BUFFER, sizeof(int) * numTransCubes, nullptr, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(1);
    glVertexAttribIPointer(1, 1, GL_INT, 0, (void*)0);

    // Upload indices
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cubeEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(cubeIndices), cubeIndices, GL_STATIC_DRAW);

    glBindVertexArray(0); // Unbind VAO
}

void initCubeShaders(GLuint* program) {
    GLint location;
    const char* cubeAttributeLocations[] = {"vertPos", "Flag"};

    program[PROG_CUBE] = glslUtility::createProgram(
        "shaders/cube.vert.glsl",
        nullptr,
        "shaders/cube.frag.glsl",
        cubeAttributeLocations,
        2
    );

    glUseProgram(program[PROG_CUBE]);

    if ((location = glGetUniformLocation(program[PROG_CUBE], "u_projMatrix")) != -1) {
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 100.0f);
        glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
    }

    if ((location = glGetUniformLocation(program[PROG_CUBE], "u_viewMatrix")) != -1) {
        glm::mat4 view = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -2.0f));
        glUniformMatrix4fv(location, 1, GL_FALSE, &view[0][0]);
    }

    if ((location = glGetUniformLocation(program[PROG_CUBE], "u_modelMatrix")) != -1) {
        glm::mat4 modelMatrix = glm::mat4(1.0f);
        glUniformMatrix4fv(location, 1, GL_FALSE, &modelMatrix[0][0]);
    }
}



