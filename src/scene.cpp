#include "scene.h"
#include "window.h"

extern int numTransCubes;

GLuint cubeVAO = 0, cubeVBO_positions = 0, cubeVBO_flags = 0, cubeVBO_sizes = 0;
void initCubeVAO() {
    std::unique_ptr<float[]> positions{ new float[3 * numTransCubes] }; 
    std::unique_ptr<int[]> flags{ new int[numTransCubes] };            
    std::unique_ptr<float[]> sizes{ new float[numTransCubes] };      

    for (int i = 0; i < numTransCubes; ++i) {
        positions[3 * i + 0] = 0.0f; // x
        positions[3 * i + 1] = 0.0f; // y
        positions[3 * i + 2] = 0.0f; // z

        flags[i] = 0;
        sizes[i] = 1.0f;
    }

    glGenVertexArrays(1, &cubeVAO); // VAO for the cubes
    glGenBuffers(1, &cubeVBO_positions); // VBO for positions
    glGenBuffers(1, &cubeVBO_flags); // VBO for flags
    glGenBuffers(1, &cubeVBO_sizes); // VBO for sizes

    glBindVertexArray(cubeVAO);

    // Position buffer
    glBindBuffer(GL_ARRAY_BUFFER, cubeVBO_positions);
    glBufferData(GL_ARRAY_BUFFER, 3 * numTransCubes * sizeof(float), positions.get(), GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0); // Attribute 0: Position

    // Flag buffer
    glBindBuffer(GL_ARRAY_BUFFER, cubeVBO_flags);
    glBufferData(GL_ARRAY_BUFFER, numTransCubes * sizeof(int), flags.get(), GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(1);
    glVertexAttribIPointer(1, 1, GL_INT, 0, (void*)0); // Attribute 1: Flag (integer)

    // Size buffer
    glBindBuffer(GL_ARRAY_BUFFER, cubeVBO_sizes);
    glBufferData(GL_ARRAY_BUFFER, numTransCubes * sizeof(float), sizes.get(), GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 0, (void*)0); // Attribute 2: Size

    glBindVertexArray(0); // Unbind VAO
}

void initCubeShaders(GLuint* program) {
    GLint location;
    const char* cubeAttributeLocations[] = { "Position", "Flag", "Size" };

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
        glm::mat4 view = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -3.0f));
        glUniformMatrix4fv(location, 1, GL_FALSE, &view[0][0]);
    }

    if ((location = glGetUniformLocation(program[PROG_CUBE], "u_modelMatrix")) != -1) {
        glm::mat4 modelMatrix = glm::mat4(1.0f);
        glUniformMatrix4fv(location, 1, GL_FALSE, &modelMatrix[0][0]);
    }
}




