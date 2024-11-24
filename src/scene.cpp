#include "scene.h"
#include "window.h"

const char* cubeAttributeLocations[] = { "Position", "Color", "Scale"};

GLuint cubeVAO, cubeVBO, cubeEBO;
void initCubeVAO() 
{
    glGenVertexArrays(1, &cubeVAO);
    glGenBuffers(1, &cubeVBO);
    glGenBuffers(1, &cubeEBO);

    glBindVertexArray(cubeVAO);

    glBindBuffer(GL_ARRAY_BUFFER, cubeVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(cubeVertices), cubeVertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cubeEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(cubeIndices), cubeIndices, GL_STATIC_DRAW);

    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
}

//void initCubeVAO() {
//    glGenVertexArrays(1, &cubeVAO);
//    glGenBuffers(1, &cubeVBO_positions);
//    glGenBuffers(1, &cubeVBO_colors);
//    glGenBuffers(1, &cubeVBO_sizes);
//
//    glBindVertexArray(cubeVAO);
//
//    glBindBuffer(GL_ARRAY_BUFFER, cubeVBO_positions);
//    glBufferData(GL_ARRAY_BUFFER, numCubes * 4 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
//    glEnableVertexAttribArray(0);
//    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
//
//    glBindBuffer(GL_ARRAY_BUFFER, cubeVBO_colors);
//    glBufferData(GL_ARRAY_BUFFER, numCubes * 4 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
//    glEnableVertexAttribArray(1);
//    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
//
//    glBindBuffer(GL_ARRAY_BUFFER, cubeVBO_sizes);
//    glBufferData(GL_ARRAY_BUFFER, numCubes * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
//    glEnableVertexAttribArray(2);
//    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)0);
//
//    glBindVertexArray(0);
//}



void initCubeShaders(GLuint* program) {
    GLint location;

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




