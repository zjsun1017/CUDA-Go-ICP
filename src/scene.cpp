#include "scene.h"

// Cube-related OpenGL buffers and data
std::vector<glm::vec3> cubePositions;
std::vector<glm::vec4> cubeColors;
GLuint cubeVAO, cubeVBO, cubeColorVBO, cubeEBO;

void initCubes(int gridSize, float cubeSize, const glm::vec3& sphereCenter, float sphereRadius) {
    // Initialize cube positions and colors
    std::vector<GLuint> cubeIndices;
    for (int x = 0; x < gridSize; x++) {
        for (int y = 0; y < gridSize; y++) {
            for (int z = 0; z < gridSize; z++) {
                glm::vec3 center(
                    (x + 0.5f) * cubeSize - 0.5f,
                    (y + 0.5f) * cubeSize - 0.5f,
                    (z + 0.5f) * cubeSize - 0.5f
                );
                cubePositions.push_back(center);

                // Default color
                glm::vec4 color(1.0f, 1.0f, 1.0f, 0.2f);
                float dist = glm::length(center - sphereCenter);
                if (dist <= sphereRadius) {
                    color = glm::vec4(0.0f, 1.0f, 0.0f, 0.5f); // Highlight green
                }
                cubeColors.push_back(color);
            }
        }
    }

    // Generate OpenGL buffers
    glGenVertexArrays(1, &cubeVAO);
    glGenBuffers(1, &cubeVBO);
    glGenBuffers(1, &cubeColorVBO);
    glGenBuffers(1, &cubeEBO);

    glBindVertexArray(cubeVAO);

    // Positions buffer
    glBindBuffer(GL_ARRAY_BUFFER, cubeVBO);
    glBufferData(GL_ARRAY_BUFFER, cubePositions.size() * sizeof(glm::vec3), cubePositions.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glEnableVertexAttribArray(0);

    // Colors buffer
    glBindBuffer(GL_ARRAY_BUFFER, cubeColorVBO);
    glBufferData(GL_ARRAY_BUFFER, cubeColors.size() * sizeof(glm::vec4), cubeColors.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), (void*)0);
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
}

void drawCubes(GLuint program) {
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glUseProgram(program);
    glBindVertexArray(cubeVAO);
    glDrawArrays(GL_POINTS, 0, cubePositions.size());
    glBindVertexArray(0);
    glUseProgram(0);
}

// Sphere-related OpenGL buffers and data
GLuint sphereVAO, sphereVBO, sphereEBO;
std::vector<glm::vec3> sphereVertices;
std::vector<GLuint> sphereIndices;

void initSphere(const glm::vec3& center, float radius) {
    // Generate sphere vertex and index data
    // (You can implement your own sphere generation logic here)

    // Generate OpenGL buffers
    glGenVertexArrays(1, &sphereVAO);
    glGenBuffers(1, &sphereVBO);
    glGenBuffers(1, &sphereEBO);

    glBindVertexArray(sphereVAO);

    // Vertex buffer
    glBindBuffer(GL_ARRAY_BUFFER, sphereVBO);
    glBufferData(GL_ARRAY_BUFFER, sphereVertices.size() * sizeof(glm::vec3), sphereVertices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glEnableVertexAttribArray(0);

    // Index buffer
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sphereEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sphereIndices.size() * sizeof(GLuint), sphereIndices.data(), GL_STATIC_DRAW);

    glBindVertexArray(0);
}

void drawSphere(GLuint program, const glm::vec3& center, float radius) {
    glm::mat4 model = glm::translate(glm::mat4(1.0f), center);
    model = glm::scale(model, glm::vec3(radius));

    glUseProgram(program);
    glBindVertexArray(sphereVAO);

    GLuint modelLoc = glGetUniformLocation(program, "u_model");
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, &model[0][0]);

    glDrawElements(GL_TRIANGLES, sphereIndices.size(), GL_UNSIGNED_INT, 0);

    glBindVertexArray(0);
    glUseProgram(0);
}