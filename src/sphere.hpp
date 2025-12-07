#pragma once
#include "mesh.hpp"
#include <vector>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

inline Mesh makeSphere(int stacks = 32, int slices = 32) {
    std::vector<float> vertices;
    std::vector<unsigned int> indices;

    float radius = 1.0f;

    // 1. Generate Vertices
    for (int i = 0; i <= stacks; ++i) {
        float lat = (float)i / stacks;       // 0 to 1
        float theta = lat * (float)M_PI;     // 0 to PI (Top to Bottom)
        float sinTheta = sin(theta);
        float cosTheta = cos(theta);

        for (int j = 0; j <= slices; ++j) {
            float lon = (float)j / slices;   // 0 to 1
            float phi = lon * 2.0f * (float)M_PI; // 0 to 2PI (Around)
            float sinPhi = sin(phi);
            float cosPhi = cos(phi);

            // Position (x, y, z)
            float x = radius * sinTheta * cosPhi;
            float y = radius * cosTheta;          // Y is Up
            float z = radius * sinTheta * sinPhi;

            // Normal (normalized position for sphere)
            // We use this as "Color" placeholder to match cube.hpp layout
            float r = x / radius; // range -1..1
            float g = y / radius;
            float b = z / radius;

            // Add Position
            vertices.push_back(x);
            vertices.push_back(y);
            vertices.push_back(z);

            // Add Color (Placeholder)
            vertices.push_back((r + 1.0f) * 0.5f);
            vertices.push_back((g + 1.0f) * 0.5f);
            vertices.push_back((b + 1.0f) * 0.5f);
        }
    }

    // 2. Generate Indices
    for (int i = 0; i < stacks; ++i) {
        for (int j = 0; j < slices; ++j) {
            int first = (i * (slices + 1)) + j;
            int second = first + slices + 1;

            indices.push_back(first);
            indices.push_back(second);
            indices.push_back(first + 1);

            indices.push_back(second);
            indices.push_back(second + 1);
            indices.push_back(first + 1);
        }
    }

    // 3. Upload to GPU
    Mesh m;
    glGenVertexArrays(1, &m.vao);
    glBindVertexArray(m.vao);

    glGenBuffers(1, &m.vbo);
    glBindBuffer(GL_ARRAY_BUFFER, m.vbo);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

    glGenBuffers(1, &m.ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

    // Attribute 0: Position
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);

    // Attribute 1: Color (Not used by new shader, but kept for compatibility)
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));

    m.indexCount = (GLsizei)indices.size();
    return m;
}