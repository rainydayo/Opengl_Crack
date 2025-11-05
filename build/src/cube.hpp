#pragma once
#include "mesh.hpp"

// ลูกบาศก์พร้อมสีต่อเวอร์เท็กซ์ (8 จุด, 12 หน้า = 36 ดัชนี)
inline Mesh makeColoredCube() {
    static const float V[] = {
        // pos            // color
        -1,-1,-1,  1,0,0,
         1,-1,-1,  0,1,0,
         1, 1,-1,  0,0,1,
        -1, 1,-1,  1,1,0,
        -1,-1, 1,  1,0,1,
         1,-1, 1,  0,1,1,
         1, 1, 1,  1,1,1,
        -1, 1, 1,  0,0,0
    };
    static const unsigned I[] = {
        0,1,2, 2,3,0,   // back
        4,5,6, 6,7,4,   // front
        0,4,7, 7,3,0,   // left
        1,5,6, 6,2,1,   // right
        3,2,6, 6,7,3,   // top
        0,1,5, 5,4,0    // bottom
    };

    Mesh m;
    glGenVertexArrays(1, &m.vao);
    glBindVertexArray(m.vao);

    glGenBuffers(1, &m.vbo);
    glBindBuffer(GL_ARRAY_BUFFER, m.vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(V), V, GL_STATIC_DRAW);

    glGenBuffers(1, &m.ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(I), I, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0); // pos
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1); // color (ไม่ได้ใช้ในเอฟเฟกต์ แต่อยู่เผื่อ)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));

    m.indexCount = 36;
    return m;
}
