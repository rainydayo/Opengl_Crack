// main.cpp
#include <stdexcept>
#include <cstdio>
#include <iostream>
#include <string>
#include <sstream>

#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "mesh.hpp"
#include "cube.hpp"

// ================== Camera ==================
glm::vec3 camPos = { 0.0f, 0.0f, 4.0f };
glm::vec3 camFront = { 0.0f, 0.0f,-1.0f };
glm::vec3 camUp = { 0.0f, 1.0f, 0.0f };

float yaw = -90.f;
float pitch = 0.f;
float fov = 60.f;

float deltaTime = 0.f;
float lastFrame = 0.f;

bool   firstMouse = true;
double lastX = 0.0, lastY = 0.0;

void processInput(GLFWwindow* w) {
    float sp = 2.5f * deltaTime;
    if (glfwGetKey(w, GLFW_KEY_W) == GLFW_PRESS) camPos += sp * camFront;
    if (glfwGetKey(w, GLFW_KEY_S) == GLFW_PRESS) camPos -= sp * camFront;
    if (glfwGetKey(w, GLFW_KEY_A) == GLFW_PRESS) camPos -= glm::normalize(glm::cross(camFront, camUp)) * sp;
    if (glfwGetKey(w, GLFW_KEY_D) == GLFW_PRESS) camPos += glm::normalize(glm::cross(camFront, camUp)) * sp;
    if (glfwGetKey(w, GLFW_KEY_E) == GLFW_PRESS) camPos += sp * camUp;
    if (glfwGetKey(w, GLFW_KEY_Q) == GLFW_PRESS) camPos -= sp * camUp;
    if (glfwGetKey(w, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(w, 1);
}
void cursorPosCallback(GLFWwindow*, double x, double y) {
    static const float S = 0.1f;
    if (firstMouse) { lastX = x; lastY = y; firstMouse = false; }
    float xoff = float(x - lastX) * S;
    float yoff = float(lastY - y) * S;
    lastX = x; lastY = y;

    yaw += xoff;
    pitch += yoff;
    pitch = glm::clamp(pitch, -89.f, 89.f);

    glm::vec3 d;
    d.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    d.y = sin(glm::radians(pitch));
    d.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    camFront = glm::normalize(d);
}
void scrollCallback(GLFWwindow*, double, double yoff) {
    fov -= float(yoff);
    fov = glm::clamp(fov, 1.f, 90.f);
}

// ================== Small console helpers ==================
float askFloat(const std::string& label, float def) {
    std::cout << label << " [" << def << "]: ";
    std::string line;
    std::getline(std::cin, line);
    if (line.empty()) return def;
    try { return std::stof(line); }
    catch (...) { return def; }
}

glm::vec3 askVec3(const std::string& label, const glm::vec3& def) {
    std::cout << label << " (x y z) [" << def.x << " " << def.y << " " << def.z << "]: ";
    std::string line;
    std::getline(std::cin, line);
    if (line.empty()) return def;
    std::stringstream ss(line);
    glm::vec3 v = def;
    if (!(ss >> v.x >> v.y >> v.z)) return def;
    return v;
}

int askInt(const std::string& label, int def) {
    std::cout << label << " [" << def << "]: ";
    std::string line;
    std::getline(std::cin, line);
    if (line.empty()) return def;
    try { return std::stoi(line); }
    catch (...) { return def; }
}

// ================== Shaders ==================
static const char* VS = R"(
#version 450 core
layout(location=0) in vec3 aPos;
out vec3 vPos;
out vec2 vUV;

// box projection ต่อหน้าลูกบาศก์
vec2 faceUV(vec3 p){
    vec3 ap = abs(p);
    if(ap.x >= ap.y && ap.x >= ap.z){           // ±X : (z,y)
        return vec2((p.z+1.0)*0.5, (p.y+1.0)*0.5);
    }else if(ap.y >= ap.x && ap.y >= ap.z){     // ±Y : (x,z)
        return vec2((p.x+1.0)*0.5, (p.z+1.0)*0.5);
    }else{                                      // ±Z : (x,y)
        return vec2((p.x+1.0)*0.5, (p.y+1.0)*0.5);
    }
}

uniform mat4 uMVP;
void main(){
  vPos = aPos;
  vUV  = faceUV(aPos);
  gl_Position = uMVP * vec4(aPos,1.0);
}
)";

static const char* FS = R"(
#version 450 core
in vec3 vPos;
in vec2 vUV;
out vec4 FragColor;

// ---------- box-UV helper ----------
vec2 faceUV(vec3 p){
    vec3 ap = abs(p);
    if(ap.x >= ap.y && ap.x >= ap.z){           // ±X : (z,y)
        return vec2((p.z+1.0)*0.5, (p.y+1.0)*0.5);
    }else if(ap.y >= ap.x && ap.y >= ap.z){     // ±Y : (x,z)
        return vec2((p.x+1.0)*0.5, (p.z+1.0)*0.5);
    }else{                                      // ±Z : (x,y)
        return vec2((p.x+1.0)*0.5, (p.y+1.0)*0.5);
    }
}

// ---------- hash / noise ----------
uint hash1(uvec2 x){
    x = (x*1664525u + 1013904223u);
    x ^= (x.yx>>16);
    return x.x * 2246822519u + x.y * 3266489917u;
}
float rand2(vec2 p){
    uvec2 u = floatBitsToUint(p);
    return float(hash1(u)) / 4294967295.0;
}
vec2 rand2_2(vec2 p){
    float r = rand2(p);
    float s = rand2(p + 19.19);
    return vec2(r,s);
}
float valueNoise(vec2 p){
    vec2 i=floor(p), f=fract(p);
    float a=rand2(i);
    float b=rand2(i+vec2(1,0));
    float c=rand2(i+vec2(0,1));
    float d=rand2(i+vec2(1,1));
    vec2  u=f*f*(3.0-2.0*f);
    return mix(mix(a,b,u.x), mix(c,d,u.x), u.y);
}
float fbm(vec2 p){
    float a=0.5, s=0.0;
    for(int i=0;i<5;i++){ s+=a*valueNoise(p); p*=2.02; a*=0.5; }
    return s;
}

// ---------- Worley/Cellular ----------
vec2 worleyF(vec2 p){
    vec2 ip=floor(p), fp=fract(p);
    float F1=1e9, F2=1e9;
    for(int j=-1;j<=1;j++){
        for(int i=-1;i<=1;i++){
            vec2 cell = ip + vec2(i,j);
            vec2 r2 = rand2_2(cell) - 0.5;
            vec2 d = (vec2(i,j) + r2) - fp;
            float dist = dot(d,d);
            if(dist < F1){ F2=F1; F1=dist; }
            else if(dist < F2){ F2=dist; }
        }
    }
    return vec2(sqrt(F1), sqrt(F2));
}

mat2 rot(float a){ float c=cos(a), s=sin(a); return mat2(c,-s,s,c); }

// ---------- uniforms ----------
uniform float uScale;
uniform float uJitter;
uniform float uCrackWidth;
uniform vec2  uStressDir;
uniform float uAniso;

// impact + material
uniform vec3  uImpactPos;
uniform vec3  uImpactDir;       // expected normalized
uniform float uImpactStrength;  // 0..~2
uniform float uMatBrittle;      // 0..1
uniform float uMatToughness;    // 0..1

// warp ด้วย cellular ตาม stress
vec2 stressWarp(vec2 uv, float scaleLocal, float jitterLocal, float anisoLocal){
    float ang = atan(uStressDir.y, uStressDir.x);
    mat2 R    = rot(ang);
    mat2 Rinv = rot(-ang);

    vec2 q = R * (uv * scaleLocal);
    q.y   *= (1.0 - 0.6*anisoLocal);
    q      = Rinv * q;

    float j = jitterLocal * (fbm(uv*scaleLocal*1.3 + 7.7) - 0.5);
    return q + j;
}

void main(){
    vec2 uv = vUV;

    // ---------- Impact-driven stress field ----------
    vec2 impactUV = faceUV(uImpactPos);

    // project impactDir onto plane of current face
    vec3 ap = abs(vPos);
    vec2 dirUV;
    if(ap.x >= ap.y && ap.x >= ap.z){           // ±X : (z,y)
        dirUV = vec2(uImpactDir.z, uImpactDir.y);
    }else if(ap.y >= ap.x && ap.y >= ap.z){     // ±Y : (x,z)
        dirUV = vec2(uImpactDir.x, uImpactDir.z);
    }else{                                      // ±Z : (x,y)
        dirUV = vec2(uImpactDir.x, uImpactDir.y);
    }
    if(length(dirUV) < 0.001) dirUV = vec2(1.0,0.0);
    dirUV = normalize(dirUV);

    vec2 rel   = uv - impactUV;
    float along = dot(rel, dirUV);
    float side  = length(rel - along*dirUV);

    float fallAlong = 3.0;
    float fallSide  = 6.0;

    float stress = uImpactStrength * exp(-abs(along)*fallAlong - side*fallSide);
    stress = clamp(stress, 0.0, 1.0);

    float brittle = clamp(uMatBrittle,   0.0, 1.0);
    float tough   = clamp(uMatToughness, 0.0, 1.0);

    // ---------- map stress + material -> cellular params ----------
    float scaleLocal  = uScale  * mix(0.8, 1.6, brittle) * (0.7 + 0.6*stress);
    float jitterLocal = uJitter * (0.4 + 0.8*brittle)    * (0.6 + 0.4*stress);
    float anisoLocal  = uAniso  * (0.3 + 0.7*brittle)    * (0.5 + 0.8*stress);

    float widthLocal  = uCrackWidth
                        * mix(1.3, 0.7, brittle)
                        * mix(0.8, 1.4, tough)
                        * (1.0 - 0.5*stress);
    widthLocal = max(widthLocal, 0.002);

    // ---------- Worley-based cracks ----------
    vec2 p = stressWarp(uv, scaleLocal, jitterLocal, anisoLocal);
    vec2 F = worleyF(p);
    float edge = F.y - F.x;

    float crack = 1.0 - smoothstep(0.0, widthLocal, edge);
    float micro = 1.0 - smoothstep(0.0, widthLocal*0.5, abs(edge - 0.02));

    vec3 base = vec3(0.72, 0.69, 0.65);
    vec3 c = mix(base, vec3(0.05), clamp(crack + 0.5*micro, 0.0, 1.0));

    // fake lighting จาก gradient ของ edge
    vec3 L = normalize(vec3(0.5, 0.8, 0.6));
    float e = 0.002;
    vec2 px = vec2(e,0), py=vec2(0,e);
    vec2 Fdx = worleyF(p+px);
    vec2 Fdy = worleyF(p+py);
    float ex = (Fdx.y-Fdx.x) - edge;
    float ey = (Fdy.y-Fdy.x) - edge;
    vec3 N = normalize(vec3(-ex, -ey, 1.0));
    float diff = max(dot(N, L), 0.0);
    c *= (0.35 + 0.65*diff);

    FragColor = vec4(c, 1.0);
}
)";

// ================== GL utils ==================
static void die(const char* m) { throw std::runtime_error(m); }
static GLuint makeProg(const char* vs, const char* fs) {
    auto comp = [&](GLenum t, const char* s) {
        GLuint sh = glCreateShader(t);
        glShaderSource(sh, 1, &s, nullptr);
        glCompileShader(sh);
        GLint ok;
        glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            char log[2048];
            glGetShaderInfoLog(sh, 2048, nullptr, log);
            std::fprintf(stderr, "%s\n", log);
            die("shader compile failed");
        }
        return sh;
        };
    GLuint v = comp(GL_VERTEX_SHADER, vs);
    GLuint f = comp(GL_FRAGMENT_SHADER, fs);
    GLuint p = glCreateProgram();
    glAttachShader(p, v);
    glAttachShader(p, f);
    glLinkProgram(p);
    GLint ok;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetProgramInfoLog(p, 2048, nullptr, log);
        std::fprintf(stderr, "%s\n", log);
        die("program link failed");
    }
    glDeleteShader(v);
    glDeleteShader(f);
    return p;
}

// ================== main ==================
int main() {
    // ---------- 1. ตั้งค่าพื้นฐาน + เลือก testcase / manual ----------
    std::cin.sync_with_stdio(false);
    std::cin.tie(nullptr);

    float baseScale = 6.0f;
    float baseJitter = 0.25f;
    float baseCrackWidth = 0.045f;
    float baseAniso = 0.6f;

    glm::vec3 impactPos = glm::vec3(0.0f, 0.0f, 1.0f);
    glm::vec3 impactDir = glm::normalize(glm::vec3(0.3f, -0.4f, -1.0f));
    float impactStrength = 1.0f;

    float matBrittle = 0.7f;
    float matToughness = 0.3f;

    std::cout << "========== Crack Test Configuration ==========\n";
    std::cout << "เลือก testcase:\n";
    std::cout << " 0  - Manual input (กรอกค่าทุกตัวเอง)\n";
    std::cout << " 1  - Glass-like (เปราะมาก, แรงสูง, แตกกระจาย)\n";
    std::cout << " 2  - Ceramic tile (เปราะปานกลาง, แตกพอประมาณ)\n";
    std::cout << " 3  - Concrete slab (กลาง ๆ, ลายหยาบ)\n";
    int preset = askInt("Preset ID", 2);

    if (preset == 1) {
        // Glass-like
        impactPos = glm::vec3(0.0f, 0.0f, 1.0f);
        impactDir = glm::normalize(glm::vec3(0.0f, 0.0f, -1.0f));
        impactStrength = 1.4f;
        baseScale = 9.0f;
        baseJitter = 0.28f;
        baseCrackWidth = 0.035f;
        baseAniso = 0.9f;
        matBrittle = 0.95f;
        matToughness = 0.1f;
    }
    else if (preset == 2) {
        // Ceramic tile
        impactPos = glm::vec3(0.0f, 0.0f, 1.0f);
        impactDir = glm::normalize(glm::vec3(0.2f, -0.3f, -1.0f));
        impactStrength = 1.0f;
        baseScale = 6.0f;
        baseJitter = 0.23f;
        baseCrackWidth = 0.045f;
        baseAniso = 0.7f;
        matBrittle = 0.7f;
        matToughness = 0.3f;
    }
    else if (preset == 3) {
        // Concrete
        impactPos = glm::vec3(0.0f, 0.0f, 1.0f);
        impactDir = glm::normalize(glm::vec3(0.0f, -0.5f, -1.0f));
        impactStrength = 0.8f;
        baseScale = 4.5f;
        baseJitter = 0.20f;
        baseCrackWidth = 0.06f;
        baseAniso = 0.4f;
        matBrittle = 0.5f;
        matToughness = 0.6f;
    }
    else {
        // Manual
        std::cout << "\nManual input (enter = ค่าเดิม)\n";
        impactPos = askVec3("Impact position (object space -1..1)", impactPos);
        impactDir = askVec3("Impact direction (will be normalized)", impactDir);
        impactStrength = askFloat("Impact strength (0..2)", impactStrength);

        baseScale = askFloat("Base cellular scale (frequency)", baseScale);
        baseJitter = askFloat("Base jitter", baseJitter);
        baseCrackWidth = askFloat("Base crack width", baseCrackWidth);
        baseAniso = askFloat("Base anisotropy 0..1", baseAniso);

        matBrittle = askFloat("Material brittleness 0..1", matBrittle);
        matToughness = askFloat("Material toughness 0..1", matToughness);
    }

    // normalize impactDir once here
    if (glm::length(impactDir) < 1e-4f) impactDir = glm::vec3(0.0f, 0.0f, -1.0f);
    impactDir = glm::normalize(impactDir);

    glm::vec2 stressDir2D(impactDir.x, impactDir.y);
    if (glm::length(stressDir2D) < 1e-3f) stressDir2D = glm::vec2(1.0f, 0.0f);
    stressDir2D = glm::normalize(stressDir2D);

    std::cout << "\n--- ใช้ค่าต่อไปนี้ ---\n";
    std::cout << "ImpactPos      : (" << impactPos.x << ", " << impactPos.y << ", " << impactPos.z << ")\n";
    std::cout << "ImpactDir      : (" << impactDir.x << ", " << impactDir.y << ", " << impactDir.z << ")\n";
    std::cout << "ImpactStrength : " << impactStrength << "\n";
    std::cout << "Scale/Jitter   : " << baseScale << " / " << baseJitter << "\n";
    std::cout << "CrackWidth     : " << baseCrackWidth << "\n";
    std::cout << "Anisotropy     : " << baseAniso << "\n";
    std::cout << "Brittle/Tough  : " << matBrittle << " / " << matToughness << "\n\n";

    // ---------- 2. Init OpenGL / GLFW ----------
    if (!glfwInit()) return -1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* win = glfwCreateWindow(1280, 720, "Crack Demo", nullptr, nullptr);
    if (!win) return -1;
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);

    if (!gladLoadGL(glfwGetProcAddress)) {
        std::fprintf(stderr, "GLAD load failed\n");
        return -1;
    }

    glfwSetInputMode(win, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetCursorPosCallback(win, cursorPosCallback);
    glfwSetScrollCallback(win, scrollCallback);

    GLuint prog = makeProg(VS, FS);
    GLint uMVP = glGetUniformLocation(prog, "uMVP");
    GLint uScale = glGetUniformLocation(prog, "uScale");
    GLint uJitter = glGetUniformLocation(prog, "uJitter");
    GLint uCrackWidth = glGetUniformLocation(prog, "uCrackWidth");
    GLint uStressDir = glGetUniformLocation(prog, "uStressDir");
    GLint uAniso = glGetUniformLocation(prog, "uAniso");
    GLint uImpactPos = glGetUniformLocation(prog, "uImpactPos");
    GLint uImpactDir = glGetUniformLocation(prog, "uImpactDir");
    GLint uImpactStr = glGetUniformLocation(prog, "uImpactStrength");
    GLint uMatBrittle = glGetUniformLocation(prog, "uMatBrittle");
    GLint uMatToughness = glGetUniformLocation(prog, "uMatToughness");

    Mesh cube = makeColoredCube();
    glEnable(GL_DEPTH_TEST);

    // ---------- 3. Render loop + timing ----------
    double cpuAccumSec = 0.0;
    long long frameCounter = 0;
    double lastTitleUpdate = glfwGetTime();

    while (!glfwWindowShouldClose(win)) {
        double frameStart = glfwGetTime();

        float now = float(frameStart);
        deltaTime = now - lastFrame;
        lastFrame = now;
        processInput(win);

        int w, h;
        glfwGetFramebufferSize(win, &w, &h);
        glViewport(0, 0, w, h);
        glClearColor(0.08f, 0.09f, 0.12f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glm::mat4 P = glm::perspective(glm::radians(fov),
            w > 0 ? (float)w / (float)h : 1.f, 0.1f, 100.f);
        glm::mat4 Vw = glm::lookAt(camPos, camPos + camFront, camUp);
        glm::mat4 M = glm::mat4(1.0f);
        glm::mat4 MVP = P * Vw * M;

        glUseProgram(prog);
        glUniformMatrix4fv(uMVP, 1, GL_FALSE, &MVP[0][0]);
        glUniform1f(uScale, baseScale);
        glUniform1f(uJitter, baseJitter);
        glUniform1f(uCrackWidth, baseCrackWidth);
        glUniform2f(uStressDir, stressDir2D.x, stressDir2D.y);
        glUniform1f(uAniso, baseAniso);

        glUniform3f(uImpactPos, impactPos.x, impactPos.y, impactPos.z);
        glUniform3f(uImpactDir, impactDir.x, impactDir.y, impactDir.z);
        glUniform1f(uImpactStr, impactStrength);

        glUniform1f(uMatBrittle, matBrittle);
        glUniform1f(uMatToughness, matToughness);

        glBindVertexArray(cube.vao);
        glDrawElements(GL_TRIANGLES, cube.indexCount, GL_UNSIGNED_INT, 0);

        glfwSwapBuffers(win);
        glfwPollEvents();

        double frameEnd = glfwGetTime();
        double cpuFrame = frameEnd - frameStart;
        cpuAccumSec += cpuFrame;
        ++frameCounter;

        // อัปเดต title ทุก ๆ ~1 วินาที
        if (frameEnd - lastTitleUpdate > 1.0 && frameCounter > 0) {
            double avgMs = (cpuAccumSec / double(frameCounter)) * 1000.0;
            char title[128];
            std::snprintf(title, sizeof(title),
                "Crack Demo  |  avg CPU frame: %.3f ms", avgMs);
            glfwSetWindowTitle(win, title);
            lastTitleUpdate = frameEnd;
        }
    }

    if (frameCounter > 0) {
        double avgMs = (cpuAccumSec / double(frameCounter)) * 1000.0;
        std::cout << "\n[Timer] Total frames: " << frameCounter
            << ", avg CPU frame time: " << avgMs << " ms\n";
    }

    destroyMesh(cube);
    glfwTerminate();
    return 0;
}
