#include <stdexcept>
#include <cstdio>
#include <algorithm>
#include <cmath>

#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "mesh.hpp"
#include "cube.hpp"

// ========== Camera ==========
glm::vec3 camPos = { 0.0f, 0.0f, 4.0f };
glm::vec3 camFront = { 0.0f, 0.0f,-1.0f };
glm::vec3 camUp = { 0.0f, 1.0f, 0.0f };

float yaw = -90.f;
float pitch = 0.f;
float fov = 60.f;

float deltaTime = 0.f;
float lastFrame = 0.f;

bool  firstMouse = true;
double lastX = 0.0, lastY = 0.0;

// ========== Multi-crack state ==========
static const int MAX_CRACKS = 16;

int       gCrackCount = 0;
glm::vec2 gCrackUV[MAX_CRACKS]; // center UV (debug)
int       gCrackFace[MAX_CRACKS]; // face id 0..5
float     gCrackRadius[MAX_CRACKS];// radius (local/world)
float     gCrackSeed[MAX_CRACKS]; // seed

glm::vec3 gCrackCenter[MAX_CRACKS];
glm::vec3 gCrackU[MAX_CRACKS];
glm::vec3 gCrackV[MAX_CRACKS];

// ========== Utils ==========
static void die(const char* msg) {
    throw std::runtime_error(msg);
}

static GLuint makeProg(const char* vs, const char* fs) {
    auto comp = [&](GLenum type, const char* src) {
        GLuint sh = glCreateShader(type);
        glShaderSource(sh, 1, &src, nullptr);
        glCompileShader(sh);
        GLint ok = GL_FALSE;
        glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            char log[2048];
            glGetShaderInfoLog(sh, sizeof(log), nullptr, log);
            std::fprintf(stderr, "Shader compile error (%s):\n%s\n",
                type == GL_VERTEX_SHADER ? "VS" : "FS", log);
            die("compile failed");
        }
        return sh;
        };

    GLuint v = comp(GL_VERTEX_SHADER, vs);
    GLuint f = comp(GL_FRAGMENT_SHADER, fs);

    GLuint p = glCreateProgram();
    glAttachShader(p, v);
    glAttachShader(p, f);
    glLinkProgram(p);
    GLint ok = GL_FALSE;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetProgramInfoLog(p, sizeof(log), nullptr, log);
        std::fprintf(stderr, "Program link error:\n%s\n", log);
        die("link failed");
    }

    glDeleteShader(v);
    glDeleteShader(f);
    return p;
}

// Ray–AABB intersection for cube at [-1,1]^3 in world space
bool intersectRayAABB(const glm::vec3& orig,
    const glm::vec3& dir,
    const glm::vec3& bmin,
    const glm::vec3& bmax,
    float& tHit)
{
    float tmin = 0.0f;
    float tmax = 1e6f;

    for (int i = 0; i < 3; ++i) {
        float o = orig[i];
        float d = dir[i];

        if (std::abs(d) < 1e-8f) {
            if (o < bmin[i] || o > bmax[i]) return false;
            continue;
        }

        float invD = 1.0f / d;
        float t0 = (bmin[i] - o) * invD;
        float t1 = (bmax[i] - o) * invD;
        if (invD < 0.0f) std::swap(t0, t1);

        if (t0 > tmin) tmin = t0;
        if (t1 < tmax) tmax = t1;
        if (tmax <= tmin)  return false;
    }

    tHit = tmin;
    return true;
}

// ========== Input handling ==========
void processInput(GLFWwindow* w) {
    float sp = 2.5f * deltaTime;
    if (glfwGetKey(w, GLFW_KEY_W) == GLFW_PRESS) camPos += sp * camFront;
    if (glfwGetKey(w, GLFW_KEY_S) == GLFW_PRESS) camPos -= sp * camFront;
    if (glfwGetKey(w, GLFW_KEY_A) == GLFW_PRESS)
        camPos -= glm::normalize(glm::cross(camFront, camUp)) * sp;
    if (glfwGetKey(w, GLFW_KEY_D) == GLFW_PRESS)
        camPos += glm::normalize(glm::cross(camFront, camUp)) * sp;
    if (glfwGetKey(w, GLFW_KEY_E) == GLFW_PRESS) camPos += sp * camUp;
    if (glfwGetKey(w, GLFW_KEY_Q) == GLFW_PRESS) camPos -= sp * camUp;

    if (glfwGetKey(w, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(w, GLFW_TRUE);
}

// Rotate camera only while right mouse button is pressed
void cursorPosCallback(GLFWwindow* window, double x, double y) {
    static const float S = 0.1f;

    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) != GLFW_PRESS) {
        firstMouse = true;
        return;
    }

    if (firstMouse) {
        lastX = x;
        lastY = y;
        firstMouse = false;
        return;
    }

    float xoff = float(x - lastX) * S;
    float yoff = float(lastY - y) * S;
    lastX = x;
    lastY = y;

    yaw += xoff;
    pitch += yoff;
    if (pitch > 89.f)  pitch = 89.f;
    if (pitch < -89.f) pitch = -89.f;

    glm::vec3 d;
    d.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    d.y = sin(glm::radians(pitch));
    d.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    camFront = glm::normalize(d);
}

void scrollCallback(GLFWwindow*, double, double yoff) {
    fov -= float(yoff);
    if (fov < 1.f) fov = 1.f;
    if (fov > 90.f) fov = 90.f;
}

// Left click: shoot ray, find hit point on cube, append a crack
void mouseButtonCallback(GLFWwindow* window, int button, int action, int /*mods*/) {
    if (button != GLFW_MOUSE_BUTTON_LEFT || action != GLFW_PRESS)
        return;

    double mouseX, mouseY;
    glfwGetCursorPos(window, &mouseX, &mouseY);

    int winW, winH;
    glfwGetWindowSize(window, &winW, &winH);
    if (winW <= 0 || winH <= 0) return;

    float aspect = (winH > 0) ? (float)winW / (float)winH : 1.0f;
    glm::mat4 P = glm::perspective(glm::radians(fov), aspect, 0.1f, 100.f);
    glm::mat4 viewM = glm::lookAt(camPos, camPos + camFront, camUp);

    // pixel -> NDC
    float x = static_cast<float>(mouseX);
    float y = static_cast<float>(mouseY);
    float ndcX = 2.0f * x / (float)winW - 1.0f;
    float ndcY = 1.0f - 2.0f * y / (float)winH;

    glm::vec4 rayClip(ndcX, ndcY, -1.0f, 1.0f);
    glm::mat4 invProj = glm::inverse(P);
    glm::mat4 invView = glm::inverse(viewM);

    glm::vec4 rayEye = invProj * rayClip;
    rayEye = glm::vec4(rayEye.x, rayEye.y, -1.0f, 0.0f);
    glm::vec4 rayWorld = invView * rayEye;

    glm::vec3 rayDir = glm::normalize(glm::vec3(rayWorld));
    glm::vec3 rayOrig = camPos;

    float tHit;
    if (!intersectRayAABB(rayOrig, rayDir,
        glm::vec3(-1.0f, -1.0f, -1.0f),
        glm::vec3(1.0f, 1.0f, 1.0f),
        tHit)) {
        std::printf("Click: no hit (ray misses cube)\n");
        return;
    }

    glm::vec3 p = rayOrig + tHit * rayDir;
    glm::vec3 ap = glm::abs(p);

    int       faceId;
    glm::vec2 uv;

    if (ap.x >= ap.y && ap.x >= ap.z) {
        faceId = (p.x >= 0.0f) ? 0 : 1;
        uv = glm::vec2((p.z + 1.0f) * 0.5f,
            (p.y + 1.0f) * 0.5f);
    }
    else if (ap.y >= ap.x && ap.y >= ap.z) {
        faceId = (p.y >= 0.0f) ? 2 : 3;
        uv = glm::vec2((p.x + 1.0f) * 0.5f,
            (p.z + 1.0f) * 0.5f);
    }
    else {
        faceId = (p.z >= 0.0f) ? 4 : 5;
        uv = glm::vec2((p.x + 1.0f) * 0.5f,
            (p.y + 1.0f) * 0.5f);
    }

    uv.x = glm::clamp(uv.x, 0.0f, 1.0f);
    uv.y = glm::clamp(uv.y, 0.0f, 1.0f);

    glm::vec3 n(0.0f);
    switch (faceId) {
    case 0: n = glm::vec3(1, 0, 0); break;
    case 1: n = glm::vec3(-1, 0, 0); break;
    case 2: n = glm::vec3(0, 1, 0); break;
    case 3: n = glm::vec3(0, -1, 0); break;
    case 4: n = glm::vec3(0, 0, 1); break;
    case 5: n = glm::vec3(0, 0, -1); break;
    }

    glm::vec3 ref = (std::abs(n.z) < 0.9f) ? glm::vec3(0, 0, 1)
        : glm::vec3(0, 1, 0);
    glm::vec3 tanU = glm::normalize(glm::cross(ref, n));
    glm::vec3 tanV = glm::normalize(glm::cross(n, tanU));

    if (gCrackCount < MAX_CRACKS) {
        int i = gCrackCount++;
        gCrackUV[i] = uv;
        gCrackFace[i] = faceId;
        gCrackRadius[i] = 0.75f;
        float t = float(glfwGetTime());
        gCrackSeed[i] = t * 3.17f + float(i) * 11.31f;

        gCrackCenter[i] = p;
        gCrackU[i] = tanU;
        gCrackV[i] = tanV;
    }
    else {
        static int overwriteIdx = 0;
        int i = overwriteIdx;
        overwriteIdx = (overwriteIdx + 1) % MAX_CRACKS;

        gCrackUV[i] = uv;
        gCrackFace[i] = faceId;
        gCrackRadius[i] = 0.75f;
        float t = float(glfwGetTime());
        gCrackSeed[i] = t * 3.17f + float(i) * 11.31f;

        gCrackCenter[i] = p;
        gCrackU[i] = tanU;
        gCrackV[i] = tanV;
    }

    std::printf(
        "Click hit cube: worldPos=(%.3f, %.3f, %.3f), face=%d, uv=(%.3f, %.3f), totalCracks=%d\n",
        p.x, p.y, p.z, faceId, uv.x, uv.y, gCrackCount
    );
}

// ========== Shaders ==========

static const char* VS = R"(
#version 450 core
layout(location=0) in vec3 aPos;

out vec3 vPos;
out vec2 vUV;
flat out int vFaceId;

uniform mat4 uMVP;

void main() {
    vPos = aPos;

    vec3 ap = abs(aPos);
    int   faceId;
    vec2  uv;

    if (ap.x >= ap.y && ap.x >= ap.z) {
        faceId = (aPos.x >= 0.0) ? 0 : 1;
        uv     = vec2((aPos.z + 1.0) * 0.5,
                      (aPos.y + 1.0) * 0.5);
    } else if (ap.y >= ap.x && ap.y >= ap.z) {
        faceId = (aPos.y >= 0.0) ? 2 : 3;
        uv     = vec2((aPos.x + 1.0) * 0.5,
                      (aPos.z + 1.0) * 0.5);
    } else {
        faceId = (aPos.z >= 0.0) ? 4 : 5;
        uv     = vec2((aPos.x + 1.0) * 0.5,
                      (aPos.y + 1.0) * 0.5);
    }

    vFaceId = faceId;
    vUV     = uv;

    gl_Position = uMVP * vec4(aPos, 1.0);
}
)";

static const char* FS = R"(
#version 450 core
in vec3 vPos;
in vec2 vUV;
flat in int vFaceId;

out vec4 FragColor;

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
    float s = rand2(p+19.19);
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

vec2 worleyF(vec2 p){
    vec2 ip=floor(p), fp=fract(p);
    float F1=1e9, F2=1e9;
    for(int j=-1;j<=1;j++){
        for(int i=-1;i<=1;i++){
            vec2 cell = ip + vec2(i,j);
            vec2 r2   = rand2_2(cell) - 0.5;
            vec2 d    = (vec2(i,j) + r2) - fp;
            float dist = dot(d,d);
            if(dist < F1){ F2=F1; F1=dist; }
            else if(dist < F2){ F2=dist; }
        }
    }
    return vec2(sqrt(F1), sqrt(F2));
}

mat2 rot(float a){ float c=cos(a), s=sin(a); return mat2(c,-s,s,c); }

// global controls
uniform float uScale;
uniform float uJitter;
uniform float uCrackWidth;
uniform vec2  uStressDir;
uniform float uAniso;
uniform float uTime;

// multi-crack data
#define MAX_CRACKS 16
uniform int   uCrackCount;
uniform vec2  uCrackUV    [MAX_CRACKS];
uniform float uCrackRadius[MAX_CRACKS];
uniform int   uCrackFace  [MAX_CRACKS];
uniform float uCrackSeed  [MAX_CRACKS];

uniform vec3  uCrackCenter[MAX_CRACKS];
uniform vec3  uCrackU     [MAX_CRACKS];
uniform vec3  uCrackV     [MAX_CRACKS];

vec2 stressWarp(vec2 coord, float seed){
    float ang = atan(uStressDir.y, uStressDir.x);
    mat2 R    = rot(ang);
    mat2 Rinv = rot(-ang);

    vec2 q = R * coord;
    q.y   *= (1.0 - 0.6*uAniso);
    q     = Rinv * q;

    float j = uJitter * (fbm(coord*1.3 + 7.7 + seed) - 0.5);
    return q + j;
}

void main(){
    vec3 baseColor = vec3(0.72, 0.69, 0.65);
    float totalCrack = 0.0;

    for (int i = 0; i < uCrackCount; ++i) {
        vec3 center = uCrackCenter[i];
        float radius = uCrackRadius[i];
        float seed   = uCrackSeed[i];

        // local tangent plane coords
        vec3 d3 = vPos - center;
        vec2 local = vec2(dot(d3, uCrackU[i]),
                          dot(d3, uCrackV[i]));
        float r = length(local);
        if (r < 1e-6) r = 1e-6;

        float baseRadius = max(radius, 1e-4);

        // ====== domain / seed density แบบเน้น center (ไม่มี angle-based radial distortion) ======
        float rNorm = clamp(r / baseRadius, 0.0, 1.0);
        float densityScale = mix(2.0, 0.6, rNorm);

        vec2 coord = local * (uScale * densityScale)
                   + vec2(seed * 0.73, seed * 1.41);
        // =======================================================================

        vec2 p = stressWarp(coord, seed);
        vec2 F = worleyF(p);
        float edge = F.y - F.x;

        float w = uCrackWidth;
        float crack = 1.0 - smoothstep(0.0, w, edge);
        float micro = 1.0 - smoothstep(0.0, w*0.5, abs(edge-0.02));
        float crackBase = clamp(crack + 0.5*micro, 0.0, 1.0);

        // ====== radial mask กลับไปแบบเวอร์ชันเดิม (วงกลม + hardFalloff + centerBoost) ======
        float innerR = 0.0;
        float outerR = baseRadius;

        float radial      = 1.0 - smoothstep(innerR, outerR, r);
        float hardFalloff = 1.0 - smoothstep(outerR, outerR*1.6, r);
        // =======================================================================

        // normal mask: จำกัดรอยแตกใกล้ระนาบหน้าที่คลิก (thickness = 0.01)
        vec3 n = normalize(cross(uCrackU[i], uCrackV[i]));
        float dN = dot(d3, n);          // distance along normal
        float thickness = 0.01;         // ตามที่ขอ
        float normalMask = 1.0 - smoothstep(thickness*0.4,
                                            thickness,
                                            abs(dN));

        float mask = radial * hardFalloff * normalMask;

        float centerBoost = pow(1.0 - clamp(r / (outerR + 1e-4), 0.0, 1.0), 0.35);
        float crackAmount = crackBase * mask * (0.4 + 0.6 * centerBoost);
        // =======================================================================

        totalCrack = max(totalCrack, crackAmount);
    }

    vec3 c = mix(baseColor, vec3(0.05), totalCrack);

    if (uCrackCount > 0 && totalCrack > 0.0) {
        vec2 localApprox = vUV * uScale;
        vec2 p0 = stressWarp(localApprox, uCrackSeed[0]);
        vec2 F0 = worleyF(p0);
        float edge0 = F0.y - F0.x;

        float e = 0.002;
        vec2 px = vec2(e,0), py=vec2(0,e);
        vec2 Fx = worleyF(p0+px);
        vec2 Fy = worleyF(p0+py);
        float ex = (Fx.y-Fx.x) - edge0;
        float ey = (Fy.y-Fy.x) - edge0;

        vec3 L = normalize(vec3(0.5, 0.8, 0.6));
        vec3 N = normalize(vec3(-ex, -ey, 1.0));
        float diff = max(dot(N, L), 0.0);
        c *= (0.35 + 0.65*diff);
    }

    FragColor = vec4(c, 1.0);
}
)";

// ========== main() ==========
int main() {
    if (!glfwInit()) return -1;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* win = glfwCreateWindow(
        1280, 720,
        "Procedural Cracks on Cube (radial reverted, thin slab)",
        nullptr, nullptr
    );
    if (!win) return -1;
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);

    if (!gladLoadGL(glfwGetProcAddress)) {
        std::fprintf(stderr, "GLAD load failed\n");
        return -1;
    }

    glfwSetInputMode(win, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    glfwSetCursorPosCallback(win, cursorPosCallback);
    glfwSetScrollCallback(win, scrollCallback);
    glfwSetMouseButtonCallback(win, mouseButtonCallback);

    GLuint prog = makeProg(VS, FS);

    GLint uMVP = glGetUniformLocation(prog, "uMVP");
    GLint uScale = glGetUniformLocation(prog, "uScale");
    GLint uJitter = glGetUniformLocation(prog, "uJitter");
    GLint uCrackWidth = glGetUniformLocation(prog, "uCrackWidth");
    GLint uStressDir = glGetUniformLocation(prog, "uStressDir");
    GLint uAniso = glGetUniformLocation(prog, "uAniso");
    GLint uTime = glGetUniformLocation(prog, "uTime");

    GLint uCrackCountLoc = glGetUniformLocation(prog, "uCrackCount");
    GLint uCrackUVLoc = glGetUniformLocation(prog, "uCrackUV");
    GLint uCrackRadiusLoc = glGetUniformLocation(prog, "uCrackRadius");
    GLint uCrackFaceLoc = glGetUniformLocation(prog, "uCrackFace");
    GLint uCrackSeedLoc = glGetUniformLocation(prog, "uCrackSeed");

    GLint uCrackCenterLoc = glGetUniformLocation(prog, "uCrackCenter");
    GLint uCrackULoc = glGetUniformLocation(prog, "uCrackU");
    GLint uCrackVLoc = glGetUniformLocation(prog, "uCrackV");

    Mesh cube = makeColoredCube();
    glEnable(GL_DEPTH_TEST);

    float scale = 10.0f;
    float jitter = 0.30f;
    float crackWidth = 0.040f;
    glm::vec2 stressDir = glm::normalize(glm::vec2(1.0f, 0.3f));
    float aniso = 0.6f;

    while (!glfwWindowShouldClose(win)) {
        float now = float(glfwGetTime());
        deltaTime = now - lastFrame;
        lastFrame = now;

        processInput(win);

        int fbW, fbH;
        glfwGetFramebufferSize(win, &fbW, &fbH);
        glViewport(0, 0, fbW, fbH);
        glClearColor(0.08f, 0.09f, 0.12f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        float aspect = (fbH > 0) ? (float)fbW / (float)fbH : 1.0f;
        glm::mat4 P = glm::perspective(glm::radians(fov), aspect, 0.1f, 100.f);
        glm::mat4 viewM = glm::lookAt(camPos, camPos + camFront, camUp);
        glm::mat4 M = glm::mat4(1.0f);
        glm::mat4 MVP = P * viewM * M;

        glUseProgram(prog);
        glUniformMatrix4fv(uMVP, 1, GL_FALSE, &MVP[0][0]);

        glUniform1f(uScale, scale);
        glUniform1f(uJitter, jitter);
        glUniform1f(uCrackWidth, crackWidth);
        glUniform2f(uStressDir, stressDir.x, stressDir.y);
        glUniform1f(uAniso, aniso);
        glUniform1f(uTime, now);

        glUniform1i(uCrackCountLoc, gCrackCount);
        if (gCrackCount > 0) {
            glUniform2fv(uCrackUVLoc, gCrackCount, &gCrackUV[0].x);
            glUniform1fv(uCrackRadiusLoc, gCrackCount, gCrackRadius);
            glUniform1iv(uCrackFaceLoc, gCrackCount, gCrackFace);
            glUniform1fv(uCrackSeedLoc, gCrackCount, gCrackSeed);

            glUniform3fv(uCrackCenterLoc, gCrackCount, &gCrackCenter[0].x);
            glUniform3fv(uCrackULoc, gCrackCount, &gCrackU[0].x);
            glUniform3fv(uCrackVLoc, gCrackCount, &gCrackV[0].x);
        }

        glBindVertexArray(cube.vao);
        glDrawElements(GL_TRIANGLES, cube.indexCount, GL_UNSIGNED_INT, 0);

        glfwSwapBuffers(win);
        glfwPollEvents();
    }

    destroyMesh(cube);
    glfwTerminate();
    return 0;
}
