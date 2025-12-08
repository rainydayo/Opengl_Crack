#include <stdexcept>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <iostream>
#include <string>

#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "mesh.hpp"
#include "cube.hpp"
#include "sphere.hpp"

// ========== Global Configuration ==========
enum ShapeType {
    SHAPE_CUBE = 1,
    SHAPE_SPHERE = 2
};

ShapeType gCurrentShape = SHAPE_CUBE;

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

// ========== Crack parameters ==========
struct CrackParams {
    float seedDensity;
    float baseRadius;
    float jitter;
    float crackWidth;
    float aniso;
    float heightScale; // POM Depth
};

CrackParams gCrackParams;

// ========== Multi-crack state ==========
static const int MAX_CRACKS = 16;

int       gCrackCount = 0;
float     gCrackRadius[MAX_CRACKS];
float     gCrackSeed[MAX_CRACKS];

glm::vec3 gCrackCenter[MAX_CRACKS];
glm::vec3 gCrackNormal[MAX_CRACKS];
glm::vec3 gCrackU[MAX_CRACKS];
glm::vec3 gCrackV[MAX_CRACKS];

// ========== Lighting ==========
glm::vec3 lightPos(2.0f, 2.0f, 3.0f);
glm::vec3 lightColor(1.0f, 1.0f, 1.0f);

// ========== Timer ==========
bool gCrackTimingPending = false;
std::chrono::high_resolution_clock::time_point gCrackStartTime;

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

// Ray Intersection Functions
bool intersectRayAABB(const glm::vec3& orig, const glm::vec3& dir, const glm::vec3& bmin, const glm::vec3& bmax, float& tHit) {
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

bool intersectRaySphere(const glm::vec3& orig, const glm::vec3& dir, const glm::vec3& center, float radius, float& tHit) {
    glm::vec3 oc = orig - center;
    float b = dot(oc, dir);
    float c = dot(oc, oc) - radius * radius;
    float h = b * b - c;
    if (h < 0.0) return false;
    h = sqrt(h);
    float t = -b - h;
    if (t < 0.0) t = -b + h;
    if (t < 0.0) return false;
    tHit = t;
    return true;
}

void initDefaultCrackParams() {
    gCrackParams.seedDensity = 10.0f;
    gCrackParams.baseRadius = 0.75f;
    gCrackParams.jitter = 0.50f;
    gCrackParams.crackWidth = 0.045f;
    gCrackParams.aniso = 0.60f;
    gCrackParams.heightScale = 0.05f; // POM Depth
}

float readFloatParam(const std::string& label, float defVal) {
    std::cout << label << " [" << defVal << "] : ";
    std::string line;
    if (!std::getline(std::cin, line) || line.empty()) return defVal;
    try { return std::stof(line); }
    catch (...) { return defVal; }
}

void configureInput() {
    std::cout << "=== Mesh Selection ===\n";
    std::cout << "1. Cube\n";
    std::cout << "2. Sphere\n";
    std::cout << "Select (1 or 2) [Default 1]: ";
    std::string line;
    std::getline(std::cin, line);
    if (line == "2") { gCurrentShape = SHAPE_SPHERE; std::cout << "Selected: Sphere\n"; }
    else { gCurrentShape = SHAPE_CUBE; std::cout << "Selected: Cube\n"; }

    std::cout << "\n=== Crack configuration ===\n";
    gCrackParams.seedDensity = readFloatParam("Seed density", gCrackParams.seedDensity);
    gCrackParams.baseRadius = readFloatParam("Base radius", gCrackParams.baseRadius);
    gCrackParams.jitter = readFloatParam("Noise / jitter", gCrackParams.jitter);
    gCrackParams.crackWidth = readFloatParam("Crack width", gCrackParams.crackWidth);
    gCrackParams.aniso = readFloatParam("Anisotropy", gCrackParams.aniso);
    gCrackParams.heightScale = readFloatParam("Height Scale (POM Depth)", gCrackParams.heightScale);
    std::cout << "\nReady. Click on the mesh to create cracks.\n\n";
}

// ========== Input / Camera ==========
void processInput(GLFWwindow* w) {
    float sp = 2.5f * deltaTime;
    if (glfwGetKey(w, GLFW_KEY_W) == GLFW_PRESS) camPos += sp * camFront;
    if (glfwGetKey(w, GLFW_KEY_S) == GLFW_PRESS) camPos -= sp * camFront;
    if (glfwGetKey(w, GLFW_KEY_A) == GLFW_PRESS) camPos -= glm::normalize(glm::cross(camFront, camUp)) * sp;
    if (glfwGetKey(w, GLFW_KEY_D) == GLFW_PRESS) camPos += glm::normalize(glm::cross(camFront, camUp)) * sp;
    if (glfwGetKey(w, GLFW_KEY_E) == GLFW_PRESS) camPos += sp * camUp;
    if (glfwGetKey(w, GLFW_KEY_Q) == GLFW_PRESS) camPos -= sp * camUp;
    if (glfwGetKey(w, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(w, GLFW_TRUE);
}

void cursorPosCallback(GLFWwindow* window, double x, double y) {
    static const float S = 0.1f;
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) != GLFW_PRESS) { firstMouse = true; return; }
    if (firstMouse) { lastX = x; lastY = y; firstMouse = false; return; }
    float xoff = float(x - lastX) * S;
    float yoff = float(lastY - y) * S;
    lastX = x; lastY = y;
    yaw += xoff; pitch += yoff;
    if (pitch > 89.f) pitch = 89.f; if (pitch < -89.f) pitch = -89.f;
    glm::vec3 d;
    d.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    d.y = sin(glm::radians(pitch));
    d.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    camFront = glm::normalize(d);
}

void scrollCallback(GLFWwindow*, double, double yoff) {
    fov -= float(yoff);
    if (fov < 1.f) fov = 1.f; if (fov > 90.f) fov = 90.f;
}

// ========== Mouse Click ==========
void mouseButtonCallback(GLFWwindow* window, int button, int action, int /*mods*/) {
    if (button != GLFW_MOUSE_BUTTON_LEFT || action != GLFW_PRESS) return;

    double mouseX, mouseY;
    glfwGetCursorPos(window, &mouseX, &mouseY);
    int winW, winH;
    glfwGetWindowSize(window, &winW, &winH);
    if (winW <= 0 || winH <= 0) return;

    float aspect = (winH > 0) ? (float)winW / (float)winH : 1.0f;
    glm::mat4 P = glm::perspective(glm::radians(fov), aspect, 0.1f, 100.f);
    glm::mat4 viewM = glm::lookAt(camPos, camPos + camFront, camUp);

    float x = static_cast<float>(mouseX);
    float y = static_cast<float>(mouseY);
    float ndcX = 2.0f * x / (float)winW - 1.0f;
    float ndcY = 1.0f - 2.0f * y / (float)winH;

    glm::vec4 rayClip(ndcX, ndcY, -1.0f, 1.0f);
    glm::mat4 invProj = glm::inverse(P);
    glm::mat4 invView = glm::inverse(viewM);
    glm::vec4 rayEye = invProj * rayClip; rayEye = glm::vec4(rayEye.x, rayEye.y, -1.0f, 0.0f);
    glm::vec4 rayWorld = invView * rayEye;
    glm::vec3 rayDir = glm::normalize(glm::vec3(rayWorld));
    glm::vec3 rayOrig = camPos;

    float tHit = 0.0f;
    bool hit = false;
    glm::vec3 hitNormal(0.0f);

    if (gCurrentShape == SHAPE_CUBE) {
        hit = intersectRayAABB(rayOrig, rayDir, glm::vec3(-1.0f), glm::vec3(1.0f), tHit);
        if (hit) {
            glm::vec3 p = rayOrig + tHit * rayDir;
            glm::vec3 ap = glm::abs(p);
            if (ap.x >= ap.y && ap.x >= ap.z) hitNormal = glm::vec3(p.x > 0 ? 1 : -1, 0, 0);
            else if (ap.y >= ap.x && ap.y >= ap.z) hitNormal = glm::vec3(0, p.y > 0 ? 1 : -1, 0);
            else hitNormal = glm::vec3(0, 0, p.z > 0 ? 1 : -1);
        }
    }
    else {
        hit = intersectRaySphere(rayOrig, rayDir, glm::vec3(0.0f), 1.0f, tHit);
        if (hit) {
            glm::vec3 p = rayOrig + tHit * rayDir;
            hitNormal = glm::normalize(p);
        }
    }

    if (!hit) return;

    glm::vec3 p = rayOrig + tHit * rayDir;
    glm::vec3 ref = (std::abs(hitNormal.z) < 0.9f) ? glm::vec3(0, 0, 1) : glm::vec3(0, 1, 0);
    glm::vec3 tanU = glm::normalize(glm::cross(ref, hitNormal));
    glm::vec3 tanV = glm::normalize(glm::cross(hitNormal, tanU));

    if (gCrackCount < MAX_CRACKS) {
        int i = gCrackCount++;
        gCrackRadius[i] = gCrackParams.baseRadius;
        gCrackSeed[i] = float(glfwGetTime()) * 3.17f + float(i) * 11.31f;
        gCrackCenter[i] = p;
        gCrackNormal[i] = hitNormal;
        gCrackU[i] = tanU;
        gCrackV[i] = tanV;
    }
    else {
        static int overwriteIdx = 0;
        int i = overwriteIdx;
        overwriteIdx = (overwriteIdx + 1) % MAX_CRACKS;
        gCrackRadius[i] = gCrackParams.baseRadius;
        gCrackSeed[i] = float(glfwGetTime()) * 3.17f + float(i) * 11.31f;
        gCrackCenter[i] = p;
        gCrackNormal[i] = hitNormal;
        gCrackU[i] = tanU;
        gCrackV[i] = tanV;
    }

    gCrackTimingPending = true;
    gCrackStartTime = std::chrono::high_resolution_clock::now();
    std::printf("Crack added at (%.2f, %.2f, %.2f). Total: %d\n", p.x, p.y, p.z, gCrackCount);
}

// ========== Shaders (Fixed Flow + POM) ==========

static const char* VS = R"(
#version 450 core
layout(location=0) in vec3 aPos;
out vec3 vPos;
uniform mat4 uMVP;
void main() {
    vPos = aPos;
    gl_Position = uMVP * vec4(aPos, 1.0);
}
)";

static const char* FS = R"(
#version 450 core
in vec3 vPos;
out vec4 FragColor;

// --- Noise Functions ---
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

// Global Uniforms
uniform float uScale;
uniform float uJitter;
uniform float uCrackWidth;
uniform vec2  uStressDir;
uniform float uAniso;
uniform float uHeightScale; // POM Scale
uniform vec3  uCamPos;
uniform vec3  uLightPos;
uniform vec3  uLightColor;

#define MAX_CRACKS 16
uniform int   uCrackCount;
uniform float uCrackRadius[MAX_CRACKS];
uniform float uCrackSeed  [MAX_CRACKS];
uniform vec3  uCrackCenter[MAX_CRACKS];
uniform vec3  uCrackNormal[MAX_CRACKS];
uniform vec3  uCrackU     [MAX_CRACKS];
uniform vec3  uCrackV     [MAX_CRACKS];

// --- Stress Warp ---
vec2 stressWarp(vec2 uv, float seed){
    float ang = atan(uStressDir.y, uStressDir.x);
    mat2 R = rot(ang), Rinv = rot(-ang);
    vec2 q = R * uv; 
    q.y *= (1.0 - 0.6*uAniso);
    q = Rinv * q;
    float j = uJitter * (fbm(uv*1.3 + 7.7 + seed) - 0.5);
    return q + j;
}

// --- Heightmap Function (Updated to use rNorm from 3D distance) ---
// uv: The projected texture coordinate
// rNorm: The NORMALIZED radius based on TRUE 3D DISTANCE (not UV length)
float getHeight(vec2 uv, float rNorm, float seed){
    
    // Calculate Radial info from local UV (Direction only)
    float rProj = length(uv) + 1e-6;
    vec2 nRad = uv / rProj;
    vec2 nTan = vec2(-nRad.y, nRad.x);
    
    // Decompose UV into Radial and Tangential components
    float rComp = dot(uv, nRad);
    float tComp = dot(uv, nTan);
    
    // --- KEY FIX FROM MAIN0: Use rNorm (True 3D) for Compression ---
    // If we used length(uv) here, it would stretch on cube sides.
    float tCompress = mix(1.0, 0.35, smoothstep(0.6, 1.0, rNorm));
    
    // Reconstruct coordinate with "flow" compression
    vec2 localAniso = nRad * rComp + nTan * (tComp * tCompress);

    // Density Scale also driven by True 3D distance
    float densityScale = mix(2.0, 0.6, rNorm);
    
    vec2 coord = localAniso * (uScale * densityScale) + vec2(seed * 0.73, seed * 1.41);

    // Stress Warp + Jitter 
    vec2 p = stressWarp(coord, seed); 

    // Worley Noise
    vec2 F = worleyF(p);
    float edge = F.y - F.x;
    
    // Profile
    float crack = 1.0 - smoothstep(0.0, uCrackWidth, edge);
    float micro = 1.0 - smoothstep(0.0, uCrackWidth*0.5, abs(edge-0.02));
    
    return 1.0 - clamp(crack + 0.5*micro, 0.0, 1.0);
}

// --- Parallax Occlusion Mapping ---
// Now passes 'rNorm' through to getHeight so consistency is maintained
vec2 parallaxMapping(vec2 uv, vec3 viewDir, float seed, float rNorm){
    float minLayers = 8.0;
    float maxLayers = 32.0;
    float numLayers = mix(maxLayers, minLayers, abs(viewDir.z));
    
    float layerDepth = 1.0 / numLayers;
    float currentLayerDepth = 0.0;
    
    vec2 P = viewDir.xy / viewDir.z * uHeightScale;
    vec2 deltaTexCoords = P / numLayers;
    
    vec2 currentTexCoords = uv;
    float currentDepthMapValue = getHeight(currentTexCoords, rNorm, seed);
    
    while(currentLayerDepth < (1.0 - currentDepthMapValue)){
        currentTexCoords -= deltaTexCoords;
        currentDepthMapValue = getHeight(currentTexCoords, rNorm, seed);
        currentLayerDepth += layerDepth;
    }
    
    vec2 prevTexCoords = currentTexCoords + deltaTexCoords;
    float afterDepth  = (1.0 - currentDepthMapValue) - currentLayerDepth;
    float beforeDepth = (1.0 - getHeight(prevTexCoords, rNorm, seed)) - currentLayerDepth + layerDepth;
    
    float weight = afterDepth / (afterDepth - beforeDepth);
    vec2 finalTexCoords = prevTexCoords * weight + currentTexCoords * (1.0 - weight);
    
    return finalTexCoords;
}

void main(){
    vec3 baseColor = vec3(0.72, 0.69, 0.65);
    vec3 finalColor = baseColor;

    vec3 dX = dFdx(vPos);
    vec3 dY = dFdy(vPos);
    vec3 fragN = normalize(cross(dX, dY)); 
    vec3 viewPos = uCamPos;
    vec3 viewDirWorld = normalize(viewPos - vPos);

    // Base Lighting
    vec3 lightDir = normalize(uLightPos - vPos);
    float diffBase = max(dot(fragN, lightDir), 0.0);
    vec3 ambient = 0.1 * uLightColor;
    vec3 diffuse = diffBase * uLightColor;
    finalColor = (ambient + diffuse) * baseColor;

    for (int i = 0; i < uCrackCount; ++i) {
        vec3 center = uCrackCenter[i];
        vec3 crackN = uCrackNormal[i];
        float radius = uCrackRadius[i];
        float seed = uCrackSeed[i];

        if(dot(fragN, crackN) < -0.1) continue;

        vec3 delta = vPos - center;
        float dist3D = length(delta); // True 3D distance

        // Project onto tangent plane for UVs
        vec3 T = uCrackU[i];
        vec3 B = uCrackV[i];
        vec2 localUV = vec2(dot(delta, T), dot(delta, B));

        // 3. Random Radius & Masking
        float angDir = atan(localUV.y, localUV.x);
        float ang01  = (angDir + 3.14159265) / 6.2831853;
        float sector = floor(ang01 * 48.0);
        float dirNoise = rand2(vec2(sector + seed*7.31, seed*3.17));
        dirNoise = clamp(dirNoise, 0.0, 1.0);

        float minR     = radius * 0.25;
        float maxExtra = radius * 0.75;
        float outerR   = minR + maxExtra * dirNoise;

        float mask = 1.0 - smoothstep(outerR, outerR + 0.05, dist3D);
        if(mask <= 0.01) continue;

        // --- FIXED FLOW LOGIC ---
        // Calculate rNorm based on True 3D Distance (Like main0)
        // This ensures proper compression even on glancing angles/cube sides
        float baseRadiusSafe = max(radius, 1e-4);
        float rNorm = clamp(dist3D / baseRadiusSafe, 0.0, 1.0);

        // 4. Parallax Occlusion Mapping
        mat3 TBN = transpose(mat3(T, B, crackN)); 
        vec3 viewDirTS = normalize(TBN * viewDirWorld);

        // Pass rNorm (3D based) to POM and getHeight
        vec2 displacedUV = parallaxMapping(localUV, viewDirTS, seed, rNorm);
        
        // 5. Get Height & Normal using rNorm
        float h = getHeight(displacedUV, rNorm, seed);

        // Compute Normal from Gradient
        float e = 0.005; 
        float hL = getHeight(displacedUV - vec2(e,0), rNorm, seed);
        float hR = getHeight(displacedUV + vec2(e,0), rNorm, seed);
        float hD = getHeight(displacedUV - vec2(0,e), rNorm, seed);
        float hU = getHeight(displacedUV + vec2(0,e), rNorm, seed);
        
        vec3 normTS;
        normTS.x = (hL - hR) / (2.0*e);
        normTS.y = (hD - hU) / (2.0*e);
        normTS.z = 1.0; 
        normTS = normalize(normTS);
        
        mat3 TBN_inv = mat3(T, B, crackN); 
        vec3 normWorld = normalize(TBN_inv * normTS);
        
        // 6. Crack Lighting
        float diffCrack = max(dot(normWorld, lightDir), 0.0);
        vec3 diffuseCrack = diffCrack * uLightColor;
        
        vec3 reflectDir = reflect(-lightDir, normWorld);
        float spec = pow(max(dot(viewDirWorld, reflectDir), 0.0), 16.0);
        vec3 specularCrack = 0.5 * spec * uLightColor;
        
        float ao = h; 
        vec3 crackColor = vec3(0.05, 0.02, 0.02);
        
        vec3 litCrack = (ambient + diffuseCrack + specularCrack) * crackColor * ao;
        
        float crackAlpha = (1.0 - smoothstep(0.95, 1.0, h)) * mask;
        
        finalColor = mix(finalColor, litCrack, crackAlpha);
    }
    
    FragColor = vec4(finalColor, 1.0);
}
)";

// ========== main() ==========
int main() {
    initDefaultCrackParams();
    configureInput();

    if (!glfwInit()) return -1;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    std::string title = "Crack Demo: " + std::string(gCurrentShape == SHAPE_CUBE ? "Cube" : "Sphere") + " (FIXED FLOW + POM)";
    GLFWwindow* win = glfwCreateWindow(1280, 720, title.c_str(), nullptr, nullptr);
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
    GLint uHeightScale = glGetUniformLocation(prog, "uHeightScale");
    GLint uCamPos = glGetUniformLocation(prog, "uCamPos");
    GLint uLightPos = glGetUniformLocation(prog, "uLightPos");
    GLint uLightColor = glGetUniformLocation(prog, "uLightColor");

    GLint uCrackCountLoc = glGetUniformLocation(prog, "uCrackCount");
    GLint uCrackRadiusLoc = glGetUniformLocation(prog, "uCrackRadius");
    GLint uCrackSeedLoc = glGetUniformLocation(prog, "uCrackSeed");

    GLint uCrackCenterLoc = glGetUniformLocation(prog, "uCrackCenter");
    GLint uCrackNormalLoc = glGetUniformLocation(prog, "uCrackNormal");
    GLint uCrackULoc = glGetUniformLocation(prog, "uCrackU");
    GLint uCrackVLoc = glGetUniformLocation(prog, "uCrackV");

    Mesh mesh;
    if (gCurrentShape == SHAPE_CUBE) {
        mesh = makeColoredCube();
    }
    else {
        mesh = makeSphere(64, 64);
    }

    glEnable(GL_DEPTH_TEST);

    glm::vec2 stressDir = glm::normalize(glm::vec2(1.0f, 0.3f));

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

        glUniform1f(uScale, gCrackParams.seedDensity);
        glUniform1f(uJitter, gCrackParams.jitter);
        glUniform1f(uCrackWidth, gCrackParams.crackWidth);
        glUniform2f(uStressDir, stressDir.x, stressDir.y);
        glUniform1f(uAniso, gCrackParams.aniso);
        glUniform1f(uHeightScale, gCrackParams.heightScale);
        glUniform3f(uCamPos, camPos.x, camPos.y, camPos.z);
        glUniform3f(uLightPos, lightPos.x, lightPos.y, lightPos.z);
        glUniform3f(uLightColor, lightColor.x, lightColor.y, lightColor.z);

        glUniform1i(uCrackCountLoc, gCrackCount);
        if (gCrackCount > 0) {
            glUniform1fv(uCrackRadiusLoc, gCrackCount, gCrackRadius);
            glUniform1fv(uCrackSeedLoc, gCrackCount, gCrackSeed);

            glUniform3fv(uCrackCenterLoc, gCrackCount, &gCrackCenter[0].x);
            glUniform3fv(uCrackNormalLoc, gCrackCount, &gCrackNormal[0].x);
            glUniform3fv(uCrackULoc, gCrackCount, &gCrackU[0].x);
            glUniform3fv(uCrackVLoc, gCrackCount, &gCrackV[0].x);
        }

        glBindVertexArray(mesh.vao);
        glDrawElements(GL_TRIANGLES, mesh.indexCount, GL_UNSIGNED_INT, 0);

        if (gCrackTimingPending) {
            glFinish();
            auto end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(end - gCrackStartTime).count();
            std::printf("Crack render time: %.3f ms (total cracks: %d)\n", ms, gCrackCount);
            gCrackTimingPending = false;
        }

        glfwSwapBuffers(win);
        glfwPollEvents();
    }

    destroyMesh(mesh);
    glfwTerminate();
    return 0;
}