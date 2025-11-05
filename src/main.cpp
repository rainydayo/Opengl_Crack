#include <stdexcept>
#include <cstdio>
#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "mesh.hpp"
#include "cube.hpp"

// ========== กล้องแบบ Free (WASD/QE + เมาส์ + สกอลล์) ==========
glm::vec3 camPos = { 0.0f, 0.0f, 4.0f };
glm::vec3 camFront = { 0.0f, 0.0f,-1.0f };
glm::vec3 camUp = { 0.0f, 1.0f, 0.0f };
float yaw = -90.f, pitch = 0.f, fov = 60.f;
float deltaTime = 0.f, lastFrame = 0.f;
bool firstMouse = true; double lastX = 0, lastY = 0;

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
    float xoff = float(x - lastX) * S, yoff = float(lastY - y) * S; lastX = x; lastY = y;
    yaw += xoff; pitch += yoff; pitch = glm::clamp(pitch, -89.f, 89.f);
    glm::vec3 d; d.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    d.y = sin(glm::radians(pitch)); d.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    camFront = glm::normalize(d);
}
void scrollCallback(GLFWwindow*, double, double yoff) {
    fov -= float(yoff); fov = glm::clamp(fov, 1.f, 90.f);
}

// ========== Shader: VS ทำ box-UV ต่อหน้า / FS วาดรอยแตก ==========
static const char* VS = R"(
#version 450 core
layout(location=0) in vec3 aPos;
out vec3 vPos;     // object-space position
out vec2 vUV;      // box-mapped UV per face

// สร้าง UV ต่อหน้าด้วยตำแหน่งเวอร์เท็กซ์ (ไม่ต้องมี UV ในเมช)
vec2 faceUV(vec3 p){
    vec3 ap = abs(p);
    if(ap.x >= ap.y && ap.x >= ap.z){           // ±X face: map (z,y)
        return vec2((p.z+1.0)*0.5, (p.y+1.0)*0.5);
    }else if(ap.y >= ap.x && ap.y >= ap.z){     // ±Y face: map (x,z)
        return vec2((p.x+1.0)*0.5, (p.z+1.0)*0.5);
    }else{                                      // ±Z face: map (x,y)
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

// ---------- Worley/Cellular noise (F1, F2) ----------
vec2 worleyF(vec2 p){
    vec2 ip=floor(p), fp=fract(p);
    float F1=1e9, F2=1e9;
    for(int j=-1;j<=1;j++){
        for(int i=-1;i<=1;i++){
            vec2 cell = ip + vec2(i,j);
            vec2 r2 = rand2_2(cell) - 0.5; // feature point jitter within cell
            vec2 d = (vec2(i,j) + r2) - fp;
            float dist = dot(d,d); // squared dist
            if(dist < F1){ F2=F1; F1=dist; }
            else if(dist < F2){ F2=dist; }
        }
    }
    return vec2(sqrt(F1), sqrt(F2)); // return euclidean distances
}

// ---------- stress-biased warp ----------
mat2 rot(float a){ float c=cos(a), s=sin(a); return mat2(c,-s,s,c); }

uniform float uScale;        // ความถี่ของลายแตก (เช่น 6.0)
uniform float uJitter;       // ความแรง jitter (0..1)
uniform float uCrackWidth;   // ความกว้างเส้นแตก (เช่น 0.05)
uniform vec2  uStressDir;    // ทิศแรง (normalize มาก็ดี)
uniform float uAniso;        // 0..1 ขนาด anisotropy (บีบตามทิศแรง)
uniform float uTime;

// ทำพิกัดใช้งาน: scale + warp จาก stress
vec2 stressWarp(vec2 uv){
    // anisotropy: scale แกนตั้งฉากกับทิศแรง (บีบลายให้แตกตามทิศ)
    float ang = atan(uStressDir.y, uStressDir.x);
    mat2 R = rot(ang), Rinv = rot(-ang);
    vec2 q = (R * (uv * uScale));
    q.y *= (1.0 - 0.6*uAniso);      // บีบแกนขวางทิศแรง
    q = Rinv * q;

    // jitter เล็กน้อยด้วย FBM
    float j = uJitter * (fbm(uv*uScale*1.3 + 7.7) - 0.5);
    return q + j;
}

void main(){
    // พิกัดทำลายแตก
    vec2 uv = vUV;
    vec2 p  = stressWarp(uv);

    // Worley distances
    vec2 F = worleyF(p);
    float edge = F.y - F.x;                // F2 - F1 → ขอบเซลล์
    // ทำเป็นเส้นแตกผ่าน smoothstep รอบ threshold
    float w = uCrackWidth;
    float crack = 1.0 - smoothstep(0.0, w, edge);     // 1=ดำแตก, 0=พื้น

    // เพิ่มรอยแตกย่อยด้วย threshold ยิบย่อย (optional)
    float micro = 1.0 - smoothstep(0.0, w*0.5, abs(edge-0.02));

    // สีพื้น + แต้มแตก
    vec3 base = vec3(0.72, 0.69, 0.65);               // สีคอนกรีตอ่อน
    vec3 c = mix(base, vec3(0.05), clamp(crack + 0.5*micro, 0.0, 1.0));

    // ไลต์แบบ lambert ง่ายๆ ให้พอมีมิติ
    vec3 L = normalize(vec3(0.5, 0.8, 0.6));
    // ปกติคร่าวๆ จากกราดสีพื้นที่ (fake normal จากเกรเดียนต์ edge)
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

// ========== Utils ==========
static void die(const char* m) { throw std::runtime_error(m); }
static GLuint makeProg(const char* vs, const char* fs) {
    auto comp = [&](GLenum t, const char* s) {
        GLuint sh = glCreateShader(t); glShaderSource(sh, 1, &s, nullptr); glCompileShader(sh);
        GLint ok; glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
        if (!ok) { char log[2048]; glGetShaderInfoLog(sh, 2048, nullptr, log); std::fprintf(stderr, "%s\n", log); die("compile"); }
        return sh;
        };
    GLuint v = comp(GL_VERTEX_SHADER, vs), f = comp(GL_FRAGMENT_SHADER, fs), p = glCreateProgram();
    glAttachShader(p, v); glAttachShader(p, f); glLinkProgram(p);
    GLint ok; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { char log[2048]; glGetProgramInfoLog(p, 2048, nullptr, log); std::fprintf(stderr, "%s\n", log); die("link"); }
    glDeleteShader(v); glDeleteShader(f); return p;
}

int main() {
    if (!glfwInit()) return -1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* win = glfwCreateWindow(1280, 720, "Procedural Cracks on Cube", nullptr, nullptr);
    if (!win) return -1;
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);

    if (!gladLoadGL(glfwGetProcAddress)) { std::fprintf(stderr, "GLAD2 load failed\n"); return -1; }

    // จับเมาส์ (เอาออกได้ถ้าไม่ต้องการ)
    glfwSetInputMode(win, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetCursorPosCallback(win, cursorPosCallback);
    glfwSetScrollCallback(win, scrollCallback);

    GLuint prog = makeProg(VS, FS);
    GLint  uMVP = glGetUniformLocation(prog, "uMVP");
    GLint  uScale = glGetUniformLocation(prog, "uScale");
    GLint  uJitter = glGetUniformLocation(prog, "uJitter");
    GLint  uCrackWidth = glGetUniformLocation(prog, "uCrackWidth");
    GLint  uStressDir = glGetUniformLocation(prog, "uStressDir");
    GLint  uAniso = glGetUniformLocation(prog, "uAniso");
    GLint  uTime = glGetUniformLocation(prog, "uTime");

    Mesh cube = makeColoredCube();
    glEnable(GL_DEPTH_TEST);

    // ค่าพื้นฐาน (จูนได้ตามวัสดุ/ซีน)
    float scale = 6.0f;   // ความถี่ลายแตก
    float jitter = 0.25f;  // ความฟุ้งตามธรรมชาติ
    float crackWidth = 0.045f; // ความกว้างเส้นแตก
    glm::vec2 stressDir = glm::normalize(glm::vec2(1.0f, 0.3f));
    float aniso = 0.6f;   // 0..1 (บีบตามทิศแรงมากน้อย)

    while (!glfwWindowShouldClose(win)) {
        float now = float(glfwGetTime());
        deltaTime = now - lastFrame; lastFrame = now;
        processInput(win);

        int w, h; glfwGetFramebufferSize(win, &w, &h);
        glViewport(0, 0, w, h);
        glClearColor(0.08f, 0.09f, 0.12f, 1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glm::mat4 P = glm::perspective(glm::radians(fov), w > 0 ? (float)w / h : 1.f, 0.1f, 100.f);
        glm::mat4 Vw = glm::lookAt(camPos, camPos + camFront, camUp);
        glm::mat4 M = glm::mat4(1.0f);
        glm::mat4 MVP = P * Vw * M;

        glUseProgram(prog);
        glUniformMatrix4fv(uMVP, 1, GL_FALSE, &MVP[0][0]);
        glUniform1f(uScale, scale);
        glUniform1f(uJitter, jitter);
        glUniform1f(uCrackWidth, crackWidth);
        glUniform2f(uStressDir, stressDir.x, stressDir.y);
        glUniform1f(uAniso, aniso);
        glUniform1f(uTime, now);

        glBindVertexArray(cube.vao);
        glDrawElements(GL_TRIANGLES, cube.indexCount, GL_UNSIGNED_INT, 0);

        glfwSwapBuffers(win);
        glfwPollEvents();
    }

    destroyMesh(cube);
    glfwTerminate();
    return 0;
}