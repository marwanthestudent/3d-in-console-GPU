// main.cu - CUDA parallel 3D cube renderer (university GPU project)
//
// Based on the rendering math in main.cpp (Mimocake/3d-in-console),
// generalized from a wireframe cube to a solid N x N x N point cloud
// so we have real data-parallel work for the GPU.
//
// Build:   nvcc -O3 main.cu -o cube
// Run:     ./cube           (defaults: N=100, iters=100)
//          ./cube 150 200   (N=150, 200 benchmark iterations)
//
// ---------------------------------------------------------------
// Thread-to-data mapping
//   * One CUDA thread per vertex.
//   * gid = blockIdx.x * blockDim.x + threadIdx.x indexes the SoA
//     arrays d_x / d_y / d_z directly. Total threads launched
//     = N*N*N rounded up to a multiple of blockDim.x (256).
//
// Memory coalescing strategy
//   * Vertex positions are stored as Struct of Arrays (SoA): three
//     separate float arrays d_x, d_y, d_z. Threads in one warp have
//     consecutive gids, so d_x[gid] / d_y[gid] / d_z[gid] become
//     consecutive 4-byte loads -> the hardware coalesces 32 of them
//     into a single 128-byte transaction. With AoS (struct {x,y,z})
//     each warp load would be strided by 12 bytes and waste bandwidth.
//
// GPU workload distribution
//   * Stage 1 (transform_gpu): N*N*N independent vertex threads,
//     each does rotation + projection then scatters into the packed
//     z-buffer with one atomicMin. No inter-thread sync.
//   * Stage 2 (zbuf_to_screen): WIDTH*HEIGHT = 3600 threads, one
//     per pixel, decode the packed value to an ASCII char.
//   * Rotation/projection matrices live in __constant__ memory:
//     every thread reads the same address, which the constant cache
//     broadcasts in a single cycle.
// ---------------------------------------------------------------

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <thread>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do {                                              \
    cudaError_t _e = (call);                                               \
    if (_e != cudaSuccess) {                                               \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                        \
                cudaGetErrorString(_e), __FILE__, __LINE__);               \
        std::exit(1);                                                      \
    }                                                                      \
} while (0)

// Console screen size (matches the original project)
static const int   WIDTH  = 120;
static const int   HEIGHT = 30;
static const int   NPIX   = WIDTH * HEIGHT;

// Projection constants (mirror main.cpp)
static const float PI    = 3.14159265f;
static const float ASP   = (float)WIDTH / (float)HEIGHT;
static const float P_ASP = 11.0f / 24.0f;
static const float FNEAR = 0.1f;
static const float FFAR  = 1000.0f;
static const float FOV   = 90.0f;

// Shading ramp from far (dim) to near (bright)
static const char  SHADES[10] = {'.',',',':',';','+','*','x','%','#','@'};

// -------- Constant memory (cached, broadcast to all threads) --------
__constant__ float c_rot [16];   // row-major 4x4 rotation matrix
__constant__ float c_proj[16];   // row-major 4x4 projection matrix
__constant__ char  c_shade[10];  // shading ramp

// ====================================================================
//                           GPU kernels
// ====================================================================

// Pack (depth, char-index) into a single 64-bit value so we can resolve
// overlapping projections with one atomicMin per fragment.
//
// Layout:  [ 32 bits depth | 32 bits char-index ]
// For positive floats, __float_as_uint preserves numeric order, so the
// unsigned-integer comparison done by atomicMin matches "closer wins".
__device__ __forceinline__
unsigned long long pack_frag(float depth, unsigned int idx)
{
    unsigned int du = __float_as_uint(depth);
    return ((unsigned long long)du << 32) | (unsigned long long)idx;
}

// Stage 1: transform every vertex and scatter into the packed z-buffer.
__global__
void transform_gpu(const float* __restrict__ d_x,
                   const float* __restrict__ d_y,
                   const float* __restrict__ d_z,
                   int n_verts,
                   unsigned long long* d_zbuf)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_verts) return;

    // Coalesced loads (SoA, see header comment).
    float x = d_x[gid];
    float y = d_y[gid];
    float z = d_z[gid];
    float w = 1.0f;

    // Rotate (constant-memory broadcast: same address across the warp).
    float rx = x*c_rot[0] + y*c_rot[4] + z*c_rot[8]  + w*c_rot[12];
    float ry = x*c_rot[1] + y*c_rot[5] + z*c_rot[9]  + w*c_rot[13];
    float rz = x*c_rot[2] + y*c_rot[6] + z*c_rot[10] + w*c_rot[14];
    float rw = x*c_rot[3] + y*c_rot[7] + z*c_rot[11] + w*c_rot[15];

    // Translate the cube away from the camera so it is in front of us.
    rz += 3.0f;

    // Project.
    float px = rx*c_proj[0] + ry*c_proj[4] + rz*c_proj[8]  + rw*c_proj[12];
    float py = rx*c_proj[1] + ry*c_proj[5] + rz*c_proj[9]  + rw*c_proj[13];
    float pw = rx*c_proj[3] + ry*c_proj[7] + rz*c_proj[11] + rw*c_proj[15];
    if (pw <= 0.0f) return;
    px /= pw;
    py /= pw;

    // NDC -> pixel.
    int sx = (int)((px + 1.0f) * 0.5f * WIDTH);
    int sy = (int)((1.0f - (py + 1.0f) * 0.5f) * HEIGHT);
    if (sx < 0 || sx >= WIDTH || sy < 0 || sy >= HEIGHT) return;

    // Pick a shading char based on rotated depth (closer => brighter).
    // rz is roughly in [3 - sqrt(0.75), 3 + sqrt(0.75)] ~= [2.13, 3.87].
    float t = (3.87f - rz) / (3.87f - 2.13f);   // 0..1, near=1
    int   shade = (int)(t * 9.0f);
    if (shade < 0) shade = 0;
    if (shade > 9) shade = 9;

    // Z-buffer resolve: atomicMin keeps the closest fragment per pixel.
    int pix = sy * WIDTH + sx;
    atomicMin(&d_zbuf[pix], pack_frag(rz, (unsigned int)shade));
}

// Stage 2: turn the packed z-buffer into ASCII characters.
__global__
void zbuf_to_screen(const unsigned long long* d_zbuf, char* d_screen)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= NPIX) return;
    unsigned long long v = d_zbuf[gid];
    if (v == 0xFFFFFFFFFFFFFFFFULL) {
        d_screen[gid] = ' ';                          // pixel never written
    } else {
        unsigned int idx = (unsigned int)(v & 0xFFFFFFFFu);
        if (idx > 9) idx = 9;
        d_screen[gid] = c_shade[idx];
    }
}

// ====================================================================
//                       CPU baseline (sequential)
// ====================================================================
void transform_cpu(const float* x, const float* y, const float* z,
                   int n_verts,
                   const float* rot, const float* proj,
                   char* screen)
{
    static float depth[NPIX];
    for (int i = 0; i < NPIX; i++) { depth[i] = 1e30f; screen[i] = ' '; }

    for (int i = 0; i < n_verts; i++) {
        float vx = x[i], vy = y[i], vz = z[i], vw = 1.0f;

        float rx = vx*rot[0] + vy*rot[4] + vz*rot[8]  + vw*rot[12];
        float ry = vx*rot[1] + vy*rot[5] + vz*rot[9]  + vw*rot[13];
        float rz = vx*rot[2] + vy*rot[6] + vz*rot[10] + vw*rot[14];
        float rw = vx*rot[3] + vy*rot[7] + vz*rot[11] + vw*rot[15];
        rz += 3.0f;

        float px = rx*proj[0] + ry*proj[4] + rz*proj[8]  + rw*proj[12];
        float py = rx*proj[1] + ry*proj[5] + rz*proj[9]  + rw*proj[13];
        float pw = rx*proj[3] + ry*proj[7] + rz*proj[11] + rw*proj[15];
        if (pw <= 0.0f) continue;
        px /= pw; py /= pw;

        int sx = (int)((px + 1.0f) * 0.5f * WIDTH);
        int sy = (int)((1.0f - (py + 1.0f) * 0.5f) * HEIGHT);
        if (sx < 0 || sx >= WIDTH || sy < 0 || sy >= HEIGHT) continue;

        int pix = sy * WIDTH + sx;
        if (rz < depth[pix]) {
            depth[pix] = rz;
            float t = (3.87f - rz) / (3.87f - 2.13f);
            int s = (int)(t * 9.0f);
            if (s < 0) s = 0;
            if (s > 9) s = 9;
            screen[pix] = SHADES[s];
        }
    }
}

// ====================================================================
//                    Matrix builders (host side)
// ====================================================================

static void mat_mul4(const float* A, const float* B, float* C) {
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            float s = 0.0f;
            for (int k = 0; k < 4; k++) s += A[i*4+k] * B[k*4+j];
            C[i*4+j] = s;
        }
}

// Combined rotation = Ry * Rx * Rz (matches the order used in main.cpp).
static void build_rot(float* m, float ax, float ay, float az) {
    float cx = cosf(ax), sx = sinf(ax);
    float cy = cosf(ay), sy = sinf(ay);
    float cz = cosf(az), sz = sinf(az);
    float Ry[16] = {  cy, 0, sy, 0,    0, 1,  0, 0,   -sy, 0, cy, 0,   0, 0, 0, 1 };
    float Rx[16] = {  1,  0, 0,  0,    0, cx, sx,0,    0, -sx,cx, 0,   0, 0, 0, 1 };
    float Rz[16] = {  cz, sz,0,  0,   -sz,cz, 0, 0,    0,  0, 1,  0,   0, 0, 0, 1 };
    float T[16];
    mat_mul4(Ry, Rx, T);
    mat_mul4(T, Rz, m);
}

static void build_proj(float* m) {
    for (int i = 0; i < 16; i++) m[i] = 0.0f;
    float t = tanf(FOV * 0.5f / 180.0f * PI);
    m[0]  = ((1.0f / ASP) / P_ASP) / t;
    m[5]  = 1.0f / t;
    m[10] = FFAR / (FFAR - FNEAR);
    m[11] = 1.0f;
    m[14] = -FFAR * FNEAR / (FFAR - FNEAR);
}

// ====================================================================
//                                main
// ====================================================================
int main(int argc, char** argv)
{
    // Modes:
    //   ./cube                  -> benchmark mode (default N=100, iters=100)
    //   ./cube N iters          -> benchmark mode, custom N / iters
    //   ./cube anim             -> animated rotating cube, Ctrl+C to stop
    //   ./cube anim N           -> animated, custom N
    bool animate = (argc > 1 && std::strcmp(argv[1], "anim") == 0);

    int N, iters;
    if (animate) {
        //I kept saying N = 1 but actually there was a minimum of 60.
        N     = (argc > 2) ? std::atoi(argv[2]) : 60;   // smaller default for anim
        iters = 0;                                       // unused
    } else {
        N     = (argc > 1) ? std::atoi(argv[1]) : 100;
        iters = (argc > 2) ? std::atoi(argv[2]) : 100;
    }
    int n_verts = N * N * N;

    if (animate)
        printf("ANIMATE: N = %d  ->  %d vertices  (Ctrl+C to stop)\n",
               N, n_verts);
    else
        printf("N = %d  ->  %d vertices,  benchmark iters = %d\n",
               N, n_verts, iters);

    // ---- Build the N x N x N point cloud in [-0.5, 0.5]^3 (SoA) ----
    std::vector<float> hx(n_verts), hy(n_verts), hz(n_verts);
    float inv = (N > 1) ? 1.0f / (float)(N - 1) : 0.0f;
    for (int k = 0; k < N; k++)
        for (int j = 0; j < N; j++)
            for (int i = 0; i < N; i++) {
                int idx = (k * N + j) * N + i;
                hx[idx] = (float)i * inv - 0.5f;
                hy[idx] = (float)j * inv - 0.5f;
                hz[idx] = (float)k * inv - 0.5f;
            }

    // ---- Matrices ----
    float h_proj[16], h_rot[16];
    build_proj(h_proj);
    build_rot(h_rot, 0.6f * 0.5f, 0.6f, 0.6f * 0.5f);   // a nice angle

    CUDA_CHECK(cudaMemcpyToSymbol(c_proj,  h_proj, sizeof(h_proj)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_rot,   h_rot,  sizeof(h_rot)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_shade, SHADES, sizeof(SHADES)));

    // ---- Device allocations ----
    float              *d_x = nullptr, *d_y = nullptr, *d_z = nullptr;
    unsigned long long *d_zbuf = nullptr;
    char               *d_screen = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x,      sizeof(float) * n_verts));
    CUDA_CHECK(cudaMalloc(&d_y,      sizeof(float) * n_verts));
    CUDA_CHECK(cudaMalloc(&d_z,      sizeof(float) * n_verts));
    CUDA_CHECK(cudaMalloc(&d_zbuf,   sizeof(unsigned long long) * NPIX));
    CUDA_CHECK(cudaMalloc(&d_screen, sizeof(char) * NPIX));

    // ---- CUDA events for isolated profiling ----
    cudaEvent_t e0, e1, e2, e3;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventCreate(&e2); cudaEventCreate(&e3);

    const int block = 256;
    const int grid_v = (n_verts + block - 1) / block;
    const int grid_p = (NPIX    + block - 1) / block;

    // =========== Animation mode: rotate forever, print each frame ===========
    if (animate) {
        // Vertex data is static -- copy it once.
        CUDA_CHECK(cudaMemcpy(d_x, hx.data(), sizeof(float) * n_verts,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_y, hy.data(), sizeof(float) * n_verts,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_z, hz.data(), sizeof(float) * n_verts,
                              cudaMemcpyHostToDevice));

        // Clear screen once at startup (ANSI: erase + cursor home).
        std::printf("\033[2J");

        char frame[NPIX + 1];
        frame[NPIX] = '\0';
        float angle = 0.0f;

        for (;;) {
            // Update rotation matrix each frame and push to constant memory.
            angle += 0.03f;
            build_rot(h_rot, angle * 0.5f, angle, angle * 0.5f);
            CUDA_CHECK(cudaMemcpyToSymbol(c_rot, h_rot, sizeof(h_rot)));

            // Render: clear z-buffer, transform vertices, decode to chars.
            CUDA_CHECK(cudaMemset(d_zbuf, 0xFF,
                                  sizeof(unsigned long long) * NPIX));
            transform_gpu  <<<grid_v, block>>>(d_x, d_y, d_z, n_verts, d_zbuf);
            zbuf_to_screen <<<grid_p, block>>>(d_zbuf, d_screen);
            CUDA_CHECK(cudaMemcpy(frame, d_screen, NPIX,
                                  cudaMemcpyDeviceToHost));

            // Move cursor home and reprint the frame in place.
            std::printf("\033[H");
            for (int row = 0; row < HEIGHT; row++) {
                std::fwrite(frame + row * WIDTH, 1, WIDTH, stdout);
                std::putchar('\n');
            }
            std::fflush(stdout);

            // ~30 FPS cap so the terminal can keep up.
            std::this_thread::sleep_for(std::chrono::milliseconds(33));
        }
        // unreachable
    }

    // =========== One timed frame (this frame is also printed) ===========

    // H2D
    cudaEventRecord(e0);
    CUDA_CHECK(cudaMemcpy(d_x, hx.data(), sizeof(float) * n_verts,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, hy.data(), sizeof(float) * n_verts,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z, hz.data(), sizeof(float) * n_verts,
                          cudaMemcpyHostToDevice));
    cudaEventRecord(e1);

    // Kernels: clear z-buffer to 0xFF...FF (max ULL) so atomicMin always wins
    // on the first scatter, then transform + resolve, then decode to chars.
    CUDA_CHECK(cudaMemset(d_zbuf, 0xFF,
                          sizeof(unsigned long long) * NPIX));
    transform_gpu  <<<grid_v, block>>>(d_x, d_y, d_z, n_verts, d_zbuf);
    zbuf_to_screen <<<grid_p, block>>>(d_zbuf, d_screen);
    cudaEventRecord(e2);

    // D2H
    char screen[NPIX + 1];
    CUDA_CHECK(cudaMemcpy(screen, d_screen, NPIX, cudaMemcpyDeviceToHost));
    screen[NPIX] = '\0';
    cudaEventRecord(e3);
    cudaEventSynchronize(e3);

    float t_h2d = 0.0f, t_kern = 0.0f, t_d2h = 0.0f;
    cudaEventElapsedTime(&t_h2d,  e0, e1);
    cudaEventElapsedTime(&t_kern, e1, e2);
    cudaEventElapsedTime(&t_d2h,  e2, e3);

    // ---- Print first (and only) frame ----
    for (int row = 0; row < HEIGHT; row++) {
        std::fwrite(screen + row * WIDTH, 1, WIDTH, stdout);
        std::putchar('\n');
    }

    printf("\n--- Per-stage GPU timing (one frame) ---\n");
    printf("H2D copy   : %8.3f ms\n", t_h2d);
    printf("Kernels    : %8.3f ms\n", t_kern);
    printf("D2H copy   : %8.3f ms\n", t_d2h);
    printf("GPU total  : %8.3f ms\n", t_h2d + t_kern + t_d2h);

    // =========== Benchmark loops (output disabled) ===========

    // GPU benchmark: kernel-only time, repeated `iters` times.
    cudaEventRecord(e0);
    for (int it = 0; it < iters; it++) {
        cudaMemset(d_zbuf, 0xFF, sizeof(unsigned long long) * NPIX);
        transform_gpu  <<<grid_v, block>>>(d_x, d_y, d_z, n_verts, d_zbuf);
        zbuf_to_screen <<<grid_p, block>>>(d_zbuf, d_screen);
    }
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float t_gpu_bench = 0.0f;
    cudaEventElapsedTime(&t_gpu_bench, e0, e1);

    // CPU benchmark.
    char cpu_screen[NPIX];
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iters; it++) {
        transform_cpu(hx.data(), hy.data(), hz.data(),
                      n_verts, h_rot, h_proj, cpu_screen);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double t_cpu_bench =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    printf("\n--- Benchmark (%d iters, no terminal output) ---\n", iters);
    printf("CPU total  : %8.3f ms   (%6.3f ms / frame)\n",
           t_cpu_bench, t_cpu_bench / iters);
    printf("GPU kernel : %8.3f ms   (%6.3f ms / frame)\n",
           t_gpu_bench, t_gpu_bench / iters);
    printf("Speedup    : %.2fx\n", t_cpu_bench / t_gpu_bench);

    // ---- Cleanup ----
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cudaEventDestroy(e2); cudaEventDestroy(e3);
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
    cudaFree(d_zbuf); cudaFree(d_screen);
    return 0;
}
