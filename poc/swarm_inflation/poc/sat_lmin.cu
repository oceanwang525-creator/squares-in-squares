// Shared-memory spatial-hash CUDA SAT inflation + elastic-(x0,y0,L) compression +
// parallel void-jump / angle-snap / cluster-shift moves + float64 final polish.
// Target: RTX 4070 SUPER (Ada Lovelace sm_89), N <= 1024.
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_N 1024
#define BLOCK_THREADS 256
#define GRID_G 22
#define NUM_CELLS (GRID_G * GRID_G)

struct SolveParams {
    int32_t B;
    int32_t N;
    int32_t use_elastic_L;
    int32_t fast_mode;
    uint32_t seed;
};

__device__ __forceinline__ uint32_t xorshift32(uint32_t& state) {
    uint32_t x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    state = x;
    return x;
}

__device__ __forceinline__ float rand_u01(uint32_t& state) {
    return (xorshift32(state) & 0x00FFFFFFu) * (1.0f / 16777216.0f);
}

__device__ __forceinline__ float rand_normal(uint32_t& state) {
    float u1 = fmaxf(rand_u01(state), 1e-7f);
    float u2 = rand_u01(state);
    return sqrtf(-2.0f * logf(u1)) * cosf(6.28318530718f * u2);
}

__device__ __forceinline__ float pair_pen_f32(
    float xi, float yi, float ci, float si,
    float xj, float yj, float cj, float sj,
    float half_a
) {
    float dx = xj - xi;
    float dy = yj - yi;
    if (dx * dx + dy * dy >= 4.25f * half_a * half_a * 4.0f) return 0.0f;
    float cr = ci * cj + si * sj;
    float sr = si * cj - ci * sj;
    float rad = half_a * (1.0f + fabsf(cr) + fabsf(sr));
    float p0 = fabsf(dx * ci + dy * si);
    float p1 = fabsf(-dx * si + dy * ci);
    float p2 = fabsf(dx * cj + dy * sj);
    float p3 = fabsf(-dx * sj + dy * cj);
    float vmax = fmaxf(fmaxf(p0, p1), fmaxf(p2, p3));
    return fmaxf(0.0f, rad - vmax);
}

__device__ __forceinline__ void pair_pen_and_grad_f32(
    float xi, float yi, float ci, float si,
    float xj, float yj, float cj, float sj,
    int i, int j,
    float half_a, float sg_eps,
    float& pen_out, float& gx_out, float& gy_out, float& gth_out
) {
    float dx = xj - xi;
    float dy = yj - yi;
    if (dx * dx + dy * dy >= 8.5f * half_a * half_a * 2.0f) {
        pen_out = 0.0f; gx_out = 0.0f; gy_out = 0.0f; gth_out = 0.0f;
        return;
    }
    float cr = ci * cj + si * sj;
    float sr = si * cj - ci * sj;
    float abs_cr = fabsf(cr);
    float abs_sr = fabsf(sr);
    float rad = half_a * (1.0f + abs_cr + abs_sr);

    float eps2 = sg_eps * sg_eps;
    float s_cr = cr * rsqrtf(cr * cr + eps2);
    float s_sr = sr * rsqrtf(sr * sr + eps2);
    float drad = half_a * (s_cr * (-sr) + s_sr * cr);

    float p0 = dx * ci + dy * si;
    float p1 = -dx * si + dy * ci;
    float p2 = dx * cj + dy * sj;
    float p3 = -dx * sj + dy * cj;

    float z0 = fabsf(p0) - rad;
    float z1 = fabsf(p1) - rad;
    float z2 = fabsf(p2) - rad;
    float z3 = fabsf(p3) - rad;

    int k_star = 0;
    float z_max = z0;
    float p_star = p0;
    float ux = ci, uy = si;
    float is_i = 1.0f;

    if (z1 > z_max) { z_max = z1; p_star = p1; ux = -si; uy = ci; is_i = 1.0f; }
    if (z2 > z_max) { z_max = z2; p_star = p2; ux = cj;  uy = sj; is_i = 0.0f; }
    if (z3 > z_max) { z_max = z3; p_star = p3; ux = -sj; uy = cj; is_i = 0.0f; }

    float pen = fmaxf(0.0f, -z_max);
    pen_out = pen;
    if (pen <= 0.0f) {
        gx_out = 0.0f; gy_out = 0.0f; gth_out = 0.0f;
        return;
    }

    float sp = (p_star > 0.0f) ? 1.0f : ((p_star < 0.0f) ? -1.0f : ((j > i) ? 1.0f : -1.0f));
    float dpen_dx = sp * ux;
    float dpen_dy = sp * uy;
    float dpen_dth = drad - sp * (dy * ux - dx * uy) * is_i;
    float w = 2.0f * pen;
    gx_out = w * dpen_dx;
    gy_out = w * dpen_dy;
    gth_out = w * dpen_dth;
}

__device__ float block_reduce_sum(float val, float* buf) {
    int tid = threadIdx.x;
    buf[tid] = val;
    __syncthreads();
    for (int s = BLOCK_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) buf[tid] += buf[tid + s];
        __syncthreads();
    }
    float res = buf[0];
    __syncthreads();
    return res;
}

__device__ double block_reduce_sum_f64(double val, double* buf) {
    int tid = threadIdx.x;
    buf[tid] = val;
    __syncthreads();
    for (int s = BLOCK_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) buf[tid] += buf[tid + s];
        __syncthreads();
    }
    double res = buf[0];
    __syncthreads();
    return res;
}

__device__ double block_reduce_min_f64(double val, double* buf) {
    int tid = threadIdx.x;
    buf[tid] = val;
    __syncthreads();
    for (int s = BLOCK_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) buf[tid] = fmin(buf[tid], buf[tid + s]);
        __syncthreads();
    }
    double res = buf[0];
    __syncthreads();
    return res;
}

__device__ double block_reduce_max_f64(double val, double* buf) {
    int tid = threadIdx.x;
    buf[tid] = val;
    __syncthreads();
    for (int s = BLOCK_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) buf[tid] = fmax(buf[tid], buf[tid + s]);
        __syncthreads();
    }
    double res = buf[0];
    __syncthreads();
    return res;
}

__device__ void rebuild_grid(
    int N, float x0, float y0, float L,
    const float* sx, const float* sy,
    int32_t* cell_head, int16_t* cell_next,
    float& gx0, float& gy0, float& cell_w
) {
    int tid = threadIdx.x;
    for (int c = tid; c < NUM_CELLS; c += BLOCK_THREADS) {
        cell_head[c] = -1;
    }
    if (tid == 0) {
        gx0 = x0 - 1.0f;
        gy0 = y0 - 1.0f;
        cell_w = fmaxf(1.48f, (L + 2.0f) / (float)GRID_G);
    }
    __syncthreads();
    float inv_cw = 1.0f / cell_w;

    for (int i = tid; i < N; i += BLOCK_THREADS) {
        int cx = max(0, min(GRID_G - 1, (int)floorf((sx[i] - gx0) * inv_cw)));
        int cy = max(0, min(GRID_G - 1, (int)floorf((sy[i] - gy0) * inv_cw)));
        int cid = cy * GRID_G + cx;
        int prev = atomicExch(&cell_head[cid], i);
        cell_next[i] = (int16_t)prev;
    }
    __syncthreads();
}

__device__ float eval_single_sq_f32(
    int i, float px, float py, float ang,
    int N, float x0, float y0, float L,
    float gx0, float gy0, float cell_w,
    float scale_a, float w_wall,
    const float* sx, const float* sy, const float* sc, const float* ss,
    const int32_t* cell_head, const int16_t* cell_next
) {
    float c = cosf(ang), s = sinf(ang);
    float half_a = 0.5f * scale_a;
    float ext = half_a * (fabsf(c) + fabsf(s));
    float vL = fmaxf(0.0f, x0 + ext - px);
    float vR = fmaxf(0.0f, px + ext - x0 - L);
    float vB = fmaxf(0.0f, y0 + ext - py);
    float vT = fmaxf(0.0f, py + ext - y0 - L);
    float ew = w_wall * (vL * vL + vR * vR + vB * vB + vT * vT);

    float inv_cw = 1.0f / cell_w;
    int cx = max(0, min(GRID_G - 1, (int)floorf((px - gx0) * inv_cw)));
    int cy = max(0, min(GRID_G - 1, (int)floorf((py - gy0) * inv_cw)));
    float ep = 0.0f;
    for (int dy = -1; dy <= 1; ++dy) {
        int ny = cy + dy;
        if (ny < 0 || ny >= GRID_G) continue;
        for (int dx = -1; dx <= 1; ++dx) {
            int nx = cx + dx;
            if (nx < 0 || nx >= GRID_G) continue;
            int j = cell_head[ny * GRID_G + nx];
            while (j >= 0) {
                if (j != i) {
                    float pen = pair_pen_f32(px, py, c, s, sx[j], sy[j], sc[j], ss[j], half_a);
                    ep += 0.5f * pen * pen;
                }
                j = (int)cell_next[j];
            }
        }
    }
    return ew + ep;
}

__device__ float compute_total_energy_f32(
    int N, float x0, float y0, float L,
    float gx0, float gy0, float cell_w,
    float scale_a, float w_wall,
    const float* sx, const float* sy, const float* sc, const float* ss,
    float* sqe, const int32_t* cell_head, const int16_t* cell_next,
    float* red_buf
) {
    int tid = threadIdx.x;
    float local_sum = 0.0f;
    float half_a = 0.5f * scale_a;
    float inv_cw = 1.0f / cell_w;

    for (int i = tid; i < N; i += BLOCK_THREADS) {
        float px = sx[i], py = sy[i], ci = sc[i], si = ss[i];
        float ext = half_a * (fabsf(ci) + fabsf(si));
        float vL = fmaxf(0.0f, x0 + ext - px);
        float vR = fmaxf(0.0f, px + ext - x0 - L);
        float vB = fmaxf(0.0f, y0 + ext - py);
        float vT = fmaxf(0.0f, py + ext - y0 - L);
        float ew = w_wall * (vL * vL + vR * vR + vB * vB + vT * vT);

        int cx = max(0, min(GRID_G - 1, (int)floorf((px - gx0) * inv_cw)));
        int cy = max(0, min(GRID_G - 1, (int)floorf((py - gy0) * inv_cw)));
        float ep = 0.0f;
        for (int dy = -1; dy <= 1; ++dy) {
            int ny = cy + dy;
            if (ny < 0 || ny >= GRID_G) continue;
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = cx + dx;
                if (nx < 0 || nx >= GRID_G) continue;
                int j = cell_head[ny * GRID_G + nx];
                while (j >= 0) {
                    if (j != i) {
                        float pen = pair_pen_f32(px, py, ci, si, sx[j], sy[j], sc[j], ss[j], half_a);
                        ep += 0.5f * pen * pen;
                    }
                    j = (int)cell_next[j];
                }
            }
        }
        float ei = ew + ep;
        sqe[i] = ei;
        local_sum += ei;
    }
    return block_reduce_sum(local_sum, red_buf);
}

__global__ void solve_batch_fp32(
    int B, int N, int use_elastic_L, int fast_mode, uint32_t base_seed,
    const float* __restrict__ in_x,
    const float* __restrict__ in_y,
    const float* __restrict__ in_th,
    const float* __restrict__ in_L,
    const uint8_t* __restrict__ in_mob,
    float* __restrict__ out_x,
    float* __restrict__ out_y,
    float* __restrict__ out_th,
    float* __restrict__ out_x0,
    float* __restrict__ out_y0,
    float* __restrict__ out_L
) {
    int b = blockIdx.x;
    if (b >= B) return;
    int tid = threadIdx.x;

    __shared__ float sx[MAX_N];
    __shared__ float sy[MAX_N];
    __shared__ float sth[MAX_N];
    __shared__ float sc[MAX_N];
    __shared__ float ss[MAX_N];
    __shared__ float sqe[MAX_N];
    __shared__ uint8_t smob[MAX_N];
    __shared__ int32_t cell_head[NUM_CELLS];
    __shared__ int16_t cell_next[MAX_N];
    __shared__ float red_buf[BLOCK_THREADS];
    __shared__ float red_b2[BLOCK_THREADS];
    __shared__ int32_t ired_buf[BLOCK_THREADS];
    __shared__ float sh_L, sh_L_init, sh_x0, sh_y0;
    __shared__ float sh_gx0, sh_gy0, sh_cw;
    __shared__ float sh_step_sc;

    uint32_t rng = base_seed ^ (uint32_t)(b * 747796405u + tid * 2891336453u + 1337u);
    if (rng == 0u) rng = 1u;

    for (int i = tid; i < N; i += BLOCK_THREADS) {
        float px = in_x[b * N + i];
        float py = in_y[b * N + i];
        float pt = in_th[b * N + i];
        sx[i] = px;
        sy[i] = py;
        sth[i] = pt;
        sc[i] = cosf(pt);
        ss[i] = sinf(pt);
        sqe[i] = 0.0f;
        smob[i] = in_mob ? in_mob[b * N + i] : 1;
    }
    if (tid == 0) {
        sh_L = in_L[b];
        sh_L_init = in_L[b];
        sh_x0 = 0.0f;
        sh_y0 = 0.0f;
    }
    __syncthreads();

    // Thread-local Adam registers for up to 4 owned squares
    float mx[4] = {0,0,0,0}, vx[4] = {0,0,0,0};
    float my[4] = {0,0,0,0}, vy[4] = {0,0,0,0};
    float mt[4] = {0,0,0,0}, vt[4] = {0,0,0,0};

    // Stage schedule: (scale_a, n_steps, lr, w_wall_or_pen, w_L, allow_L_opt)
    struct Stage {
        float a;
        int steps;
        float lr;
        float w_pen;
        float w_L;
        int opt_L;
    };

    const Stage stages_fast[14] = {
        {0.880f, 350, 0.0160f, 5.0f,    0.0f, 0},
        {0.910f, 350, 0.0130f, 5.0f,    0.0f, 0},
        {0.940f, 400, 0.0100f, 5.0f,    0.0f, 0},
        {0.962f, 450, 0.0075f, 5.0f,    0.0f, 0},
        // Deflate pulse 1 + wall-balance L control
        {0.950f, 350, 0.0080f, 40.0f,   1.0f, 1},
        {0.975f, 550, 0.0050f, 80.0f,   1.0f, 1},
        {0.988f, 600, 0.0032f, 250.0f,  1.0f, 1},
        // Deflate pulse 2
        {0.972f, 400, 0.0040f, 400.0f,  1.0f, 1},
        {0.992f, 650, 0.0022f, 1200.0f, 1.0f, 1},
        {0.997f, 650, 0.0012f, 3500.0f, 1.0f, 1},
        {0.999f, 650, 0.0006f, 10000.0f,1.0f, 1},
        {1.000f, 750, 0.0003f, 25000.0f,1.0f, 1},
        {1.000f, 650, 0.00014f,60000.0f,1.0f, 1},
        {1.000f, 550, 0.00006f,120000.0f,1.0f,1}
    };
    const Stage stages_gentle[10] = {
        {0.945f, 350, 0.0055f, 15.0f,   1.0f, 1},
        {0.962f, 400, 0.0042f, 40.0f,   1.0f, 1},
        {0.976f, 450, 0.0030f, 120.0f,  1.0f, 1},
        {0.986f, 500, 0.0020f, 350.0f,  1.0f, 1},
        {0.992f, 550, 0.0013f, 1000.0f, 1.0f, 1},
        {0.996f, 600, 0.0008f, 3000.0f, 1.0f, 1},
        {0.9985f,600, 0.00045f,9000.0f, 1.0f, 1},
        {1.000f, 750, 0.00025f,25000.0f,1.0f, 1},
        {1.000f, 650, 0.00012f,60000.0f,1.0f, 1},
        {1.000f, 550, 0.00005f,120000.0f,1.0f,1}
    };
    const Stage stages_deep[16] = {
        {0.880f, 350, 0.0160f, 5.0f,     0.0f, 0},
        {0.915f, 350, 0.0130f, 5.0f,     0.0f, 0},
        {0.948f, 450, 0.0100f, 5.0f,     0.0f, 0},
        {0.970f, 500, 0.0070f, 15.0f,    1.0f, 1},
        {0.955f, 450, 0.0080f, 45.0f,    1.0f, 1},
        {0.978f, 600, 0.0052f, 90.0f,    1.0f, 1},
        {0.964f, 450, 0.0055f, 180.0f,   1.0f, 1},
        {0.986f, 650, 0.0036f, 350.0f,   1.0f, 1},
        {0.975f, 500, 0.0038f, 600.0f,   1.0f, 1},
        {0.991f, 700, 0.0024f, 1200.0f,  1.0f, 1},
        {0.995f, 700, 0.0014f, 3200.0f,  1.0f, 1},
        {0.998f, 700, 0.00075f,8000.0f,  1.0f, 1},
        {0.9993f,700, 0.00040f,18000.0f, 1.0f, 1},
        {1.000f, 850, 0.00022f,35000.0f, 1.0f, 1},
        {1.000f, 750, 0.00010f,75000.0f, 1.0f, 1},
        {1.000f, 650, 0.000045f,140000.0f,1.0f,1}
    };
    const Stage stages_micro[12] = {
        {0.965f, 350, 0.0038f, 25.0f,    1.0f, 1},
        {0.975f, 400, 0.0030f, 70.0f,    1.0f, 1},
        {0.983f, 450, 0.0023f, 180.0f,   1.0f, 1},
        {0.989f, 500, 0.0017f, 450.0f,   1.0f, 1},
        {0.982f, 400, 0.0020f, 800.0f,   1.0f, 1},
        {0.991f, 550, 0.0013f, 1600.0f,  1.0f, 1},
        {0.995f, 600, 0.00085f,4000.0f,  1.0f, 1},
        {0.9975f,600, 0.00050f,10000.0f, 1.0f, 1},
        {0.9990f,650, 0.00030f,25000.0f, 1.0f, 1},
        {1.000f, 750, 0.00018f,50000.0f, 1.0f, 1},
        {1.000f, 650, 0.00009f,90000.0f, 1.0f, 1},
        {1.000f, 550, 0.00004f,150000.0f,1.0f, 1}
    };
    const Stage stages_breathe[12] = {
        {0.940f, 450, 0.0110f, 20.0f,    0.0f, 0},
        {0.955f, 450, 0.0085f, 50.0f,    1.0f, 1},
        {0.968f, 500, 0.0060f, 120.0f,   1.0f, 1},
        {0.958f, 450, 0.0065f, 250.0f,   1.0f, 1},
        {0.978f, 550, 0.0040f, 550.0f,   1.0f, 1},
        {0.986f, 600, 0.0026f, 1200.0f,  1.0f, 1},
        {0.992f, 650, 0.0017f, 3000.0f,  1.0f, 1},
        {0.996f, 700, 0.0010f, 7500.0f,  1.0f, 1},
        {0.9985f,700, 0.00050f,18000.0f, 1.0f, 1},
        {1.000f, 800, 0.00022f,45000.0f, 1.0f, 1},
        {1.000f, 750, 0.00010f,90000.0f, 1.0f, 1},
        {1.000f, 650, 0.00004f,150000.0f,1.0f, 1}
    };
    const Stage stages_melt[12] = {
        {0.850f, 350, 0.0150f, 12.0f,    0.0f, 0},
        {0.880f, 400, 0.0120f, 30.0f,    0.0f, 1},
        {0.910f, 450, 0.0095f, 70.0f,    1.0f, 1},
        {0.935f, 450, 0.0075f, 160.0f,   1.0f, 1},
        {0.958f, 500, 0.0055f, 360.0f,   1.0f, 1},
        {0.974f, 550, 0.0038f, 800.0f,   1.0f, 1},
        {0.986f, 600, 0.0024f, 1800.0f,  1.0f, 1},
        {0.993f, 650, 0.0014f, 4200.0f,  1.0f, 1},
        {0.9975f,650, 0.00075f,10000.0f, 1.0f, 1},
        {1.000f, 750, 0.00035f,25000.0f, 1.0f, 1},
        {1.000f, 750, 0.00014f,65000.0f, 1.0f, 1},
        {1.000f, 650, 0.00005f,140000.0f,1.0f, 1}
    };
    const Stage stages_masked[12] = {
        {0.800f, 350, 0.0180f, 15.0f,    0.0f, 0},
        {0.840f, 400, 0.0150f, 35.0f,    0.0f, 0},
        {0.880f, 400, 0.0120f, 80.0f,    0.0f, 0},
        {0.915f, 450, 0.0090f, 180.0f,   0.0f, 0},
        {0.945f, 500, 0.0065f, 400.0f,   0.0f, 0},
        {0.968f, 550, 0.0045f, 900.0f,   0.0f, 0},
        {0.984f, 600, 0.0028f, 2000.0f,  0.0f, 0},
        {0.994f, 650, 0.0016f, 5000.0f,  0.0f, 0},
        {0.9985f,650, 0.0008f, 12000.0f, 0.0f, 0},
        {1.000f, 700, 0.00030f,30000.0f, 1.0f, 1},
        {1.000f, 750, 0.00012f,75000.0f, 1.0f, 1},
        {1.000f, 650, 0.00004f,150000.0f,1.0f, 1}
    };
    const int n_stages = (fast_mode == 2) ? 10 : ((fast_mode == 3) ? 16 : ((fast_mode >= 4) ? 12 : 14));
    const Stage* stages = (fast_mode == 2) ? stages_gentle
        : ((fast_mode == 3) ? stages_deep
        : ((fast_mode == 4) ? stages_micro
        : ((fast_mode == 5) ? stages_breathe
        : ((fast_mode == 6) ? stages_melt
        : ((fast_mode == 7) ? stages_masked : stages_fast)))));
    const float mult_steps = (fast_mode == 0) ? 1.8f : ((fast_mode == 3) ? 1.45f : ((fast_mode >= 4) ? 1.20f : 1.0f));

    int step_g = 0;
    float w_align_base = (fast_mode >= 4) ? 0.0f : ((b % 2 == 1) ? 0.016f : 0.0f);

    for (int st = 0; st < n_stages; ++st) {
        float scale_a = stages[st].a;
        int n_steps = (int)(stages[st].steps * mult_steps);
        float lr_base = stages[st].lr;
        float w_pen = stages[st].w_pen;
        int stage_tight = (stages[st].opt_L != 0) ? 1 : 0;
        int opt_L = (use_elastic_L && stage_tight) ? 1 : 0;
        float half_a = 0.5f * scale_a;
        float sg_eps = (scale_a < 0.99f) ? 1e-5f : 1e-7f;
        float w_align = (scale_a < 0.968f) ? w_align_base * fmaxf(0.0f, (0.968f - scale_a) / 0.088f) : 0.0f;

        for (int r = 0; r < 4; ++r) {
            mx[r] *= 0.5f; my[r] *= 0.5f; mt[r] *= 0.5f;
        }
        __syncthreads();

        rebuild_grid(N, sh_x0, sh_y0, sh_L, sx, sy, cell_head, cell_next, sh_gx0, sh_gy0, sh_cw);

        for (int it = 0; it < n_steps; ++it) {
            if (it > 0 && (it % 10 == 0)) {
                rebuild_grid(N, sh_x0, sh_y0, sh_L, sx, sy, cell_head, cell_next, sh_gx0, sh_gy0, sh_cw);
            }

            float cur_L = sh_L, cur_x0 = sh_x0, cur_y0 = sh_y0;
            float gx0 = sh_gx0, gy0 = sh_gy0, cw = sh_cw, inv_cw = 1.0f / cw;
            float fx_reg[4], fy_reg[4], fth_reg[4], qe_reg[4];
            float g2_local = 0.0f;
            float dvx_loc = 0.0f, dvy_loc = 0.0f, wall_loc = 0.0f;
            float pen_scale = stage_tight ? w_pen : 1.0f;
            float wall_scale = stage_tight ? w_pen : 5.0f;

            for (int r = 0; r < 4; ++r) {
                int i = tid + r * BLOCK_THREADS;
                if (i >= N) break;
                float xi = sx[i], yi = sy[i], ci = sc[i], si = ss[i];
                float ext = half_a * (fabsf(ci) + fabsf(si));
                float eps2 = sg_eps * sg_eps;
                float sc_s = ci * rsqrtf(ci * ci + eps2);
                float ss_s = si * rsqrtf(si * si + eps2);
                float dext = half_a * (ss_s * ci - sc_s * si);

                float vL = fmaxf(0.0f, cur_x0 + ext - xi);
                float vR = fmaxf(0.0f, xi + ext - cur_x0 - cur_L);
                float vB = fmaxf(0.0f, cur_y0 + ext - yi);
                float vT = fmaxf(0.0f, yi + ext - cur_y0 - cur_L);

                float fx = wall_scale * (-2.0f * vL + 2.0f * vR);
                float fy = wall_scale * (-2.0f * vB + 2.0f * vT);
                float fth = wall_scale * 2.0f * (vL + vR + vB + vT) * dext;
                float qe = wall_scale * (vL * vL + vR * vR + vB * vB + vT * vT);

                if (opt_L) {
                    dvx_loc += (vR - vL);
                    dvy_loc += (vT - vB);
                    wall_loc += (vL + vR + vB + vT);
                }

                int cx = max(0, min(GRID_G - 1, (int)floorf((xi - gx0) * inv_cw)));
                int cy = max(0, min(GRID_G - 1, (int)floorf((yi - gy0) * inv_cw)));
                float ep = 0.0f, gxp = 0.0f, gyp = 0.0f, gthp = 0.0f;
                float thi = sth[i];

                for (int dy = -1; dy <= 1; ++dy) {
                    int ny = cy + dy;
                    if (ny < 0 || ny >= GRID_G) continue;
                    for (int dx = -1; dx <= 1; ++dx) {
                        int nx = cx + dx;
                        if (nx < 0 || nx >= GRID_G) continue;
                        int j = cell_head[ny * GRID_G + nx];
                        while (j >= 0) {
                            if (j != i) {
                                float p_ij, tx, ty, tt;
                                pair_pen_and_grad_f32(
                                    xi, yi, ci, si, sx[j], sy[j], sc[j], ss[j],
                                    i, j, half_a, sg_eps, p_ij, tx, ty, tt
                                );
                                ep += 0.5f * p_ij * p_ij;
                                gxp += tx;
                                gyp += ty;
                                gthp += tt;
                                if (w_align > 0.0f) {
                                    float ddx = sx[j] - xi, ddy = sy[j] - yi;
                                    if (ddx * ddx + ddy * ddy < 2.25f * scale_a * scale_a) {
                                        float dth = thi - sth[j];
                                        gthp += w_align * 2.0f * sinf(4.0f * dth);
                                    }
                                }
                            }
                            j = (int)cell_next[j];
                        }
                    }
                }
                fx += pen_scale * gxp;
                fy += pen_scale * gyp;
                fth += pen_scale * gthp;
                qe += pen_scale * ep;

                fx_reg[r] = fx; fy_reg[r] = fy; fth_reg[r] = fth; qe_reg[r] = qe;
                sqe[i] = qe;
                bool frozen_i = (fast_mode == 7 && st < 9 && smob[i] == 0);
                if (!frozen_i) {
                    g2_local += fx * fx + fy * fy + fth * fth;
                }
            }

            float sum_vx   = opt_L ? block_reduce_sum(dvx_loc, red_buf) : 0.0f;
            float sum_vy   = opt_L ? block_reduce_sum(dvy_loc, red_buf) : 0.0f;
            float sum_wall = opt_L ? block_reduce_sum(wall_loc, red_buf) : 0.0f;
            float g2_tot   = block_reduce_sum(g2_local, red_buf);

            if (tid == 0) {
                float gn = sqrtf(g2_tot + 1e-12f);
                float clip_target = stage_tight ? (10.0f * w_pen) : 10.0f;
                sh_step_sc = fminf(1.0f, clip_target / gn);
            }
            __syncthreads();
            float gscale = sh_step_sc;

            float frac = (float)it / (float)max(1, n_steps - 1);
            float lr_mult = (frac < 0.5f) ? 1.0f : 0.5f * (1.0f + cosf(3.14159265f * (frac - 0.5f) * 2.0f));
            float lr_xy = lr_base * (0.25f + 0.75f * lr_mult);
            float lr_th = lr_xy * ((scale_a < 0.97f) ? 1.3f : 0.75f);

            step_g += 1;
            float b1 = 0.85f, b2 = 0.99f;
            float bc1 = 1.0f - powf(b1, (float)min(step_g, 50));
            float bc2 = 1.0f - powf(b2, (float)min(step_g, 50));

            for (int r = 0; r < 4; ++r) {
                int i = tid + r * BLOCK_THREADS;
                if (i >= N) break;
                if (fast_mode == 7 && st < 9 && smob[i] == 0) continue;
                float fx = fx_reg[r] * gscale;
                float fy = fy_reg[r] * gscale;
                float ft = fth_reg[r] * gscale;

                mx[r] = b1 * mx[r] + (1.0f - b1) * fx;
                vx[r] = b2 * vx[r] + (1.0f - b2) * fx * fx;
                my[r] = b1 * my[r] + (1.0f - b1) * fy;
                vy[r] = b2 * vy[r] + (1.0f - b2) * fy * fy;
                mt[r] = b1 * mt[r] + (1.0f - b1) * ft;
                vt[r] = b2 * vt[r] + (1.0f - b2) * ft * ft;

                float nx = sx[i] - lr_xy * (mx[r] / bc1) / (sqrtf(vx[r] / bc2) + 1e-8f);
                float ny = sy[i] - lr_xy * (my[r] / bc1) / (sqrtf(vy[r] / bc2) + 1e-8f);
                float nt = sth[i] - lr_th * (mt[r] / bc1) / (sqrtf(vt[r] / bc2) + 1e-8f);

                nx = fminf(fmaxf(nx, cur_x0 - 0.25f), cur_x0 + cur_L + 0.25f);
                ny = fminf(fmaxf(ny, cur_y0 - 0.25f), cur_y0 + cur_L + 0.25f);
                sx[i] = nx; sy[i] = ny; sth[i] = nt;
                sc[i] = cosf(nt); ss[i] = sinf(nt);
            }

            if (opt_L && tid == 0) {
                float sfrac = (fast_mode == 2) ? ((float)st / 9.0f)
                    : ((fast_mode >= 4) ? ((float)st / 11.0f)
                    : ((fast_mode == 3) ? ((float)max(0, st - 3) / 12.0f) : ((float)max(0, st - 4) / 9.0f)));
                float target_w = (fast_mode == 2)
                    ? (0.030f * (1.0f - sfrac) + 0.0001f)
                    : ((fast_mode >= 4)
                        ? (0.020f * (1.0f - sfrac) + 0.00015f)
                        : ((fast_mode == 3)
                            ? (0.028f * (1.0f - sfrac) + 0.00015f)
                            : (0.045f * (1.0f - sfrac) + 0.0005f)));
                float max_L_b = sh_L_init + ((fast_mode == 2)
                    ? ((b % 3 == 0) ? 0.002f : ((b % 3 == 1) ? 0.010f : 0.025f))
                    : ((fast_mode == 7)
                        ? ((b % 4 == 0) ? 0.0001f : ((b % 4 == 1) ? 0.0004f : ((b % 4 == 2) ? 0.0010f : 0.0020f)))
                        : ((fast_mode == 6)
                            ? ((b % 4 == 0) ? 0.0001f : ((b % 4 == 1) ? 0.0005f : ((b % 4 == 2) ? 0.0012f : 0.0025f)))
                            : ((fast_mode == 4 || fast_mode == 5)
                                ? ((b % 4 == 0) ? -0.0002f : ((b % 4 == 1) ? 0.0008f : ((b % 4 == 2) ? 0.0025f : 0.0060f)))
                                : ((fast_mode == 3)
                                    ? ((b % 4 == 0) ? -0.0002f : ((b % 4 == 1) ? 0.0015f : ((b % 4 == 2) ? 0.0045f : 0.0120f)))
                                    : ((b % 3 == 0) ? 0.003f : ((b % 3 == 1) ? 0.015f : 0.040f)))))));
                float min_L_b = fmaxf(1.0f, sh_L_init * 0.75f);
                float step_c = fminf(0.0018f, lr_xy * 0.5f);
                sh_x0 += step_c * tanhf(sum_vx * 2.0f);
                sh_y0 += step_c * tanhf(sum_vy * 2.0f);
                float denom_w = (fast_mode >= 4 && sum_wall < target_w)
                    ? fmaxf(target_w, 1.0e-5f)
                    : fmaxf(target_w, (fast_mode >= 4) ? 0.008f : 0.015f);
                float err_w = (sum_wall - target_w) / denom_w;
                sh_L += step_c * 0.8f * fminf(1.0f, fmaxf(-1.0f, err_w));
                sh_L = fminf(max_L_b, fmaxf(min_L_b, sh_L));
            }
            __syncthreads();

            // Active discrete + collective moves during mobile inflation / anneal
            const bool do_moves = (fast_mode == 7) ? false : ((fast_mode >= 3)
                ? (scale_a <= 0.995f && ((it == n_steps / 4) || (it == n_steps / 2) || (it == (3 * n_steps) / 4)))
                : (scale_a <= 0.988f && ((it == n_steps / 3) || (it == (2 * n_steps) / 3))));
            if (do_moves) {
                rebuild_grid(N, sh_x0, sh_y0, sh_L, sx, sy, cell_head, cell_next, sh_gx0, sh_gy0, sh_cw);
                float w_wall_unit = stage_tight ? 1.0f : 5.0f;

                // 1) Parallel 256-thread void relocation on the worst-stress squares
                if (fast_mode != 2 && fast_mode < 4) {
                const int n_mv = (fast_mode == 3) ? 3 : 2;
                const int n_pr = (fast_mode == 3) ? 4 : 1;
                const int n_ang = (fast_mode == 3) ? 7 : 3;
                for (int mv = 0; mv < n_mv; ++mv) {
                    compute_total_energy_f32(
                        N, sh_x0, sh_y0, sh_L, sh_gx0, sh_gy0, sh_cw,
                        scale_a, w_wall_unit, sx, sy, sc, ss, sqe, cell_head, cell_next, red_buf
                    );
                    float my_max = -1.0f;
                    int my_idx = -1;
                    for (int r = 0; r < 4; ++r) {
                        int i = tid + r * BLOCK_THREADS;
                        if (i < N && sqe[i] > my_max) { my_max = sqe[i]; my_idx = i; }
                    }
                    red_buf[tid] = my_max;
                    ired_buf[tid] = my_idx;
                    __syncthreads();
                    for (int s = BLOCK_THREADS / 2; s > 0; s >>= 1) {
                        if (tid < s && red_buf[tid + s] > red_buf[tid]) {
                            red_buf[tid] = red_buf[tid + s];
                            ired_buf[tid] = ired_buf[tid + s];
                        }
                        __syncthreads();
                    }
                    int w_i = ired_buf[0];
                    float e_old = red_buf[0];
                    __syncthreads();
                    if (w_i < 0 || e_old <= 1e-6f) break;

                    float best_px = sh_x0 + 0.52f, best_py = sh_y0 + 0.52f;
                    float best_ang = 0.0f;
                    float best_ep = 1e9f;
                    float cand_a[7] = {0.0f, 0.78539816f, -0.78539816f, 0.4636476f, -0.4636476f, 0.3217506f, sth[w_i]};
                    for (int pr = 0; pr < n_pr; ++pr) {
                        float px, py;
                        if (fast_mode == 3 && pr == 3) {
                            // Wall & corner bias (width ~0.5..2.8 from boundary)
                            float margin = 0.50f + rand_u01(rng) * fminf(2.3f, 0.25f * sh_L);
                            uint32_t side = xorshift32(rng) & 3u;
                            float along = 0.52f + rand_u01(rng) * fmaxf(0.2f, sh_L - 1.04f);
                            px = sh_x0 + ((side == 0) ? margin : ((side == 1) ? (sh_L - margin) : along));
                            py = sh_y0 + ((side == 2) ? margin : ((side == 3) ? (sh_L - margin) : along));
                        } else {
                            px = sh_x0 + 0.52f + rand_u01(rng) * fmaxf(0.2f, sh_L - 1.04f);
                            py = sh_y0 + 0.52f + rand_u01(rng) * fmaxf(0.2f, sh_L - 1.04f);
                        }
                        for (int ak = 0; ak < n_ang; ++ak) {
                            int ai = (fast_mode == 3) ? ak : ((ak == 2) ? 6 : ak);
                            float ec = eval_single_sq_f32(
                                w_i, px, py, cand_a[ai], N, sh_x0, sh_y0, sh_L,
                                sh_gx0, sh_gy0, sh_cw, scale_a, w_wall_unit,
                                sx, sy, sc, ss, cell_head, cell_next
                            );
                            if (ec < best_ep) { best_ep = ec; best_px = px; best_py = py; best_ang = cand_a[ai]; }
                        }
                    }
                    red_buf[tid] = best_ep;
                    ired_buf[tid] = tid;
                    __syncthreads();
                    // Parallel argmin
                    for (int s = BLOCK_THREADS / 2; s > 0; s >>= 1) {
                        if (tid < s && red_buf[tid + s] < red_buf[tid]) {
                            red_buf[tid] = red_buf[tid + s];
                            ired_buf[tid] = ired_buf[tid + s];
                        }
                        __syncthreads();
                    }
                    int win_t = ired_buf[0];
                    float e_best = red_buf[0];
                    __syncthreads();
                    if (tid == win_t && e_best + 1e-6f < e_old) {
                        sx[w_i] = best_px; sy[w_i] = best_py; sth[w_i] = best_ang;
                        sc[w_i] = cosf(best_ang); ss[w_i] = sinf(best_ang);
                    }
                    __syncthreads();
                    rebuild_grid(N, sh_x0, sh_y0, sh_L, sx, sy, cell_head, cell_next, sh_gx0, sh_gy0, sh_cw);
                }
                }

                // 2) Collective orientation-cluster / spatial-slab rigid translation move
                for (int cm = 0; cm < 2; ++cm) {
                    int seed_idx = (int)(xorshift32(rng) % (uint32_t)N);
                    if (tid == 0) ired_buf[0] = seed_idx;
                    __syncthreads();
                    seed_idx = ired_buf[0];
                    __syncthreads();

                    float seed_th = sth[seed_idx];
                    float seed_x = sx[seed_idx], seed_y = sy[seed_idx];
                    float cx0 = sh_x0 + 0.5f * sh_L, cy0 = sh_y0 + 0.5f * sh_L;
                    float seed_d1 = (seed_x - cx0) - (seed_y - cy0);
                    float seed_d2 = (seed_x - cx0) + (seed_y - cy0);
                    bool seed_upright = (fabsf(sinf(2.0f * seed_th)) < 0.25f);
                    float e0 = compute_total_energy_f32(
                        N, sh_x0, sh_y0, sh_L, sh_gx0, sh_gy0, sh_cw,
                        scale_a, w_wall_unit, sx, sy, sc, ss, sqe, cell_head, cell_next, red_buf
                    );
                    float loc_fx = 0.0f, loc_fy = 0.0f;
                    float ox[4], oy[4];
                    uint8_t in_cl[4] = {0,0,0,0};
                    float half_a_loc = 0.5f * scale_a;
                    float inv_cw_l = 1.0f / sh_cw;

                    for (int r = 0; r < 4; ++r) {
                        int i = tid + r * BLOCK_THREADS;
                        if (i >= N) break;
                        ox[r] = sx[i]; oy[r] = sy[i];
                        bool mask;
                        if (cm == 0) {
                            mask = (fabsf(cosf(2.0f * (sth[i] - seed_th))) > 0.96f);
                            if (mask && fast_mode == 4 && seed_upright) {
                                // Split upright 0-deg crystal into independent half-grains (g_hi vs g_lo)
                                float di = ((st + it) & 1)
                                    ? (((sx[i] - cx0) + (sy[i] - cy0)) * seed_d2)
                                    : (((sx[i] - cx0) - (sy[i] - cy0)) * seed_d1);
                                mask = (di >= 0.0f);
                            }
                        } else {
                            mask = (fabsf(sx[i] - seed_x) < 3.5f || fabsf(sy[i] - seed_y) < 3.5f);
                        }
                        if (mask) {
                            in_cl[r] = 1;
                            float xi = sx[i], yi = sy[i], ci = sc[i], si = ss[i];
                            float ext = half_a_loc * (fabsf(ci) + fabsf(si));
                            float vL = fmaxf(0.0f, sh_x0 + ext - xi);
                            float vR = fmaxf(0.0f, xi + ext - sh_x0 - sh_L);
                            float vB = fmaxf(0.0f, sh_y0 + ext - yi);
                            float vT = fmaxf(0.0f, yi + ext - sh_y0 - sh_L);
                            loc_fx += w_wall_unit * (-2.0f * vL + 2.0f * vR);
                            loc_fy += w_wall_unit * (-2.0f * vB + 2.0f * vT);
                            int cx = max(0, min(GRID_G - 1, (int)floorf((xi - sh_gx0) * inv_cw_l)));
                            int cy = max(0, min(GRID_G - 1, (int)floorf((yi - sh_gy0) * inv_cw_l)));
                            for (int dy = -1; dy <= 1; ++dy) {
                                int ny = cy + dy; if (ny < 0 || ny >= GRID_G) continue;
                                for (int dx = -1; dx <= 1; ++dx) {
                                    int nx = cx + dx; if (nx < 0 || nx >= GRID_G) continue;
                                    int j = cell_head[ny * GRID_G + nx];
                                    while (j >= 0) {
                                        if (j != i) {
                                            float p_ij, tx, ty, tt;
                                            pair_pen_and_grad_f32(xi, yi, ci, si, sx[j], sy[j], sc[j], ss[j],
                                                                  i, j, half_a_loc, 1e-6f, p_ij, tx, ty, tt);
                                            loc_fx += tx; loc_fy += ty;
                                        }
                                        j = (int)cell_next[j];
                                    }
                                }
                            }
                        }
                    }
                    float tot_fx = block_reduce_sum(loc_fx, red_buf);
                    float tot_fy = block_reduce_sum(loc_fy, red_b2);
                    float fnorm = sqrtf(tot_fx * tot_fx + tot_fy * tot_fy);
                    if (fnorm > 1e-7f) {
                        float step_cl = 0.018f * (1.02f - scale_a);
                        float ddx = -step_cl * (tot_fx / fnorm);
                        float ddy = -step_cl * (tot_fy / fnorm);
                        for (int r = 0; r < 4; ++r) {
                            int i = tid + r * BLOCK_THREADS;
                            if (i < N && in_cl[r]) { sx[i] = ox[r] + ddx; sy[i] = oy[r] + ddy; }
                        }
                        rebuild_grid(N, sh_x0, sh_y0, sh_L, sx, sy, cell_head, cell_next, sh_gx0, sh_gy0, sh_cw);
                        float e1 = compute_total_energy_f32(
                            N, sh_x0, sh_y0, sh_L, sh_gx0, sh_gy0, sh_cw,
                            scale_a, w_wall_unit, sx, sy, sc, ss, sqe, cell_head, cell_next, red_buf
                        );
                        if (e1 >= e0) {
                            for (int r = 0; r < 4; ++r) {
                                int i = tid + r * BLOCK_THREADS;
                                if (i < N && in_cl[r]) { sx[i] = ox[r]; sy[i] = oy[r]; }
                            }
                            rebuild_grid(N, sh_x0, sh_y0, sh_L, sx, sy, cell_head, cell_next, sh_gx0, sh_gy0, sh_cw);
                        }
                    }
                }
            }
        }
    }

    for (int i = tid; i < N; i += BLOCK_THREADS) {
        out_x[b * N + i] = sx[i];
        out_y[b * N + i] = sy[i];
        out_th[b * N + i] = sth[i];
    }
    if (tid == 0) {
        out_x0[b] = sh_x0;
        out_y0[b] = sh_y0;
        out_L[b] = sh_L;
    }
}

// Float64 final polish + exact all-pairs uniform dilation L_cert (one block per trajectory)
__global__ void polish_and_cert_fp64(
    int B, int N, int n_polish_steps, int use_elastic_L, int fast_mode,
    const float* __restrict__ in_x,
    const float* __restrict__ in_y,
    const float* __restrict__ in_th,
    const float* __restrict__ in_x0,
    const float* __restrict__ in_y0,
    const float* __restrict__ in_L,
    double* __restrict__ out_x,
    double* __restrict__ out_y,
    double* __restrict__ out_th,
    double* __restrict__ out_L_cert,
    double* __restrict__ out_lam,
    double* __restrict__ out_pen
) {
    int b = blockIdx.x;
    if (b >= B) return;
    int tid = threadIdx.x;

    // 5 * 1024 * 8 = 40 KB + 2.3 KB cell_head + 2 KB cell_next + 2 KB red = ~46.3 KB < 48 KB
    __shared__ double sx[MAX_N];
    __shared__ double sy[MAX_N];
    __shared__ double sth[MAX_N];
    __shared__ double sc[MAX_N];
    __shared__ double ss[MAX_N];
    __shared__ int32_t cell_head[NUM_CELLS];
    __shared__ int16_t cell_next[MAX_N];
    __shared__ double red_buf[BLOCK_THREADS];
    __shared__ double sh_x0, sh_y0, sh_L, sh_L_init, sh_gx0, sh_gy0, sh_cw, sh_sc;

    for (int i = tid; i < N; i += BLOCK_THREADS) {
        sx[i] = (double)in_x[b * N + i];
        sy[i] = (double)in_y[b * N + i];
        sth[i] = (double)in_th[b * N + i];
        sc[i] = cos(sth[i]);
        ss[i] = sin(sth[i]);
    }
    if (tid == 0) {
        sh_x0 = (double)in_x0[b];
        sh_y0 = (double)in_y0[b];
        sh_L  = (double)in_L[b];
        sh_L_init = sh_L;
    }
    __syncthreads();

    double mx[4] = {0,0,0,0}, vx[4] = {0,0,0,0};
    double my[4] = {0,0,0,0}, vy[4] = {0,0,0,0};
    double mt[4] = {0,0,0,0}, vt[4] = {0,0,0,0};
    const double sg_eps = 1e-11;

    for (int it = 0; it < n_polish_steps; ++it) {
        if (it % 12 == 0) {
            for (int c = tid; c < NUM_CELLS; c += BLOCK_THREADS) cell_head[c] = -1;
            if (tid == 0) {
                sh_gx0 = sh_x0 - 1.0;
                sh_gy0 = sh_y0 - 1.0;
                sh_cw = fmax(1.48, (sh_L + 2.0) / (double)GRID_G);
            }
            __syncthreads();
            double inv_cw = 1.0 / sh_cw;
            for (int i = tid; i < N; i += BLOCK_THREADS) {
                int cx = max(0, min(GRID_G - 1, (int)floor((sx[i] - sh_gx0) * inv_cw)));
                int cy = max(0, min(GRID_G - 1, (int)floor((sy[i] - sh_gy0) * inv_cw)));
                cell_next[i] = (int16_t)atomicExch(&cell_head[cy * GRID_G + cx], i);
            }
            __syncthreads();
        }

        double cur_x0 = sh_x0, cur_y0 = sh_y0, cur_L = sh_L;
        double gx0 = sh_gx0, gy0 = sh_gy0, cw = sh_cw, inv_cw = 1.0 / cw;
        double w_pen = 1.0e5 * (1.0 + 3.0 * ((double)it / (double)max(1, n_polish_steps - 1)));
        double fx_r[4], fy_r[4], ft_r[4];
        double dvx_l = 0.0, dvy_l = 0.0, wall_l = 0.0, g2_l = 0.0;

        for (int r = 0; r < 4; ++r) {
            int i = tid + r * BLOCK_THREADS;
            if (i >= N) break;
            double xi = sx[i], yi = sy[i], ci = sc[i], si = ss[i];
            double ext = 0.5 * (fabs(ci) + fabs(si));
            double eps2 = sg_eps * sg_eps;
            double sc_s = ci / sqrt(ci * ci + eps2);
            double ss_s = si / sqrt(si * si + eps2);
            double dext = 0.5 * (ss_s * ci - sc_s * si);

            double vL = fmax(0.0, cur_x0 + ext - xi);
            double vR = fmax(0.0, xi + ext - cur_x0 - cur_L);
            double vB = fmax(0.0, cur_y0 + ext - yi);
            double vT = fmax(0.0, yi + ext - cur_y0 - cur_L);

            double fx = w_pen * (-2.0 * vL + 2.0 * vR);
            double fy = w_pen * (-2.0 * vB + 2.0 * vT);
            double ft = w_pen * 2.0 * (vL + vR + vB + vT) * dext;
            dvx_l += (vR - vL);
            dvy_l += (vT - vB);
            wall_l += (vL + vR + vB + vT);

            int cx = max(0, min(GRID_G - 1, (int)floor((xi - gx0) * inv_cw)));
            int cy = max(0, min(GRID_G - 1, (int)floor((yi - gy0) * inv_cw)));
            for (int dy = -1; dy <= 1; ++dy) {
                int ny = cy + dy; if (ny < 0 || ny >= GRID_G) continue;
                for (int dx = -1; dx <= 1; ++dx) {
                    int nx = cx + dx; if (nx < 0 || nx >= GRID_G) continue;
                    int j = cell_head[ny * GRID_G + nx];
                    while (j >= 0) {
                        if (j != i) {
                            double ddx = sx[j] - xi, ddy = sy[j] - yi;
                            if (ddx * ddx + ddy * ddy < 2.15) {
                                double cj = sc[j], sj = ss[j];
                                double cr = ci * cj + si * sj;
                                double sr = si * cj - ci * sj;
                                double rad = 0.5 * (1.0 + fabs(cr) + fabs(sr));
                                double eps2 = sg_eps * sg_eps;
                                double drad = 0.5 * ((cr / sqrt(cr * cr + eps2)) * (-sr) + (sr / sqrt(sr * sr + eps2)) * cr);
                                double p0 = ddx * ci + ddy * si, p1 = -ddx * si + ddy * ci;
                                double p2 = ddx * cj + ddy * sj, p3 = -ddx * sj + ddy * cj;
                                double z0 = fabs(p0) - rad, z1 = fabs(p1) - rad;
                                double z2 = fabs(p2) - rad, z3 = fabs(p3) - rad;
                                double zmax = z0, ps = p0, ux = ci, uy = si, isi = 1.0;
                                if (z1 > zmax) { zmax = z1; ps = p1; ux = -si; uy = ci; isi = 1.0; }
                                if (z2 > zmax) { zmax = z2; ps = p2; ux = cj;  uy = sj; isi = 0.0; }
                                if (z3 > zmax) { zmax = z3; ps = p3; ux = -sj; uy = cj; isi = 0.0; }
                                double pen = fmax(0.0, -zmax);
                                if (pen > 0.0) {
                                    double sp = (ps > 0.0) ? 1.0 : ((ps < 0.0) ? -1.0 : ((j > i) ? 1.0 : -1.0));
                                    double w = w_pen * 2.0 * pen;
                                    fx += w * sp * ux;
                                    fy += w * sp * uy;
                                    ft += w * (drad - sp * (ddy * ux - ddx * uy) * isi);
                                }
                            }
                        }
                        j = (int)cell_next[j];
                    }
                }
            }
            fx_r[r] = fx; fy_r[r] = fy; ft_r[r] = ft;
            g2_l += fx * fx + fy * fy + ft * ft;
        }

        double sdx = block_reduce_sum_f64(dvx_l, red_buf);
        double sdy = block_reduce_sum_f64(dvy_l, red_buf);
        double swall = block_reduce_sum_f64(wall_l, red_buf);
        double g2_all = block_reduce_sum_f64(g2_l, red_buf);
        if (tid == 0) sh_sc = fmin(1.0, (10.0 * w_pen) / sqrt(g2_all + 1e-18));
        __syncthreads();
        double gscale = sh_sc;

        double lr = 5.0e-5 * (0.2 + 0.8 * (1.0 - (double)it / (double)max(1, n_polish_steps)));
        double b1 = 0.85, b2 = 0.99;
        double bc1 = 1.0 - pow(b1, (double)min(it + 1, 50));
        double bc2 = 1.0 - pow(b2, (double)min(it + 1, 50));

        for (int r = 0; r < 4; ++r) {
            int i = tid + r * BLOCK_THREADS;
            if (i >= N) break;
            double fx = fx_r[r] * gscale, fy = fy_r[r] * gscale, ft = ft_r[r] * gscale;
            mx[r] = b1 * mx[r] + (1.0 - b1) * fx; vx[r] = b2 * vx[r] + (1.0 - b2) * fx * fx;
            my[r] = b1 * my[r] + (1.0 - b1) * fy; vy[r] = b2 * vy[r] + (1.0 - b2) * fy * fy;
            mt[r] = b1 * mt[r] + (1.0 - b1) * ft; vt[r] = b2 * vt[r] + (1.0 - b2) * ft * ft;
            sx[i] -= lr * (mx[r] / bc1) / (sqrt(vx[r] / bc2) + 1e-12);
            sy[i] -= lr * (my[r] / bc1) / (sqrt(vy[r] / bc2) + 1e-12);
            double nt = sth[i] - 0.5 * lr * (mt[r] / bc1) / (sqrt(vt[r] / bc2) + 1e-12);
            sth[i] = nt; sc[i] = cos(nt); ss[i] = sin(nt);
        }
        if (tid == 0) {
            sh_x0 += 0.5 * lr * tanh(sdx * 5.0);
            sh_y0 += 0.5 * lr * tanh(sdy * 5.0);
            if (use_elastic_L) {
                double denom = (fast_mode >= 4 && swall < 1.0e-4) ? 2.0e-4 : 0.005;
                double err_w = (swall - 1.0e-4) / denom;
                sh_L += 0.4 * lr * fmin(1.0, fmax(-1.0, err_w));
                double max_add = (fast_mode >= 4) ? 0.0015 : 0.015;
                sh_L = fmin(sh_L_init + max_add, fmax(fmax(1.0, sh_L_init * 0.75), sh_L));
            }
        }
        __syncthreads();
    }

    // Shift to strict non-negative bbox [1e-10, ...] before exact all-pairs lambda
    double loc_xmin = 1e18, loc_ymin = 1e18;
    for (int r = 0; r < 4; ++r) {
        int i = tid + r * BLOCK_THREADS;
        if (i >= N) break;
        double e = 0.5 * (fabs(sc[i]) + fabs(ss[i]));
        loc_xmin = fmin(loc_xmin, sx[i] - e);
        loc_ymin = fmin(loc_ymin, sy[i] - e);
    }
    double xmin_all = block_reduce_min_f64(loc_xmin, red_buf);
    double ymin_all = block_reduce_min_f64(loc_ymin, red_buf);

    for (int i = tid; i < N; i += BLOCK_THREADS) {
        sx[i] -= (xmin_all - 1e-10);
        sy[i] -= (ymin_all - 1e-10);
    }
    __syncthreads();

    // Exact O(N^2) all-pairs lambda & max_pen in float64
    double lam_loc = 1.0, pen_loc = 0.0;
    const double tau = 1.5e-10;
    for (int i = tid; i < N; i += BLOCK_THREADS) {
        double xi = sx[i], yi = sy[i], ci = sc[i], si = ss[i];
        for (int j = i + 1; j < N; ++j) {
            double dx = sx[j] - xi, dy = sy[j] - yi;
            if (dx * dx + dy * dy >= 2.05) continue;
            double cj = sc[j], sj = ss[j];
            double rad = 0.5 * (1.0 + fabs(ci * cj + si * sj) + fabs(si * cj - ci * sj));
            double p0 = fabs(dx * ci + dy * si), p1 = fabs(-dx * si + dy * ci);
            double p2 = fabs(dx * cj + dy * sj), p3 = fabs(-dx * sj + dy * cj);
            double vmax = fmax(fmax(p0, p1), fmax(p2, p3));
            double pen = rad - vmax;
            if (pen > pen_loc) pen_loc = pen;
            double req = (vmax > 1e-12) ? ((rad + tau) / vmax) : 1e9;
            if (req > lam_loc) lam_loc = req;
        }
    }
    double lam_all = block_reduce_max_f64(lam_loc, red_buf);
    double pen_all = block_reduce_max_f64(pen_loc, red_buf);

    // Exact L_cert after uniform expansion lam_all
    double lx_min = 1e18, lx_max = -1e18, ly_min = 1e18, ly_max = -1e18;
    for (int i = tid; i < N; i += BLOCK_THREADS) {
        double px = sx[i] * lam_all, py = sy[i] * lam_all;
        double e = 0.5 * (fabs(sc[i]) + fabs(ss[i]));
        lx_min = fmin(lx_min, px - e); lx_max = fmax(lx_max, px + e);
        ly_min = fmin(ly_min, py - e); ly_max = fmax(ly_max, py + e);
        out_x[b * N + i] = px;
        out_y[b * N + i] = py;
        out_th[b * N + i] = sth[i];
    }
    double axmin = block_reduce_min_f64(lx_min, red_buf);
    double axmax = block_reduce_max_f64(lx_max, red_buf);
    double aymin = block_reduce_min_f64(ly_min, red_buf);
    double aymax = block_reduce_max_f64(ly_max, red_buf);

    if (tid == 0) {
        out_L_cert[b] = fmax(axmax - axmin, aymax - aymin);
        out_lam[b] = lam_all;
        out_pen[b] = pen_all;
    }
}

// Kernel specifically for unit-testing exact SAT energy + analytical gradient against PyTorch/FD
__global__ void grad_check_kernel(
    int N, float scale_a, float box_L, float w_wall, float sg_eps,
    const float* __restrict__ in_x,
    const float* __restrict__ in_y,
    const float* __restrict__ in_th,
    float* __restrict__ out_E,
    float* __restrict__ out_gx,
    float* __restrict__ out_gy,
    float* __restrict__ out_gth
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float xi = in_x[i], yi = in_y[i], thi = in_th[i];
    float ci = cosf(thi), si = sinf(thi);
    float half_a = 0.5f * scale_a;
    float ext = half_a * (fabsf(ci) + fabsf(si));
    float eps2 = sg_eps * sg_eps;
    float sc_s = ci * rsqrtf(ci * ci + eps2);
    float ss_s = si * rsqrtf(si * si + eps2);
    float dext = half_a * (ss_s * ci - sc_s * si);

    float vL = fmaxf(0.0f, ext - xi);
    float vR = fmaxf(0.0f, xi + ext - box_L);
    float vB = fmaxf(0.0f, ext - yi);
    float vT = fmaxf(0.0f, yi + ext - box_L);

    float fx = w_wall * (-2.0f * vL + 2.0f * vR);
    float fy = w_wall * (-2.0f * vB + 2.0f * vT);
    float fth = w_wall * 2.0f * (vL + vR + vB + vT) * dext;
    float qe = w_wall * (vL * vL + vR * vR + vB * vB + vT * vT);

    for (int j = 0; j < N; ++j) {
        if (j == i) continue;
        float cj = cosf(in_th[j]), sj = sinf(in_th[j]);
        float p_ij, tx, ty, tt;
        pair_pen_and_grad_f32(xi, yi, ci, si, in_x[j], in_y[j], cj, sj, i, j, half_a, sg_eps, p_ij, tx, ty, tt);
        qe += 0.5f * p_ij * p_ij;
        fx += tx; fy += ty; fth += tt;
    }
    out_E[i] = qe;
    out_gx[i] = fx;
    out_gy[i] = fy;
    out_gth[i] = fth;
}

static void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", msg, cudaGetErrorString(err));
        exit(2);
    }
}

int main(int argc, char** argv) {
    const char* in_path = nullptr;
    const char* out_path = nullptr;
    int grad_check = 0;
    float gc_a = 1.0f, gc_L = 5.0f, gc_w = 5.0f;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--in") == 0 && i + 1 < argc) in_path = argv[++i];
        else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) out_path = argv[++i];
        else if (strcmp(argv[i], "--grad-check") == 0) grad_check = 1;
        else if (strcmp(argv[i], "--a") == 0 && i + 1 < argc) gc_a = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--L") == 0 && i + 1 < argc) gc_L = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--w") == 0 && i + 1 < argc) gc_w = (float)atof(argv[++i]);
    }
    if (!in_path || !out_path) {
        fprintf(stderr, "Usage: %s --in <in.bin> --out <out.bin> [--grad-check]\n", argv[0]);
        return 1;
    }

    FILE* fin = fopen(in_path, "rb");
    if (!fin) { perror("fopen in"); return 1; }

    if (grad_check) {
        int32_t N = 0;
        if (fread(&N, sizeof(int32_t), 1, fin) != 1) return 1;
        float *hx = (float*)malloc(N * sizeof(float));
        float *hy = (float*)malloc(N * sizeof(float));
        float *ht = (float*)malloc(N * sizeof(float));
        fread(hx, sizeof(float), N, fin);
        fread(hy, sizeof(float), N, fin);
        fread(ht, sizeof(float), N, fin);
        fclose(fin);

        float *dx, *dy, *dt, *de, *dgx, *dgy, *dgt;
        cudaMalloc(&dx, N * sizeof(float)); cudaMalloc(&dy, N * sizeof(float)); cudaMalloc(&dt, N * sizeof(float));
        cudaMalloc(&de, N * sizeof(float)); cudaMalloc(&dgx, N * sizeof(float)); cudaMalloc(&dgy, N * sizeof(float)); cudaMalloc(&dgt, N * sizeof(float));
        cudaMemcpy(dx, hx, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dy, hy, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dt, ht, N * sizeof(float), cudaMemcpyHostToDevice);
        grad_check_kernel<<<(N + 127) / 128, 128>>>(N, gc_a, gc_L, gc_w, 1e-6f, dx, dy, dt, de, dgx, dgy, dgt);
        check_cuda(cudaDeviceSynchronize(), "grad_check");

        float *he = (float*)malloc(N * sizeof(float));
        float *hgx = (float*)malloc(N * sizeof(float));
        float *hgy = (float*)malloc(N * sizeof(float));
        float *hgt = (float*)malloc(N * sizeof(float));
        cudaMemcpy(he, de, N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hgx, dgx, N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hgy, dgy, N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hgt, dgt, N * sizeof(float), cudaMemcpyDeviceToHost);

        FILE* fout = fopen(out_path, "wb");
        fwrite(he, sizeof(float), N, fout);
        fwrite(hgx, sizeof(float), N, fout);
        fwrite(hgy, sizeof(float), N, fout);
        fwrite(hgt, sizeof(float), N, fout);
        fclose(fout);
        return 0;
    }

    SolveParams sp;
    if (fread(&sp, sizeof(SolveParams), 1, fin) != 1) return 1;
    if (sp.N <= 0 || sp.N > MAX_N || sp.B <= 0) {
        fprintf(stderr, "Invalid B=%d N=%d (max %d)\n", sp.B, sp.N, MAX_N);
        return 1;
    }
    size_t bn = (size_t)sp.B * (size_t)sp.N;
    float *h_x = (float*)malloc(bn * sizeof(float));
    float *h_y = (float*)malloc(bn * sizeof(float));
    float *h_th = (float*)malloc(bn * sizeof(float));
    float *h_L = (float*)malloc(sp.B * sizeof(float));
    if (fread(h_x, sizeof(float), bn, fin) != bn) return 1;
    if (fread(h_y, sizeof(float), bn, fin) != bn) return 1;
    if (fread(h_th, sizeof(float), bn, fin) != bn) return 1;
    if (fread(h_L, sizeof(float), sp.B, fin) != (size_t)sp.B) return 1;
    uint8_t *h_mob = nullptr;
    uint32_t magic = 0;
    if (fread(&magic, sizeof(uint32_t), 1, fin) == 1 && magic == 0x4D41534Bu) {
        h_mob = (uint8_t*)malloc(bn * sizeof(uint8_t));
        if (fread(h_mob, sizeof(uint8_t), bn, fin) != bn) {
            free(h_mob);
            h_mob = nullptr;
        }
    }
    fclose(fin);

    float *d_x, *d_y, *d_th, *d_L, *d_ox, *d_oy, *d_oth, *d_ox0, *d_oy0, *d_oL;
    uint8_t *d_mob = nullptr;
    check_cuda(cudaMalloc(&d_x, bn * sizeof(float)), "malloc d_x");
    check_cuda(cudaMalloc(&d_y, bn * sizeof(float)), "malloc d_y");
    check_cuda(cudaMalloc(&d_th, bn * sizeof(float)), "malloc d_th");
    check_cuda(cudaMalloc(&d_L, sp.B * sizeof(float)), "malloc d_L");
    check_cuda(cudaMalloc(&d_ox, bn * sizeof(float)), "malloc d_ox");
    check_cuda(cudaMalloc(&d_oy, bn * sizeof(float)), "malloc d_oy");
    check_cuda(cudaMalloc(&d_oth, bn * sizeof(float)), "malloc d_oth");
    check_cuda(cudaMalloc(&d_ox0, sp.B * sizeof(float)), "malloc d_ox0");
    check_cuda(cudaMalloc(&d_oy0, sp.B * sizeof(float)), "malloc d_oy0");
    check_cuda(cudaMalloc(&d_oL, sp.B * sizeof(float)), "malloc d_oL");

    cudaMemcpy(d_x, h_x, bn * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, bn * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_th, h_th, bn * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_L, h_L, sp.B * sizeof(float), cudaMemcpyHostToDevice);
    if (h_mob) {
        check_cuda(cudaMalloc(&d_mob, bn * sizeof(uint8_t)), "malloc d_mob");
        cudaMemcpy(d_mob, h_mob, bn * sizeof(uint8_t), cudaMemcpyHostToDevice);
    }

    solve_batch_fp32<<<sp.B, BLOCK_THREADS>>>(
        sp.B, sp.N, sp.use_elastic_L, sp.fast_mode, sp.seed,
        d_x, d_y, d_th, d_L, d_mob, d_ox, d_oy, d_oth, d_ox0, d_oy0, d_oL
    );
    check_cuda(cudaDeviceSynchronize(), "solve_batch_fp32");

    double *d_px, *d_py, *d_pth, *d_Lc, *d_lam, *d_pen;
    check_cuda(cudaMalloc(&d_px, bn * sizeof(double)), "malloc d_px");
    check_cuda(cudaMalloc(&d_py, bn * sizeof(double)), "malloc d_py");
    check_cuda(cudaMalloc(&d_pth, bn * sizeof(double)), "malloc d_pth");
    check_cuda(cudaMalloc(&d_Lc, sp.B * sizeof(double)), "malloc d_Lc");
    check_cuda(cudaMalloc(&d_lam, sp.B * sizeof(double)), "malloc d_lam");
    check_cuda(cudaMalloc(&d_pen, sp.B * sizeof(double)), "malloc d_pen");

    int polish_steps = (sp.fast_mode >= 4) ? 1100 : (sp.fast_mode ? 800 : 1800);
    polish_and_cert_fp64<<<sp.B, BLOCK_THREADS>>>(
        sp.B, sp.N, polish_steps, sp.use_elastic_L, sp.fast_mode,
        d_ox, d_oy, d_oth, d_ox0, d_oy0, d_oL,
        d_px, d_py, d_pth, d_Lc, d_lam, d_pen
    );
    check_cuda(cudaDeviceSynchronize(), "polish_and_cert_fp64");

    double *ho_x = (double*)malloc(bn * sizeof(double));
    double *ho_y = (double*)malloc(bn * sizeof(double));
    double *ho_th = (double*)malloc(bn * sizeof(double));
    double *ho_Lc = (double*)malloc(sp.B * sizeof(double));
    double *ho_lam = (double*)malloc(sp.B * sizeof(double));
    double *ho_pen = (double*)malloc(sp.B * sizeof(double));

    cudaMemcpy(ho_x, d_px, bn * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(ho_y, d_py, bn * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(ho_th, d_pth, bn * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(ho_Lc, d_Lc, sp.B * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(ho_lam, d_lam, sp.B * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(ho_pen, d_pen, sp.B * sizeof(double), cudaMemcpyDeviceToHost);

    FILE* fout = fopen(out_path, "wb");
    if (!fout) { perror("fopen out"); return 1; }
    fwrite(ho_x, sizeof(double), bn, fout);
    fwrite(ho_y, sizeof(double), bn, fout);
    fwrite(ho_th, sizeof(double), bn, fout);
    fwrite(ho_Lc, sizeof(double), sp.B, fout);
    fwrite(ho_lam, sizeof(double), sp.B, fout);
    fwrite(ho_pen, sizeof(double), sp.B, fout);
    fclose(fout);

    free(h_x); free(h_y); free(h_th); free(h_L);
    if (h_mob) free(h_mob);
    if (d_mob) cudaFree(d_mob);
    free(ho_x); free(ho_y); free(ho_th); free(ho_Lc); free(ho_lam); free(ho_pen);
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_th); cudaFree(d_L);
    cudaFree(d_ox); cudaFree(d_oy); cudaFree(d_oth); cudaFree(d_ox0); cudaFree(d_oy0); cudaFree(d_oL);
    cudaFree(d_px); cudaFree(d_py); cudaFree(d_pth); cudaFree(d_Lc); cudaFree(d_lam); cudaFree(d_pen);
    return 0;
}
