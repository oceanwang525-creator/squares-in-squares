// Fixed-crystal GPU Sub-Pocket Inflation kernel (RTX 4070 SUPER, sm_89).
// Freezes squares [N_mob .. N-1] at fixed L_fixed[b] and runs multi-reheat
// SAT inflation (a: 0.72 -> 1.00) + void-jump/angle-snap + float64 polish
// ONLY on mobile squares [0 .. N_mob-1].
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#define MAX_N 1024
#define BLOCK_THREADS 256
#define GRID_G 22
#define NUM_CELLS (GRID_G * GRID_G)

struct SubParams {
    int32_t B;
    int32_t N;
    int32_t N_mob;
    int32_t mode;
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
    if (dx * dx + dy * dy >= 16.5f * half_a * half_a) return 0.0f;
    float cr = ci * cj + si * sj;
    float sr = si * cj - ci * sj;
    float rad = half_a * (1.0f + fabsf(cr) + fabsf(sr));
    float p0 = fabsf(dx * ci + dy * si);
    float p1 = fabsf(-dx * si + dy * ci);
    float p2 = fabsf(dx * cj + dy * sj);
    float p3 = fabsf(-dx * sj + dy * cj);
    return fmaxf(0.0f, rad - fmaxf(fmaxf(p0, p1), fmaxf(p2, p3)));
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
    if (dx * dx + dy * dy >= 17.0f * half_a * half_a) {
        pen_out = 0.0f; gx_out = 0.0f; gy_out = 0.0f; gth_out = 0.0f;
        return;
    }
    float cr = ci * cj + si * sj;
    float sr = si * cj - ci * sj;
    float rad = half_a * (1.0f + fabsf(cr) + fabsf(sr));
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

    float z_max = z0, p_star = p0, ux = ci, uy = si, is_i = 1.0f;
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
    float dpen_dth = drad - sp * (dy * ux - dx * uy) * is_i;
    float w = 2.0f * pen;
    gx_out = w * sp * ux;
    gy_out = w * sp * uy;
    gth_out = w * dpen_dth;
}

__device__ float reduce_sum_f32(float v, float* buf) {
    int tid = threadIdx.x;
    buf[tid] = v;
    __syncthreads();
    for (int s = BLOCK_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) buf[tid] += buf[tid + s];
        __syncthreads();
    }
    float out = buf[0];
    __syncthreads();
    return out;
}

__device__ double reduce_max_f64(double v, double* buf) {
    int tid = threadIdx.x;
    buf[tid] = v;
    __syncthreads();
    for (int s = BLOCK_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) buf[tid] = fmax(buf[tid], buf[tid + s]);
        __syncthreads();
    }
    double out = buf[0];
    __syncthreads();
    return out;
}

__device__ double reduce_sum_f64(double v, double* buf) {
    int tid = threadIdx.x;
    buf[tid] = v;
    __syncthreads();
    for (int s = BLOCK_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) buf[tid] += buf[tid + s];
        __syncthreads();
    }
    double out = buf[0];
    __syncthreads();
    return out;
}

__device__ void rebuild_grid(
    int N, float inv_cell,
    const float* sh_x, const float* sh_y,
    int32_t* cell_head, int32_t* sq_next
) {
    int tid = threadIdx.x;
    for (int c = tid; c < NUM_CELLS; c += BLOCK_THREADS) cell_head[c] = -1;
    __syncthreads();
    for (int i = tid; i < N; i += BLOCK_THREADS) {
        int cx = max(0, min(GRID_G - 1, (int)floorf(sh_x[i] * inv_cell)));
        int cy = max(0, min(GRID_G - 1, (int)floorf(sh_y[i] * inv_cell)));
        int cid = cy * GRID_G + cx;
        sq_next[i] = atomicExch(&cell_head[cid], i);
    }
    __syncthreads();
}

__device__ void void_relocate_sub(
    int N, int N_mob, float L, float inv_cell, float half_a,
    float mob_xmin, float mob_xmax, float mob_ymin, float mob_ymax,
    float* sh_x, float* sh_y, float* sh_th, float* sh_c, float* sh_s,
    const int32_t* cell_head, const int32_t* sq_next,
    float* red_f, int32_t* red_i, uint32_t& rng
) {
    int tid = threadIdx.x;
    float worst_val = 0.0f;
    int worst_idx = -1;
    for (int i = tid; i < N_mob; i += BLOCK_THREADS) {
        float xi = sh_x[i], yi = sh_y[i], ci = sh_c[i], si = sh_s[i];
        float ext = half_a * (fabsf(ci) + fabsf(si));
        float pen = fmaxf(0.0f, ext - xi) + fmaxf(0.0f, xi + ext - L) +
                    fmaxf(0.0f, ext - yi) + fmaxf(0.0f, yi + ext - L);
        int cx = max(0, min(GRID_G - 1, (int)floorf(xi * inv_cell)));
        int cy = max(0, min(GRID_G - 1, (int)floorf(yi * inv_cell)));
        for (int dy = -1; dy <= 1; ++dy) {
            int ny = cy + dy; if (ny < 0 || ny >= GRID_G) continue;
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = cx + dx; if (nx < 0 || nx >= GRID_G) continue;
                int j = cell_head[ny * GRID_G + nx];
                while (j >= 0) {
                    if (j != i) pen = fmaxf(pen, pair_pen_f32(xi, yi, ci, si, sh_x[j], sh_y[j], sh_c[j], sh_s[j], half_a));
                    j = sq_next[j];
                }
            }
        }
        if (pen > worst_val) { worst_val = pen; worst_idx = i; }
    }
    red_f[tid] = worst_val;
    red_i[tid] = worst_idx;
    __syncthreads();
    for (int s = BLOCK_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s && red_f[tid + s] > red_f[tid]) {
            red_f[tid] = red_f[tid + s];
            red_i[tid] = red_i[tid + s];
        }
        __syncthreads();
    }
    int wi = red_i[0];
    float cur_worst = red_f[0];
    if (wi < 0 || cur_worst < 0.015f) return;

    const float cand_angs[9] = {
        0.0f, 0.78539816f, -0.46364761f, 0.46364761f,
        -0.64350111f, 0.64350111f, -0.39479112f, -0.490037f, sh_th[wi]
    };

    float best_p = cur_worst * 0.90f;
    float bx = sh_x[wi], by = sh_y[wi], bth = sh_th[wi];
    for (int p = 0; p < 4; ++p) {
        float px, py;
        if ((tid & 3) == 0) {
            // Focus near mobile bounding box + walls
            px = fminf(fmaxf(mob_xmin - 0.5f + rand_u01(rng) * (mob_xmax - mob_xmin + 1.0f), half_a), L - half_a);
            py = fminf(fmaxf(mob_ymin - 0.5f + rand_u01(rng) * (mob_ymax - mob_ymin + 1.0f), half_a), L - half_a);
        } else {
            px = half_a + rand_u01(rng) * (L - 2.0f * half_a);
            py = half_a + rand_u01(rng) * (L - 2.0f * half_a);
        }
        int cx = max(0, min(GRID_G - 1, (int)floorf(px * inv_cell)));
        int cy = max(0, min(GRID_G - 1, (int)floorf(py * inv_cell)));
        for (int ak = 0; ak < 9; ++ak) {
            float ta = cand_angs[ak];
            float tc = cosf(ta), ts = sinf(ta);
            float ext = half_a * (fabsf(tc) + fabsf(ts));
            float mp = fmaxf(0.0f, ext - px) + fmaxf(0.0f, px + ext - L) +
                       fmaxf(0.0f, ext - py) + fmaxf(0.0f, py + ext - L);
            if (mp >= best_p) continue;
            for (int dy = -1; dy <= 1; ++dy) {
                int ny = cy + dy; if (ny < 0 || ny >= GRID_G) continue;
                for (int dx = -1; dx <= 1; ++dx) {
                    int nx = cx + dx; if (nx < 0 || nx >= GRID_G) continue;
                    int j = cell_head[ny * GRID_G + nx];
                    while (j >= 0) {
                        if (j != wi) {
                            float pj = pair_pen_f32(px, py, tc, ts, sh_x[j], sh_y[j], sh_c[j], sh_s[j], half_a);
                            if (pj > mp) mp = pj;
                            if (mp >= best_p) break;
                        }
                        j = sq_next[j];
                    }
                    if (mp >= best_p) break;
                }
                if (mp >= best_p) break;
            }
            if (mp < best_p) { best_p = mp; bx = px; by = py; bth = ta; }
        }
    }
    red_f[tid] = best_p;
    red_i[tid] = tid;
    __syncthreads();
    for (int s = BLOCK_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s && red_f[tid + s] < red_f[tid]) {
            red_f[tid] = red_f[tid + s];
            red_i[tid] = red_i[tid + s];
        }
        __syncthreads();
    }
    int win_t = red_i[0];
    if (red_f[0] < cur_worst * 0.86f) {
        if (tid == win_t) {
            sh_x[wi] = bx; sh_y[wi] = by; sh_th[wi] = bth;
            sh_c[wi] = cosf(bth); sh_s[wi] = sinf(bth);
        }
        __syncthreads();
    }
}

__global__ void subpocket_kernel(
    const float* __restrict__ in_x,
    const float* __restrict__ in_y,
    const float* __restrict__ in_th,
    const float* __restrict__ in_L,
    double* __restrict__ out_x,
    double* __restrict__ out_y,
    double* __restrict__ out_th,
    double* __restrict__ out_pen,
    double* __restrict__ out_lam,
    SubParams sp
) {
    int b = blockIdx.x;
    if (b >= sp.B) return;
    int tid = threadIdx.x;
    int N = sp.N;
    int N_mob = sp.N_mob;

    __shared__ float sh_x[MAX_N], sh_y[MAX_N], sh_th[MAX_N], sh_c[MAX_N], sh_s[MAX_N];
    __shared__ float sh_vx[MAX_N], sh_vy[MAX_N], sh_vth[MAX_N];
    __shared__ float sh_bx[MAX_N], sh_by[MAX_N], sh_bth[MAX_N];
    __shared__ int32_t cell_head[NUM_CELLS], sq_next[MAX_N];
    __shared__ float red_f[BLOCK_THREADS];
    __shared__ int32_t red_i[BLOCK_THREADS];
    __shared__ double red_d[BLOCK_THREADS];

    size_t base = (size_t)b * (size_t)N;
    for (int i = tid; i < N; i += BLOCK_THREADS) {
        float x = in_x[base + i], y = in_y[base + i], t = in_th[base + i];
        sh_x[i] = x; sh_y[i] = y; sh_th[i] = t;
        sh_c[i] = cosf(t); sh_s[i] = sinf(t);
        sh_vx[i] = 0.0f; sh_vy[i] = 0.0f; sh_vth[i] = 0.0f;
        sh_bx[i] = x; sh_by[i] = y; sh_bth[i] = t;
    }
    __syncthreads();

    float L = in_L[b];
    float cell_sz = fmaxf(1.48f, L / (float)GRID_G + 1e-4f);
    float inv_cell = 1.0f / cell_sz;
    uint32_t rng = (sp.seed ^ ((uint32_t)(b + 1) * 0x9e3779b9u) ^ ((uint32_t)(tid + 1) * 0x85ebca6bu));
    if (rng == 0) rng = 1234567u;

    // Bounding box of initial mobile pocket for focused void probing
    float lmin_x = L, lmax_x = 0.0f, lmin_y = L, lmax_y = 0.0f;
    for (int i = tid; i < N_mob; i += BLOCK_THREADS) {
        lmin_x = fminf(lmin_x, sh_x[i]); lmax_x = fmaxf(lmax_x, sh_x[i]);
        lmin_y = fminf(lmin_y, sh_y[i]); lmax_y = fmaxf(lmax_y, sh_y[i]);
    }
    red_f[tid] = lmin_x; __syncthreads();
    for (int s = BLOCK_THREADS/2; s>0; s>>=1) { if (tid<s) red_f[tid] = fminf(red_f[tid], red_f[tid+s]); __syncthreads(); }
    float mob_xmin = red_f[0];
    red_f[tid] = lmax_x; __syncthreads();
    for (int s = BLOCK_THREADS/2; s>0; s>>=1) { if (tid<s) red_f[tid] = fmaxf(red_f[tid], red_f[tid+s]); __syncthreads(); }
    float mob_xmax = red_f[0];
    red_f[tid] = lmin_y; __syncthreads();
    for (int s = BLOCK_THREADS/2; s>0; s>>=1) { if (tid<s) red_f[tid] = fminf(red_f[tid], red_f[tid+s]); __syncthreads(); }
    float mob_ymin = red_f[0];
    red_f[tid] = lmax_y; __syncthreads();
    for (int s = BLOCK_THREADS/2; s>0; s>>=1) { if (tid<s) red_f[tid] = fmaxf(red_f[tid], red_f[tid+s]); __syncthreads(); }
    float mob_ymax = red_f[0];

    struct Stage { float a; int steps; float lr; float noise; float w_align; };
    const Stage stages_gentle[10] = {
        {0.960f, 260, 0.0045f, 0.0025f, 0.0f},
        {0.974f, 300, 0.0032f, 0.0015f, 0.0f},
        {0.985f, 320, 0.0022f, 0.0008f, 0.0f},
        {0.992f, 340, 0.0014f, 0.0002f, 0.0f},
        {0.996f, 340, 0.0009f, 0.0f,    0.0f},
        {0.9985f,340, 0.0005f, 0.0f,    0.0f},
        {0.9996f,340, 0.00028f,0.0f,    0.0f},
        {1.000f, 400, 0.00016f,0.0f,    0.0f},
        {1.000f, 360, 0.00008f,0.0f,    0.0f},
        {1.000f, 300, 0.00003f,0.0f,    0.0f}
    };
    const Stage stages_deep[10] = {
        {0.895f, 240, 0.014f, 0.010f, 0.015f},
        {0.935f, 280, 0.010f, 0.006f, 0.010f},
        {0.962f, 320, 0.0065f,0.003f, 0.005f},
        {0.982f, 340, 0.0040f,0.001f, 0.0f},
        {0.952f, 220, 0.0060f,0.002f, 0.0f}, // Mild reheat
        {0.985f, 340, 0.0032f,0.0005f,0.0f},
        {0.995f, 360, 0.0016f,0.0f,    0.0f},
        {0.9988f,340, 0.0007f,0.0f,    0.0f},
        {1.000f, 400, 0.0003f,0.0f,    0.0f},
        {1.000f, 360, 0.0001f,0.0f,    0.0f}
    };
    const Stage* stages = (sp.mode == 2) ? stages_deep : stages_gentle;

    float mx = 0.0f, my = 0.0f, mt = 0.0f;
    float vx = 0.0f, vy = 0.0f, vt = 0.0f;
    int step_g = 0;
    float best_final_pen = 1e18f;

    for (int st = 0; st < 10; ++st) {
        float scale_a = stages[st].a;
        float half_a = 0.5f * scale_a;
        int nsteps = (sp.mode == 1) ? max(60, stages[st].steps / 3) : stages[st].steps;
        float lr_base = stages[st].lr;
        float noise_base = stages[st].noise;
        float w_align = stages[st].w_align;
        float sg_eps = (scale_a < 0.99f) ? 1e-5f : 1e-7f;

        mx *= 0.4f; my *= 0.4f; mt *= 0.4f;
        __syncthreads();
        rebuild_grid(N, inv_cell, sh_x, sh_y, cell_head, sq_next);

        for (int s = 0; s < nsteps; ++s) {
            if (s > 0 && (s % 10 == 0)) rebuild_grid(N, inv_cell, sh_x, sh_y, cell_head, sq_next);
            if (scale_a <= 0.995f && (s == 30 || s == 110 || s == 200)) {
                void_relocate_sub(N, N_mob, L, inv_cell, half_a,
                                  mob_xmin, mob_xmax, mob_ymin, mob_ymax,
                                  sh_x, sh_y, sh_th, sh_c, sh_s, cell_head, sq_next,
                                  red_f, red_i, rng);
                rebuild_grid(N, inv_cell, sh_x, sh_y, cell_head, sq_next);
            }

            float frac = (float)s / (float)max(1, nsteps - 1);
            float lr_mult = (frac < 0.45f) ? 1.0f : 0.5f * (1.0f + cosf(3.14159265f * (frac - 0.45f) / 0.55f));
            float lr_xy = lr_base * (0.20f + 0.80f * lr_mult);
            float lr_th = lr_xy * ((scale_a < 0.97f) ? 1.35f : 0.70f);
            float n_cur = noise_base * (1.0f - 0.8f * frac);

            float step_pen = 0.0f;
            float fx_l = 0.0f, fy_l = 0.0f, fth_l = 0.0f;
            if (tid < N_mob) {
                int i = tid;
                float xi = sh_x[i], yi = sh_y[i], th = sh_th[i];
                float ci = sh_c[i], si = sh_s[i];
                float ext = half_a * (fabsf(ci) + fabsf(si));
                float eps2 = sg_eps * sg_eps;
                float s_c = ci * rsqrtf(ci * ci + eps2);
                float s_s = si * rsqrtf(si * si + eps2);
                float dext = half_a * (s_s * ci - s_c * si);

                float vw0 = fmaxf(0.0f, ext - xi);
                float vw1 = fmaxf(0.0f, xi + ext - L);
                float vw2 = fmaxf(0.0f, ext - yi);
                float vw3 = fmaxf(0.0f, yi + ext - L);
                step_pen += 5.0f * (vw0 * vw0 + vw1 * vw1 + vw2 * vw2 + vw3 * vw3);
                fx_l += 5.0f * (-2.0f * vw0 + 2.0f * vw1);
                fy_l += 5.0f * (-2.0f * vw2 + 2.0f * vw3);
                fth_l += 5.0f * 2.0f * (vw0 + vw1 + vw2 + vw3) * dext;

                int cx = max(0, min(GRID_G - 1, (int)floorf(xi * inv_cell)));
                int cy = max(0, min(GRID_G - 1, (int)floorf(yi * inv_cell)));
                for (int dy = -1; dy <= 1; ++dy) {
                    int ny = cy + dy; if (ny < 0 || ny >= GRID_G) continue;
                    for (int dx = -1; dx <= 1; ++dx) {
                        int nx = cx + dx; if (nx < 0 || nx >= GRID_G) continue;
                        int j = cell_head[ny * GRID_G + nx];
                        while (j >= 0) {
                            if (j != i) {
                                float pp, px, py, pt;
                                pair_pen_and_grad_f32(xi, yi, ci, si, sh_x[j], sh_y[j], sh_c[j], sh_s[j],
                                                      i, j, half_a, sg_eps, pp, px, py, pt);
                                float scale_f = (j >= N_mob) ? 2.0f : 1.0f;
                                if (pp > 0.0f) {
                                    step_pen += pp * pp * (j < N_mob ? 0.5f : 1.0f);
                                    fx_l += scale_f * px;
                                    fy_l += scale_f * py;
                                    fth_l += scale_f * pt;
                                }
                                if (w_align > 0.0f) {
                                    float ddx = sh_x[j] - xi, ddy = sh_y[j] - yi;
                                    if (ddx * ddx + ddy * ddy < 2.25f * scale_a * scale_a) {
                                        fth_l += w_align * 2.0f * sinf(4.0f * (th - sh_th[j]));
                                    }
                                }
                            }
                            j = sq_next[j];
                        }
                    }
                }
            }

            float tot_pen = reduce_sum_f32(step_pen, red_f);
            float g2_tot = reduce_sum_f32(fx_l * fx_l + fy_l * fy_l + fth_l * fth_l, red_f);
            float gscale = fminf(1.0f, 10.0f / sqrtf(g2_tot + 1e-12f));

            step_g += 1;
            float b1 = 0.85f, b2 = 0.99f;
            float bc1 = 1.0f - powf(b1, (float)min(step_g, 50));
            float bc2 = 1.0f - powf(b2, (float)min(step_g, 50));

            if (tid < N_mob) {
                int i = tid;
                float fx = fx_l * gscale;
                float fy = fy_l * gscale;
                float ft = fth_l * gscale;

                mx = b1 * mx + (1.0f - b1) * fx;
                vx = b2 * vx + (1.0f - b2) * fx * fx;
                my = b1 * my + (1.0f - b1) * fy;
                vy = b2 * vy + (1.0f - b2) * fy * fy;
                mt = b1 * mt + (1.0f - b1) * ft;
                vt = b2 * vt + (1.0f - b2) * ft * ft;

                float nx = sh_x[i] - lr_xy * (mx / bc1) / (sqrtf(vx / bc2) + 1e-8f);
                float ny = sh_y[i] - lr_xy * (my / bc1) / (sqrtf(vy / bc2) + 1e-8f);
                float nt = sh_th[i] - lr_th * (mt / bc1) / (sqrtf(vt / bc2) + 1e-8f);
                if (n_cur > 0.0f) {
                    nx += n_cur * rand_normal(rng);
                    ny += n_cur * rand_normal(rng);
                    nt += (n_cur * 1.4f) * rand_normal(rng);
                }
                nx = fminf(fmaxf(nx, 0.35f), L - 0.35f);
                ny = fminf(fmaxf(ny, 0.35f), L - 0.35f);
                sh_x[i] = nx; sh_y[i] = ny; sh_th[i] = nt;
                sh_c[i] = cosf(nt); sh_s[i] = sinf(nt);
            }
            __syncthreads();

            if (scale_a >= 0.9999f && ((s & 7) == 7 || s == nsteps - 1)) {
                if (tot_pen < best_final_pen) {
                    best_final_pen = tot_pen;
                    for (int i = tid; i < N_mob; i += BLOCK_THREADS) {
                        sh_bx[i] = sh_x[i]; sh_by[i] = sh_y[i]; sh_bth[i] = sh_th[i];
                    }
                }
                __syncthreads();
            }
        }
    }

    if (best_final_pen < 1e17f) {
        for (int i = tid; i < N_mob; i += BLOCK_THREADS) {
            sh_x[i] = sh_bx[i]; sh_y[i] = sh_by[i]; sh_th[i] = sh_bth[i];
            sh_c[i] = cosf(sh_th[i]); sh_s[i] = sinf(sh_th[i]);
        }
        __syncthreads();
    }

    // Compute float64 max penetration & required scale lambda on mobile + mobile-frozen pairs
    double ld = (double)L;
    double max_p = 0.0;
    double lam_lo = 0.90, lam_hi = 1.05;
    // Quick exact max violation at a=1.0
    for (int i = tid; i < N_mob; i += BLOCK_THREADS) {
        double xi = (double)sh_x[i], yi = (double)sh_y[i], ti = (double)sh_th[i];
        double ci = cos(ti), si = sin(ti);
        double ext = 0.5 * (fabs(ci) + fabs(si));
        max_p = fmax(max_p, ext - xi);
        max_p = fmax(max_p, xi + ext - ld);
        max_p = fmax(max_p, ext - yi);
        max_p = fmax(max_p, yi + ext - ld);
        for (int j = 0; j < N; ++j) {
            if (j == i) continue;
            double dx = (double)sh_x[j] - xi, dy = (double)sh_y[j] - yi;
            if (dx * dx + dy * dy >= 4.2) continue;
            double tj = (double)sh_th[j], cj = cos(tj), sj = sin(tj);
            double rad = 0.5 * (1.0 + fabs(ci * cj + si * sj) + fabs(si * cj - ci * sj));
            double p0 = fabs(dx * ci + dy * si), p1 = fabs(-dx * si + dy * ci);
            double p2 = fabs(dx * cj + dy * sj), p3 = fabs(-dx * sj + dy * cj);
            double pmax = fmax(fmax(p0, p1), fmax(p2, p3));
            max_p = fmax(max_p, rad - pmax);
        }
    }
    double g_max_p = reduce_max_f64(max_p, red_d);

    // Binary search max feasible a <= 1.05 in float64
    for (int it = 0; it < 28; ++it) {
        double mid = 0.5 * (lam_lo + lam_hi);
        double bad = 0.0;
        for (int i = tid; i < N_mob; i += BLOCK_THREADS) {
            double xi = (double)sh_x[i], yi = (double)sh_y[i], ti = (double)sh_th[i];
            double ci = cos(ti), si = sin(ti);
            double ext = 0.5 * mid * (fabs(ci) + fabs(si));
            if (ext - xi > 0 || xi + ext - ld > 0 || ext - yi > 0 || yi + ext - ld > 0) bad = 1.0;
            for (int j = 0; j < N; ++j) {
                if (j == i) continue;
                double dx = (double)sh_x[j] - xi, dy = (double)sh_y[j] - yi;
                if (dx * dx + dy * dy >= 4.2 * mid * mid) continue;
                double tj = (double)sh_th[j], cj = cos(tj), sj = sin(tj);
                double rad = 0.5 * mid * (1.0 + fabs(ci * cj + si * sj) + fabs(si * cj - ci * sj));
                double p0 = fabs(dx * ci + dy * si), p1 = fabs(-dx * si + dy * ci);
                double p2 = fabs(dx * cj + dy * sj), p3 = fabs(-dx * sj + dy * cj);
                if (rad - fmax(fmax(p0, p1), fmax(p2, p3)) > 1e-11) bad = 1.0;
            }
        }
        double any_bad = reduce_max_f64(bad, red_d);
        if (any_bad > 0.5) lam_hi = mid; else lam_lo = mid;
    }
    double req_lam = 1.0 / fmax(0.5, lam_lo);

    for (int i = tid; i < N; i += BLOCK_THREADS) {
        out_x[base + i] = (double)sh_x[i];
        out_y[base + i] = (double)sh_y[i];
        out_th[base + i] = (double)sh_th[i];
    }
    if (tid == 0) {
        out_pen[b] = g_max_p;
        out_lam[b] = req_lam;
    }
}

int main(int argc, char** argv) {
    const char* in_path = nullptr;
    const char* out_path = nullptr;
    for (int i = 1; i + 1 < argc; ++i) {
        if (strcmp(argv[i], "--in") == 0) in_path = argv[++i];
        else if (strcmp(argv[i], "--out") == 0) out_path = argv[++i];
    }
    if (!in_path || !out_path) return 1;
    FILE* fi = fopen(in_path, "rb");
    if (!fi) return 2;
    SubParams sp{};
    if (fread(&sp, sizeof(sp), 1, fi) != 1) { fclose(fi); return 3; }
    if (sp.N > MAX_N || sp.N_mob > sp.N) { fclose(fi); return 4; }
    size_t bn = (size_t)sp.B * (size_t)sp.N;
    std::vector<float> hx(bn), hy(bn), hth(bn), hL(sp.B);
    if (fread(hx.data(), sizeof(float), bn, fi) != bn) { fclose(fi); return 5; }
    if (fread(hy.data(), sizeof(float), bn, fi) != bn) { fclose(fi); return 6; }
    if (fread(hth.data(), sizeof(float), bn, fi) != bn) { fclose(fi); return 7; }
    if (fread(hL.data(), sizeof(float), sp.B, fi) != (size_t)sp.B) { fclose(fi); return 8; }
    fclose(fi);

    float *dx = nullptr, *dy = nullptr, *dth = nullptr, *dL = nullptr;
    double *ox_d = nullptr, *oy_d = nullptr, *oth_d = nullptr, *open_d = nullptr, *olam_d = nullptr;
    cudaMalloc(&dx, bn * sizeof(float));
    cudaMalloc(&dy, bn * sizeof(float));
    cudaMalloc(&dth, bn * sizeof(float));
    cudaMalloc(&dL, sp.B * sizeof(float));
    cudaMalloc(&ox_d, bn * sizeof(double));
    cudaMalloc(&oy_d, bn * sizeof(double));
    cudaMalloc(&oth_d, bn * sizeof(double));
    cudaMalloc(&open_d, sp.B * sizeof(double));
    cudaMalloc(&olam_d, sp.B * sizeof(double));

    cudaMemcpy(dx, hx.data(), bn * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy.data(), bn * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dth, hth.data(), bn * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dL, hL.data(), sp.B * sizeof(float), cudaMemcpyHostToDevice);

    subpocket_kernel<<<sp.B, BLOCK_THREADS>>>(dx, dy, dth, dL, ox_d, oy_d, oth_d, open_d, olam_d, sp);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 9;
    }

    std::vector<double> ox(bn), oy(bn), oth(bn), open(sp.B), olam(sp.B);
    cudaMemcpy(ox.data(), ox_d, bn * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(oy.data(), oy_d, bn * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(oth.data(), oth_d, bn * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(open.data(), open_d, sp.B * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(olam.data(), olam_d, sp.B * sizeof(double), cudaMemcpyDeviceToHost);

    FILE* fo = fopen(out_path, "wb");
    if (!fo) return 10;
    fwrite(ox.data(), sizeof(double), bn, fo);
    fwrite(oy.data(), sizeof(double), bn, fo);
    fwrite(oth.data(), sizeof(double), bn, fo);
    fwrite(open.data(), sizeof(double), sp.B, fo);
    fwrite(olam.data(), sizeof(double), sp.B, fo);
    fclose(fo);

    cudaFree(dx); cudaFree(dy); cudaFree(dth); cudaFree(dL);
    cudaFree(ox_d); cudaFree(oy_d); cudaFree(oth_d); cudaFree(open_d); cudaFree(olam_d);
    return 0;
}
