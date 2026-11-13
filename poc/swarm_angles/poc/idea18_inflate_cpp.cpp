// Idea 18: Fast C++ Spatial-Hash SAT Energy/Gradient, L-BFGS, Soft-Core Inflation & Basin-Hopping.
// Compiled into /tmp/libidea18.so (CPU only, std::thread worker pool capped <= 7 threads).

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <thread>
#include <vector>

namespace {

struct GridWorkspace {
    std::vector<double> c, s, hx, dhx;
    std::vector<int> gx, gy, head, nxt;

    void resize(int n) {
        c.resize(n);
        s.resize(n);
        hx.resize(n);
        dhx.resize(n);
        gx.resize(n);
        gy.resize(n);
        nxt.resize(n);
    }
};

struct LbfgsWorkspace {
    int n = 0;
    int dim = 0;
    static constexpr int M = 10;
    std::vector<double> z_new, g, g_new, q, d;
    std::vector<double> s_hist, y_hist, rho, alpha_buf;
    GridWorkspace grid;

    void init(int n_in) {
        if (n == n_in) return;
        n = n_in;
        dim = 3 * n;
        z_new.resize(dim);
        g.resize(dim);
        g_new.resize(dim);
        q.resize(dim);
        d.resize(dim);
        s_hist.assign(M * dim, 0.0);
        y_hist.assign(M * dim, 0.0);
        rho.assign(M, 0.0);
        alpha_buf.assign(M, 0.0);
        grid.resize(n);
    }
};

inline double compute_sat_bbox_and_pen(
    int n, const double* x, const double* y, const double* th,
    double side, GridWorkspace& gw, double* out_L
) {
    gw.resize(n);
    const double half = 0.5 * side;
    double minx = 1e18, maxx = -1e18, miny = 1e18, maxy = -1e18;
    double cxmin = 1e18, cxmax = -1e18, cymin = 1e18, cymax = -1e18;

    for (int i = 0; i < n; ++i) {
        const double ci = std::cos(th[i]);
        const double si = std::sin(th[i]);
        const double hxi = half * (std::fabs(ci) + std::fabs(si));
        gw.c[i] = ci;
        gw.s[i] = si;
        gw.hx[i] = hxi;
        minx = std::min(minx, x[i] - hxi);
        maxx = std::max(maxx, x[i] + hxi);
        miny = std::min(miny, y[i] - hxi);
        maxy = std::max(maxy, y[i] + hxi);
        cxmin = std::min(cxmin, x[i]);
        cxmax = std::max(cxmax, x[i]);
        cymin = std::min(cymin, y[i]);
        cymax = std::max(cymax, y[i]);
    }
    if (out_L) {
        *out_L = std::max(maxx - minx, maxy - miny);
    }

    const double cell = 1.42 * side + 1e-6;
    const int gx_dim = static_cast<int>(std::floor((cxmax - cxmin) / cell)) + 1;
    const int gy_dim = static_cast<int>(std::floor((cymax - cymin) / cell)) + 1;

    double max_pen = 0.0;
    auto check_pair = [&](int i, int j) {
        const double dx = x[j] - x[i];
        const double hsum = gw.hx[i] + gw.hx[j];
        if (std::fabs(dx) >= hsum) return;
        const double dy = y[j] - y[i];
        if (std::fabs(dy) >= hsum) return;

        const double ci = gw.c[i], si = gw.s[i];
        const double cj = gw.c[j], sj = gw.s[j];
        const double rad = half * (1.0 + std::fabs(ci * cj + si * sj) + std::fabs(ci * sj - si * cj));

        const double pen1 = rad - std::fabs(dx * ci + dy * si);
        if (pen1 <= 0.0) return;
        const double pen2 = rad - std::fabs(-dx * si + dy * ci);
        if (pen2 <= 0.0) return;
        const double pen3 = rad - std::fabs(dx * cj + dy * sj);
        if (pen3 <= 0.0) return;
        const double pen4 = rad - std::fabs(-dx * sj + dy * cj);
        if (pen4 <= 0.0) return;

        const double pen = std::min(std::min(pen1, pen2), std::min(pen3, pen4));
        if (pen > max_pen) max_pen = pen;
    };

    if (gx_dim > 0 && gy_dim > 0 && gx_dim <= 512 && gy_dim <= 512) {
        const int n_cells = gx_dim * gy_dim;
        if (static_cast<int>(gw.head.size()) < n_cells) {
            gw.head.resize(n_cells);
        }
        std::fill_n(gw.head.data(), n_cells, -1);
        for (int i = 0; i < n; ++i) {
            int gxi = static_cast<int>(std::floor((x[i] - cxmin) / cell));
            int gyi = static_cast<int>(std::floor((y[i] - cymin) / cell));
            if (gxi < 0) gxi = 0;
            if (gxi >= gx_dim) gxi = gx_dim - 1;
            if (gyi < 0) gyi = 0;
            if (gyi >= gy_dim) gyi = gy_dim - 1;
            gw.gx[i] = gxi;
            gw.gy[i] = gyi;
            const int cid = gxi * gy_dim + gyi;
            gw.nxt[i] = gw.head[cid];
            gw.head[cid] = i;
        }
        for (int i = 0; i < n; ++i) {
            const int x0 = std::max(0, gw.gx[i] - 1);
            const int x1 = std::min(gx_dim - 1, gw.gx[i] + 1);
            const int y0 = std::max(0, gw.gy[i] - 1);
            const int y1 = std::min(gy_dim - 1, gw.gy[i] + 1);
            for (int cx = x0; cx <= x1; ++cx) {
                const int row_base = cx * gy_dim;
                for (int cy = y0; cy <= y1; ++cy) {
                    int j = gw.head[row_base + cy];
                    while (j != -1) {
                        if (j > i) check_pair(i, j);
                        j = gw.nxt[j];
                    }
                }
            }
        }
    } else {
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                check_pair(i, j);
            }
        }
    }
    return max_pen;
}

inline double compute_energy_grad(
    int n, const double* z, double L, double w_wall, double w_pair,
    double side, GridWorkspace& gw, double* grad, double* per_sq_pen = nullptr,
    const double* th_ref = nullptr, double w_ang = 0.0
) {
    gw.resize(n);
    const double* x = z;
    const double* y = z + n;
    const double* th = z + 2 * n;
    std::fill_n(grad, 3 * n, 0.0);
    if (per_sq_pen) std::fill_n(per_sq_pen, n, 0.0);

    double* g_x = grad;
    double* g_y = grad + n;
    double* g_th = grad + 2 * n;

    const double half = 0.5 * side;
    double energy = 0.0;
    double cxmin = 1e18, cxmax = -1e18, cymin = 1e18, cymax = -1e18;

    for (int i = 0; i < n; ++i) {
        const double ci = std::cos(th[i]);
        const double si = std::sin(th[i]);
        const double hxi = half * (std::fabs(ci) + std::fabs(si));
        const double sc_sgn = (ci >= 0.0) ? 1.0 : -1.0;
        const double ss_sgn = (si >= 0.0) ? 1.0 : -1.0;
        const double dhxi = half * (-sc_sgn * si + ss_sgn * ci);
        gw.c[i] = ci;
        gw.s[i] = si;
        gw.hx[i] = hxi;
        gw.dhx[i] = dhxi;

        cxmin = std::min(cxmin, x[i]);
        cxmax = std::max(cxmax, x[i]);
        cymin = std::min(cymin, y[i]);
        cymax = std::max(cymax, y[i]);

        double v = hxi - x[i];
        if (v > 0.0) {
            energy += w_wall * v * v;
            const double g = 2.0 * w_wall * v;
            g_x[i] -= g;
            g_th[i] += g * dhxi;
            if (per_sq_pen) per_sq_pen[i] += v;
        }
        v = x[i] + hxi - L;
        if (v > 0.0) {
            energy += w_wall * v * v;
            const double g = 2.0 * w_wall * v;
            g_x[i] += g;
            g_th[i] += g * dhxi;
            if (per_sq_pen) per_sq_pen[i] += v;
        }
        v = hxi - y[i];
        if (v > 0.0) {
            energy += w_wall * v * v;
            const double g = 2.0 * w_wall * v;
            g_y[i] -= g;
            g_th[i] += g * dhxi;
            if (per_sq_pen) per_sq_pen[i] += v;
        }
        v = y[i] + hxi - L;
        if (v > 0.0) {
            energy += w_wall * v * v;
            const double g = 2.0 * w_wall * v;
            g_y[i] += g;
            g_th[i] += g * dhxi;
            if (per_sq_pen) per_sq_pen[i] += v;
        }
        if (w_ang > 0.0 && th_ref) {
            const double dth = th[i] - th_ref[i];
            energy += w_ang * dth * dth;
            g_th[i] += 2.0 * w_ang * dth;
        }
    }

    auto accum_pair = [&](int i, int j) {
        const double dx = x[j] - x[i];
        const double hsum = gw.hx[i] + gw.hx[j];
        if (std::fabs(dx) >= hsum) return;
        const double dy = y[j] - y[i];
        if (std::fabs(dy) >= hsum) return;

        const double ci = gw.c[i], si = gw.s[i];
        const double cj = gw.c[j], sj = gw.s[j];
        const double cr = ci * cj + si * sj;
        const double sr = ci * sj - si * cj;
        const double rad = half * (1.0 + std::fabs(cr) + std::fabs(sr));

        const double p1 = dx * ci + dy * si;
        const double pen1 = rad - std::fabs(p1);
        if (pen1 <= 0.0) return;

        const double p2 = -dx * si + dy * ci;
        const double pen2 = rad - std::fabs(p2);
        if (pen2 <= 0.0) return;

        const double p3 = dx * cj + dy * sj;
        const double pen3 = rad - std::fabs(p3);
        if (pen3 <= 0.0) return;

        const double p4 = -dx * sj + dy * cj;
        const double pen4 = rad - std::fabs(p4);
        if (pen4 <= 0.0) return;

        int best_k = 0;
        double pen = pen1;
        if (pen2 < pen) { pen = pen2; best_k = 1; }
        if (pen3 < pen) { pen = pen3; best_k = 2; }
        if (pen4 < pen) { pen = pen4; best_k = 3; }

        energy += w_pair * pen * pen;
        if (per_sq_pen) {
            per_sq_pen[i] += pen;
            per_sq_pen[j] += pen;
        }
        const double g = 2.0 * w_pair * pen;
        const double sgn_cr = (cr >= 0.0) ? 1.0 : -1.0;
        const double sgn_sr = (sr >= 0.0) ? 1.0 : -1.0;
        const double drad_dr = half * (-sgn_cr * sr + sgn_sr * cr);

        if (best_k == 0) {
            const double sp = (p1 >= 0.0) ? 1.0 : -1.0;
            const double dpx = -sp * ci;
            const double dpy = -sp * si;
            g_x[i] -= g * dpx;
            g_x[j] += g * dpx;
            g_y[i] -= g * dpy;
            g_y[j] += g * dpy;
            g_th[i] += g * (-drad_dr - sp * p2);
            g_th[j] += g * drad_dr;
        } else if (best_k == 1) {
            const double sp = (p2 >= 0.0) ? 1.0 : -1.0;
            const double dpx = sp * si;
            const double dpy = -sp * ci;
            g_x[i] -= g * dpx;
            g_x[j] += g * dpx;
            g_y[i] -= g * dpy;
            g_y[j] += g * dpy;
            g_th[i] += g * (-drad_dr + sp * p1);
            g_th[j] += g * drad_dr;
        } else if (best_k == 2) {
            const double sp = (p3 >= 0.0) ? 1.0 : -1.0;
            const double dpx = -sp * cj;
            const double dpy = -sp * sj;
            g_x[i] -= g * dpx;
            g_x[j] += g * dpx;
            g_y[i] -= g * dpy;
            g_y[j] += g * dpy;
            g_th[i] += g * (-drad_dr);
            g_th[j] += g * (drad_dr - sp * p4);
        } else {
            const double sp = (p4 >= 0.0) ? 1.0 : -1.0;
            const double dpx = sp * sj;
            const double dpy = -sp * cj;
            g_x[i] -= g * dpx;
            g_x[j] += g * dpx;
            g_y[i] -= g * dpy;
            g_y[j] += g * dpy;
            g_th[i] += g * (-drad_dr);
            g_th[j] += g * (drad_dr + sp * p3);
        }
    };

    const double cell = 1.42 * side + 1e-6;
    const int gx_dim = static_cast<int>(std::floor((cxmax - cxmin) / cell)) + 1;
    const int gy_dim = static_cast<int>(std::floor((cymax - cymin) / cell)) + 1;

    if (gx_dim > 0 && gy_dim > 0 && gx_dim <= 512 && gy_dim <= 512) {
        const int n_cells = gx_dim * gy_dim;
        if (static_cast<int>(gw.head.size()) < n_cells) {
            gw.head.resize(n_cells);
        }
        std::fill_n(gw.head.data(), n_cells, -1);
        for (int i = 0; i < n; ++i) {
            int gxi = static_cast<int>(std::floor((x[i] - cxmin) / cell));
            int gyi = static_cast<int>(std::floor((y[i] - cymin) / cell));
            if (gxi < 0) gxi = 0;
            if (gxi >= gx_dim) gxi = gx_dim - 1;
            if (gyi < 0) gyi = 0;
            if (gyi >= gy_dim) gyi = gy_dim - 1;
            gw.gx[i] = gxi;
            gw.gy[i] = gyi;
            const int cid = gxi * gy_dim + gyi;
            gw.nxt[i] = gw.head[cid];
            gw.head[cid] = i;
        }
        for (int i = 0; i < n; ++i) {
            const int x0 = std::max(0, gw.gx[i] - 1);
            const int x1 = std::min(gx_dim - 1, gw.gx[i] + 1);
            const int y0 = std::max(0, gw.gy[i] - 1);
            const int y1 = std::min(gy_dim - 1, gw.gy[i] + 1);
            for (int cx = x0; cx <= x1; ++cx) {
                const int row_base = cx * gy_dim;
                for (int cy = y0; cy <= y1; ++cy) {
                    int j = gw.head[row_base + cy];
                    while (j != -1) {
                        if (j > i) accum_pair(i, j);
                        j = gw.nxt[j];
                    }
                }
            }
        }
    } else {
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                accum_pair(i, j);
            }
        }
    }

    return energy;
}

double lbfgs_minimize(
    int n, double* z, double L, double w_wall, double w_pair, double side,
    int maxiter, LbfgsWorkspace& ws, const double* th_ref = nullptr, double w_ang = 0.0
) {
    ws.init(n);
    const int dim = ws.dim;
    constexpr int M = LbfgsWorkspace::M;

    double f = compute_energy_grad(n, z, L, w_wall, w_pair, side, ws.grid, ws.g.data(), nullptr, th_ref, w_ang);
    if (f < 1e-15) return f;

    int head = 0;
    int hist_cnt = 0;
    double gamma = 1.0;

    for (int it = 0; it < maxiter; ++it) {
        double g_inf = 0.0;
        for (int k = 0; k < dim; ++k) {
            const double ak = std::fabs(ws.g[k]);
            if (ak > g_inf) g_inf = ak;
        }
        if (g_inf < 1e-9 || f < 1e-14) break;

        // Two-loop recursion
        std::copy_n(ws.g.data(), dim, ws.q.data());
        for (int idx = 0; idx < hist_cnt; ++idx) {
            const int slot = (head - 1 - idx + M) % M;
            const double* sk = ws.s_hist.data() + slot * dim;
            const double* yk = ws.y_hist.data() + slot * dim;
            double dot = 0.0;
            for (int k = 0; k < dim; ++k) dot += sk[k] * ws.q[k];
            const double a = ws.rho[slot] * dot;
            ws.alpha_buf[slot] = a;
            for (int k = 0; k < dim; ++k) ws.q[k] -= a * yk[k];
        }
        for (int k = 0; k < dim; ++k) ws.d[k] = gamma * ws.q[k];
        for (int idx = hist_cnt - 1; idx >= 0; --idx) {
            const int slot = (head - 1 - idx + M) % M;
            const double* sk = ws.s_hist.data() + slot * dim;
            const double* yk = ws.y_hist.data() + slot * dim;
            double dot = 0.0;
            for (int k = 0; k < dim; ++k) dot += yk[k] * ws.d[k];
            const double b = ws.rho[slot] * dot;
            const double coef = ws.alpha_buf[slot] - b;
            for (int k = 0; k < dim; ++k) ws.d[k] += coef * sk[k];
        }
        for (int k = 0; k < dim; ++k) ws.d[k] = -ws.d[k];

        double dg = 0.0;
        double d_inf = 0.0;
        for (int k = 0; k < dim; ++k) {
            dg += ws.g[k] * ws.d[k];
            const double ad = std::fabs(ws.d[k]);
            if (ad > d_inf) d_inf = ad;
        }
        if (dg >= 0.0 || !std::isfinite(dg)) {
            hist_cnt = 0;
            gamma = 1.0;
            d_inf = 0.0;
            dg = 0.0;
            for (int k = 0; k < dim; ++k) {
                ws.d[k] = -ws.g[k];
                dg -= ws.g[k] * ws.g[k];
                const double ad = std::fabs(ws.d[k]);
                if (ad > d_inf) d_inf = ad;
            }
        }
        if (d_inf < 1e-15) break;

        double alpha = (hist_cnt == 0) ? std::min(1.0, 0.25 / std::max(1.0, d_inf)) : 1.0;
        if (alpha * d_inf > 0.45 * side) {
            alpha = (0.45 * side) / d_inf;
        }

        bool accepted = false;
        double f_new = f;
        for (int ls = 0; ls < 20; ++ls) {
            for (int k = 0; k < dim; ++k) ws.z_new[k] = z[k] + alpha * ws.d[k];
            f_new = compute_energy_grad(n, ws.z_new.data(), L, w_wall, w_pair, side, ws.grid, ws.g_new.data(), nullptr, th_ref, w_ang);
            if (std::isfinite(f_new) && f_new <= f + 1e-4 * alpha * dg) {
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }

        if (!accepted) {
            if (hist_cnt > 0) {
                hist_cnt = 0;
                gamma = 1.0;
                continue;
            }
            break;
        }

        double ys = 0.0, yy = 0.0, ss = 0.0;
        double* sk = ws.s_hist.data() + head * dim;
        double* yk = ws.y_hist.data() + head * dim;
        for (int k = 0; k < dim; ++k) {
            const double skv = ws.z_new[k] - z[k];
            const double ykv = ws.g_new[k] - ws.g[k];
            sk[k] = skv;
            yk[k] = ykv;
            ys += skv * ykv;
            yy += ykv * ykv;
            ss += skv * skv;
        }
        std::copy_n(ws.z_new.data(), dim, z);
        std::copy_n(ws.g_new.data(), dim, ws.g.data());
        const double rel_df = (f - f_new) / std::max(1.0, f);
        f = f_new;

        if (ys > 1e-14 * std::sqrt(yy * ss + 1e-30) && yy > 1e-24) {
            ws.rho[head] = 1.0 / ys;
            gamma = ys / yy;
            head = (head + 1) % M;
            if (hist_cnt < M) ++hist_cnt;
        }

        if (it > 15 && rel_df < 1e-12 && f < 1e-9) break;
    }
    return f;
}

void run_single_inflate_trajectory(
    int n, double* z, double L_tgt,
    int n_stages, const double* a_sched, const int* stage_iters, const double* stage_wang,
    int n_kicks, double kick_ang, double kick_xy, uint64_t rng_seed,
    LbfgsWorkspace& ws, std::vector<double>& z_best, std::vector<double>& per_pen,
    std::vector<double>& th_ref,
    double& out_E, double& out_pen, double& out_L_bbox
) {
    const int dim = 3 * n;
    z_best.resize(dim);
    per_pen.resize(n);
    th_ref.assign(z + 2 * n, z + 3 * n);
    std::mt19937_64 rng(rng_seed);
    std::uniform_real_distribution<double> u_ang(-kick_ang, kick_ang);
    std::uniform_real_distribution<double> u_xy(-kick_xy, kick_xy);
    std::uniform_real_distribution<double> u_01(0.0, 1.0);

    // Clamp outliers smoothly inside [0.35, L_tgt - 0.35] before stage 0
    for (int i = 0; i < n; ++i) {
        if (z[i] < 0.3) z[i] = 0.35;
        if (z[i] > L_tgt - 0.3) z[i] = L_tgt - 0.35;
        if (z[n + i] < 0.3) z[n + i] = 0.35;
        if (z[n + i] > L_tgt - 0.3) z[n + i] = L_tgt - 0.35;
    }

    double cur_E = 0.0;
    for (int st = 0; st < n_stages; ++st) {
        const double a = a_sched[st];
        const int mit = stage_iters[st];
        const double wang = stage_wang ? stage_wang[st] : 0.0;
        cur_E = lbfgs_minimize(n, z, L_tgt, 2.0, 1.0, a, mit, ws, th_ref.data(), wang);

        if (n_kicks > 0 && a >= 0.92 && a <= 0.988 && cur_E > 1e-10) {
            std::copy_n(z, dim, z_best.data());
            double best_E = cur_E;
            compute_energy_grad(n, z_best.data(), L_tgt, 2.0, 1.0, a, ws.grid, ws.g.data(), per_pen.data(), th_ref.data(), wang);

            int hot_idx = 0;
            double hot_val = -1.0;
            for (int i = 0; i < n; ++i) {
                if (per_pen[i] > hot_val) {
                    hot_val = per_pen[i];
                    hot_idx = i;
                }
            }
            const double hx_c = z_best[hot_idx];
            const double hy_c = z_best[n + hot_idx];

            for (int k = 0; k < n_kicks; ++k) {
                std::copy_n(z_best.data(), dim, z);
                const double r2_cut = (k % 2 == 0) ? 9.0 : 25.0;
                const double rot_bias = u_ang(rng);
                for (int i = 0; i < n; ++i) {
                    const double dx = z[i] - hx_c;
                    const double dy = z[n + i] - hy_c;
                    const bool in_cluster = (dx * dx + dy * dy <= r2_cut);
                    if (in_cluster || per_pen[i] > 1e-4 || u_01(rng) < 0.05) {
                        z[i] += u_xy(rng);
                        z[n + i] += u_xy(rng);
                        z[2 * n + i] += (k % 2 == 0) ? rot_bias * 0.8 + 0.3 * u_ang(rng) : u_ang(rng);
                    }
                }
                const double trial_E = lbfgs_minimize(n, z, L_tgt, 2.0, 1.0, a, std::max(80, mit / 2), ws, th_ref.data(), wang);
                if (trial_E < best_E - 1e-12) {
                    best_E = trial_E;
                    std::copy_n(z, dim, z_best.data());
                    if (best_E < 1e-12) break;
                }
            }
            std::copy_n(z_best.data(), dim, z);
            cur_E = best_E;
        }
    }

    // Final high-precision wall-tightening polish at side = 1.0 (zero angle tether!)
    cur_E = lbfgs_minimize(n, z, L_tgt, 4.0, 1.0, 1.0, 450, ws);
    if (cur_E > 1e-13) {
        cur_E = lbfgs_minimize(n, z, L_tgt, 12.0, 2.0, 1.0, 300, ws);
    }

    double bbox_L = 0.0;
    const double pen = compute_sat_bbox_and_pen(n, z, z + n, z + 2 * n, 1.0, ws.grid, &bbox_L);
    out_E = cur_E;
    out_pen = pen;
    out_L_bbox = bbox_L;
}

}  // namespace

extern "C" {

double eval_sat_pen_bbox(int n, const double* x, const double* y, const double* th, double side, double* out_L) {
    GridWorkspace gw;
    return compute_sat_bbox_and_pen(n, x, y, th, side, gw, out_L);
}

double eval_sat_energy_grad(
    int n, const double* x, const double* y, const double* th,
    double L, double w_wall, double w_pair, double side, double* grad_out
) {
    std::vector<double> z(3 * n);
    std::copy_n(x, n, z.data());
    std::copy_n(y, n, z.data() + n);
    std::copy_n(th, n, z.data() + 2 * n);
    GridWorkspace gw;
    return compute_energy_grad(n, z.data(), L, w_wall, w_pair, side, gw, grad_out, nullptr);
}

void batch_inflate_solve(
    int batch_size, int n,
    double* states_xyz,      // [batch_size * 3 * n], modified in-place
    const double* L_targets, // [batch_size]
    int n_stages, const double* a_sched, const int* stage_iters, const double* stage_wang,
    int n_kicks, double kick_ang, double kick_xy,
    int n_threads, uint64_t base_seed,
    double* out_E, double* out_pen, double* out_L_bbox
) {
    const int threads = std::max(1, std::min(7, n_threads));
    const int stride = 3 * n;
    std::atomic_int next_idx{0};

    auto worker_fn = [&]() {
        LbfgsWorkspace ws;
        std::vector<double> z_best, per_pen, th_ref;
        while (true) {
            const int b = next_idx.fetch_add(1, std::memory_order_relaxed);
            if (b >= batch_size) break;
            double e = 0.0, p = 0.0, lb = 0.0;
            run_single_inflate_trajectory(
                n, states_xyz + static_cast<size_t>(b) * stride, L_targets[b],
                n_stages, a_sched, stage_iters, stage_wang, n_kicks, kick_ang, kick_xy,
                base_seed + static_cast<uint64_t>(b) * 1000003ULL + 7ULL,
                ws, z_best, per_pen, th_ref, e, p, lb
            );
            out_E[b] = e;
            out_pen[b] = p;
            out_L_bbox[b] = lb;
        }
    };

    if (threads == 1 || batch_size <= 1) {
        worker_fn();
    } else {
        std::vector<std::thread> pool;
        pool.reserve(threads);
        for (int t = 0; t < threads; ++t) pool.emplace_back(worker_fn);
        for (auto& thd : pool) thd.join();
    }
}

}  // extern "C"
