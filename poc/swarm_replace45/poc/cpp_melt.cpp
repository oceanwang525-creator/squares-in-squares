// Fast C++17 Spatial-Hash SAT L-BFGS a-melt + L-continuation + Cavity Relocation Engine.
// CPU only; threads strictly capped at <= 4.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
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
    std::vector<double> z_new, g, g_new, q_vec, d, per_pen;
    std::vector<double> s_hist, y_hist, rho, alpha_buf;
    GridWorkspace grid;

    void init(int n_in) {
        if (n == n_in) return;
        n = n_in;
        dim = 3 * n;
        z_new.resize(dim);
        g.resize(dim);
        g_new.resize(dim);
        q_vec.resize(dim);
        d.resize(dim);
        per_pen.resize(n);
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
    if (out_L) *out_L = std::max(maxx - minx, maxy - miny);

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
        if (static_cast<int>(gw.head.size()) < n_cells) gw.head.resize(n_cells);
        std::fill_n(gw.head.data(), n_cells, -1);
        for (int i = 0; i < n; ++i) {
            int gxi = std::max(0, std::min(gx_dim - 1, static_cast<int>(std::floor((x[i] - cxmin) / cell))));
            int gyi = std::max(0, std::min(gy_dim - 1, static_cast<int>(std::floor((y[i] - cymin) / cell))));
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
            for (int j = i + 1; j < n; ++j) check_pair(i, j);
        }
    }
    return max_pen;
}

inline double compute_energy_grad_full(
    int n, const double* z, double L, double w_wall, double w_pair,
    double side, const int* free_mask, GridWorkspace& gw, double* grad, double* per_sq_pen = nullptr
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
            const double ev = w_wall * v * v;
            energy += ev;
            if (per_sq_pen) per_sq_pen[i] += ev;
            const double g = 2.0 * w_wall * v;
            g_x[i] -= g;
            g_th[i] += g * dhxi;
        }
        v = x[i] + hxi - L;
        if (v > 0.0) {
            const double ev = w_wall * v * v;
            energy += ev;
            if (per_sq_pen) per_sq_pen[i] += ev;
            const double g = 2.0 * w_wall * v;
            g_x[i] += g;
            g_th[i] += g * dhxi;
        }
        v = hxi - y[i];
        if (v > 0.0) {
            const double ev = w_wall * v * v;
            energy += ev;
            if (per_sq_pen) per_sq_pen[i] += ev;
            const double g = 2.0 * w_wall * v;
            g_y[i] -= g;
            g_th[i] += g * dhxi;
        }
        v = y[i] + hxi - L;
        if (v > 0.0) {
            const double ev = w_wall * v * v;
            energy += ev;
            if (per_sq_pen) per_sq_pen[i] += ev;
            const double g = 2.0 * w_wall * v;
            g_y[i] += g;
            g_th[i] += g * dhxi;
        }
    }

    const double cell = 1.42 * side + 1e-6;
    const int gx_dim = static_cast<int>(std::floor((cxmax - cxmin) / cell)) + 1;
    const int gy_dim = static_cast<int>(std::floor((cymax - cymin) / cell)) + 1;

    auto accum_pair = [&](int i, int j) {
        const double dx = x[j] - x[i];
        const double hsum = gw.hx[i] + gw.hx[j];
        if (std::fabs(dx) >= hsum) return;
        const double dy = y[j] - y[i];
        if (std::fabs(dy) >= hsum) return;

        const double ci = gw.c[i], si = gw.s[i];
        const double cj = gw.c[j], sj = gw.s[j];
        const double cc = ci * cj + si * sj;
        const double cs = ci * sj - si * cj;
        const double rad = half * (1.0 + std::fabs(cc) + std::fabs(cs));

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

        int win = 0;
        double pen = pen1;
        if (pen2 < pen) { pen = pen2; win = 1; }
        if (pen3 < pen) { pen = pen3; win = 2; }
        if (pen4 < pen) { pen = pen4; win = 3; }

        const double ep = w_pair * pen * pen;
        energy += ep;
        if (per_sq_pen) {
            per_sq_pen[i] += ep;
            per_sq_pen[j] += ep;
        }
        const double gp = 2.0 * w_pair * pen;

        const double s_cc = (cc >= 0.0) ? 1.0 : -1.0;
        const double s_cs = (cs >= 0.0) ? 1.0 : -1.0;
        const double dci = -si, dsi = ci;
        const double dcj = -sj, dsj = cj;
        const double drad_ti = half * (s_cc * (dci * cj + dsi * sj) + s_cs * (dci * sj - dsi * cj));
        const double drad_tj = half * (s_cc * (ci * dcj + si * dsj) + s_cs * (ci * dsj - si * dcj));

        if (win == 0) {
            const double sp = (p1 >= 0.0) ? 1.0 : -1.0;
            const double d_dx = -sp * ci;
            const double d_dy = -sp * si;
            g_x[i] -= gp * d_dx;
            g_x[j] += gp * d_dx;
            g_y[i] -= gp * d_dy;
            g_y[j] += gp * d_dy;
            g_th[i] += gp * (drad_ti - sp * (dx * dci + dy * dsi));
            g_th[j] += gp * drad_tj;
        } else if (win == 1) {
            const double sp = (p2 >= 0.0) ? 1.0 : -1.0;
            const double d_dx = sp * si;
            const double d_dy = -sp * ci;
            g_x[i] -= gp * d_dx;
            g_x[j] += gp * d_dx;
            g_y[i] -= gp * d_dy;
            g_y[j] += gp * d_dy;
            g_th[i] += gp * (drad_ti - sp * (-dx * dsi + dy * dci));
            g_th[j] += gp * drad_tj;
        } else if (win == 2) {
            const double sp = (p3 >= 0.0) ? 1.0 : -1.0;
            const double d_dx = -sp * cj;
            const double d_dy = -sp * sj;
            g_x[i] -= gp * d_dx;
            g_x[j] += gp * d_dx;
            g_y[i] -= gp * d_dy;
            g_y[j] += gp * d_dy;
            g_th[i] += gp * drad_ti;
            g_th[j] += gp * (drad_tj - sp * (dx * dcj + dy * dsj));
        } else {
            const double sp = (p4 >= 0.0) ? 1.0 : -1.0;
            const double d_dx = sp * sj;
            const double d_dy = -sp * cj;
            g_x[i] -= gp * d_dx;
            g_x[j] += gp * d_dx;
            g_y[i] -= gp * d_dy;
            g_y[j] += gp * d_dy;
            g_th[i] += gp * drad_ti;
            g_th[j] += gp * (drad_tj - sp * (-dx * dsj + dy * dcj));
        }
    };

    if (gx_dim > 0 && gy_dim > 0 && gx_dim <= 512 && gy_dim <= 512) {
        const int n_cells = gx_dim * gy_dim;
        if (static_cast<int>(gw.head.size()) < n_cells) gw.head.resize(n_cells);
        std::fill_n(gw.head.data(), n_cells, -1);
        for (int i = 0; i < n; ++i) {
            int gxi = std::max(0, std::min(gx_dim - 1, static_cast<int>(std::floor((x[i] - cxmin) / cell))));
            int gyi = std::max(0, std::min(gy_dim - 1, static_cast<int>(std::floor((y[i] - cymin) / cell))));
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
            for (int j = i + 1; j < n; ++j) accum_pair(i, j);
        }
    }

    if (free_mask) {
        for (int i = 0; i < n; ++i) {
            if (!free_mask[i]) {
                g_x[i] = 0.0;
                g_y[i] = 0.0;
                g_th[i] = 0.0;
            }
        }
    }
    return energy;
}

double lbfgs_minimize(
    int n, double* z, double L, double w_wall, double w_pair,
    double side, int maxiter, const int* free_mask, LbfgsWorkspace& ws
) {
    ws.init(n);
    const int dim = ws.dim;
    constexpr int M = LbfgsWorkspace::M;

    double f = compute_energy_grad_full(n, z, L, w_wall, w_pair, side, free_mask, ws.grid, ws.g.data());
    if (f < 1e-16) return f;

    int head = 0;
    int hist_cnt = 0;
    double gamma = 1.0;

    for (int it = 0; it < maxiter; ++it) {
        double g_inf = 0.0;
        for (int k = 0; k < dim; ++k) g_inf = std::max(g_inf, std::fabs(ws.g[k]));
        if (g_inf < 1e-10 || f < 1e-15) break;

        std::copy_n(ws.g.data(), dim, ws.q_vec.data());
        for (int idx = 0; idx < hist_cnt; ++idx) {
            const int slot = (head - 1 - idx + M) % M;
            const double* sk = ws.s_hist.data() + slot * dim;
            const double* yk = ws.y_hist.data() + slot * dim;
            double dot = 0.0;
            for (int k = 0; k < dim; ++k) dot += sk[k] * ws.q_vec[k];
            const double a = ws.rho[slot] * dot;
            ws.alpha_buf[slot] = a;
            for (int k = 0; k < dim; ++k) ws.q_vec[k] -= a * yk[k];
        }
        for (int k = 0; k < dim; ++k) ws.d[k] = gamma * ws.q_vec[k];
        for (int idx = hist_cnt - 1; idx >= 0; --idx) {
            const int slot = (head - 1 - idx + M) % M;
            const double* sk = ws.s_hist.data() + slot * dim;
            const double* yk = ws.y_hist.data() + slot * dim;
            double dot = 0.0;
            for (int k = 0; k < dim; ++k) dot += yk[k] * ws.d[k];
            const double coef = ws.alpha_buf[slot] - ws.rho[slot] * dot;
            for (int k = 0; k < dim; ++k) ws.d[k] += coef * sk[k];
        }
        for (int k = 0; k < dim; ++k) ws.d[k] = -ws.d[k];

        double dg = 0.0, d_inf = 0.0;
        for (int k = 0; k < dim; ++k) {
            dg += ws.g[k] * ws.d[k];
            d_inf = std::max(d_inf, std::fabs(ws.d[k]));
        }
        if (dg >= 0.0 || !std::isfinite(dg)) {
            hist_cnt = 0;
            gamma = 1.0;
            d_inf = 0.0;
            dg = 0.0;
            for (int k = 0; k < dim; ++k) {
                ws.d[k] = -ws.g[k];
                dg -= ws.g[k] * ws.g[k];
                d_inf = std::max(d_inf, std::fabs(ws.d[k]));
            }
        }
        if (d_inf < 1e-15) break;

        double alpha = (hist_cnt == 0) ? std::min(1.0, 0.25 / std::max(1.0, d_inf)) : 1.0;
        if (alpha * d_inf > 0.45 * side) alpha = (0.45 * side) / d_inf;

        bool accepted = false;
        double f_new = f;
        for (int ls = 0; ls < 20; ++ls) {
            for (int k = 0; k < dim; ++k) ws.z_new[k] = z[k] + alpha * ws.d[k];
            f_new = compute_energy_grad_full(
                n, ws.z_new.data(), L, w_wall, w_pair, side, free_mask, ws.grid, ws.g_new.data()
            );
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
        if (it > 15 && rel_df < 1e-12 && f < 1e-10) break;
    }
    return f;
}

void run_melt_trajectory(
    int n, double* z, double* z_feas,
    int n_L_steps, const double* L_sched,
    int n_a_steps, const double* a_sched,
    const int* iters_per_a,
    int ils_jitter_rounds,
    const int* free_mask,
    uint64_t seed,
    LbfgsWorkspace& ws,
    double* out_best_L_feas, double* out_E, double* out_pen, double* out_L_bbox
) {
    ws.init(n);
    const int dim = ws.dim;
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u_01(0.0, 1.0);
    std::normal_distribution<double> n_xy(0.0, 0.08);
    std::normal_distribution<double> n_th(0.0, 0.09);

    double best_L_feas = 1e18;
    double best_overall_E = 1e18;
    std::vector<double> z_best(z, z + dim);
    std::vector<double> z_trial(dim);
    std::vector<double> z_stage_best(dim);

    auto check_feas = [&](const double* cand, double L_box) -> double {
        double minx = 1e18, maxx = -1e18, miny = 1e18, maxy = -1e18;
        for (int i = 0; i < n; ++i) {
            const double ci = std::cos(cand[2 * n + i]);
            const double si = std::sin(cand[2 * n + i]);
            const double hxi = 0.5 * (std::fabs(ci) + std::fabs(si));
            minx = std::min(minx, cand[i] - hxi);
            maxx = std::max(maxx, cand[i] + hxi);
            miny = std::min(miny, cand[n + i] - hxi);
            maxy = std::max(maxy, cand[n + i] + hxi);
        }
        const double l_bb = std::max(maxx - minx, maxy - miny);
        const double pen = compute_sat_bbox_and_pen(n, cand, cand + n, cand + 2 * n, 1.0, ws.grid, nullptr);
        if (pen <= 1e-11 && l_bb < best_L_feas) {
            best_L_feas = l_bb;
            std::copy_n(cand, dim, z_feas);
        }
        return pen;
    };

    for (int ls = 0; ls < n_L_steps; ++ls) {
        const double L_tgt = L_sched[ls];
        for (int as = 0; as < n_a_steps; ++as) {
            lbfgs_minimize(n, z, L_tgt, 3.0, 1.0, a_sched[as], iters_per_a[as], free_mask, ws);
        }
        double cur_E = compute_energy_grad_full(n, z, L_tgt, 5.0, 1.0, 1.0, free_mask, ws.grid, ws.g.data(), ws.per_pen.data());
        double cur_pen = check_feas(z, L_tgt);

        if (cur_pen > 1e-9 && ils_jitter_rounds > 0) {
            std::copy_n(z, dim, z_stage_best.data());
            double stage_best_E = cur_E;
            for (int jr = 0; jr < ils_jitter_rounds; ++jr) {
                std::copy_n(z_stage_best.data(), dim, z_trial.data());
                // Find highest-penalty square as local hotspot center
                int hot = 0;
                double hot_e = -1.0;
                for (int i = 0; i < n; ++i) {
                    if (free_mask && !free_mask[i]) continue;
                    if (ws.per_pen[i] > hot_e) {
                        hot_e = ws.per_pen[i];
                        hot = i;
                    }
                }
                const double hx_c = z_trial[hot], hy_c = z_trial[n + hot];
                const double rad2 = (jr % 2 == 0) ? 9.0 : 20.0;
                for (int i = 0; i < n; ++i) {
                    if (free_mask && !free_mask[i]) continue;
                    const double dx = z_trial[i] - hx_c, dy = z_trial[n + i] - hy_c;
                    if (dx * dx + dy * dy <= rad2 || u_01(rng) < 0.03) {
                        z_trial[i] += n_xy(rng);
                        z_trial[n + i] += n_xy(rng);
                        z_trial[2 * n + i] += n_th(rng);
                    }
                }
                lbfgs_minimize(n, z_trial.data(), L_tgt, 3.0, 1.0, 0.94, 140, free_mask, ws);
                const double tE = lbfgs_minimize(n, z_trial.data(), L_tgt, 5.0, 1.0, 1.0, 220, free_mask, ws);
                if (tE < stage_best_E - 1e-12) {
                    stage_best_E = tE;
                    std::copy_n(z_trial.data(), dim, z_stage_best.data());
                    check_feas(z_stage_best.data(), L_tgt);
                    if (tE < 1e-12) break;
                }
            }
            std::copy_n(z_stage_best.data(), dim, z);
            cur_E = stage_best_E;
            cur_pen = check_feas(z, L_tgt);
        }

        if (ls == n_L_steps - 1 || cur_E <= best_overall_E) {
            best_overall_E = cur_E;
            std::copy_n(z, dim, z_best.data());
            *out_E = cur_E;
            *out_pen = cur_pen;
            compute_sat_bbox_and_pen(n, z, z + n, z + 2 * n, 1.0, ws.grid, out_L_bbox);
        }
    }

    *out_best_L_feas = best_L_feas;
    std::copy_n(z_best.data(), dim, z);
}

}  // namespace

extern "C" {

void batch_melt_solve(
    int batch_size, int n,
    double* z_batch,          // [batch_size * 3 * n]
    double* z_feas_batch,     // [batch_size * 3 * n]
    int n_L_steps, const double* L_sched_batch,  // [batch_size * n_L_steps]
    int n_a_steps, const double* a_sched,        // [n_a_steps]
    const int* iters_per_a,                      // [n_a_steps]
    int ils_jitter_rounds,
    const int* free_mask_batch,                  // nullable or [batch_size * n]
    int n_threads, uint64_t base_seed,
    double* out_best_L_feas, double* out_E, double* out_pen, double* out_L_bbox
) {
    const int threads = std::max(1, std::min(4, n_threads));
    const int dim = 3 * n;
    std::atomic_int next_idx{0};

    auto worker_fn = [&]() {
        LbfgsWorkspace ws;
        while (true) {
            const int b = next_idx.fetch_add(1, std::memory_order_relaxed);
            if (b >= batch_size) break;
            double bL_feas = 1e18, e = 0.0, p = 0.0, lb = 0.0;
            const int* fmask = free_mask_batch ? (free_mask_batch + static_cast<size_t>(b) * n) : nullptr;
            run_melt_trajectory(
                n,
                z_batch + static_cast<size_t>(b) * dim,
                z_feas_batch + static_cast<size_t>(b) * dim,
                n_L_steps, L_sched_batch + static_cast<size_t>(b) * n_L_steps,
                n_a_steps, a_sched, iters_per_a,
                ils_jitter_rounds, fmask,
                base_seed + static_cast<uint64_t>(b) * 1000003ULL + 7ULL,
                ws, &bL_feas, &e, &p, &lb
            );
            out_best_L_feas[b] = bL_feas;
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
