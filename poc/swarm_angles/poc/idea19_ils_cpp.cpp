// Idea 19: Symmetry-Reduced (C4 / C2 / Free) C++ Spatial-Hash SAT Energy/Gradient,
// Cavity-Relocation ILS, Local Cluster Melt, and L-Continuation Solver.
// Compiled into /tmp/libidea19.so (CPU only, std::thread capped <= 7).

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

constexpr double kPi = 3.14159265358979323846;

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

struct SymLbfgsWorkspace {
    int n = 0;
    int sym_mode = 1;
    int m_orbits = 0;
    int has_center = 0;
    int dim = 0;
    static constexpr int M = 10;
    std::vector<double> z_full, g_full, per_pen;
    std::vector<double> q_new, g, g_new, q_vec, d;
    std::vector<double> s_hist, y_hist, rho, alpha_buf;
    GridWorkspace grid;

    void init(int n_in, int mode_in) {
        if (n == n_in && sym_mode == mode_in) return;
        n = n_in;
        sym_mode = mode_in;
        has_center = (mode_in == 1) ? 0 : (n % mode_in);
        m_orbits = (mode_in == 1) ? n : ((n - has_center) / mode_in);
        dim = 3 * m_orbits + has_center;

        z_full.resize(3 * n);
        g_full.resize(3 * n);
        per_pen.resize(n);

        q_new.resize(dim);
        g.resize(dim);
        g_new.resize(dim);
        q_vec.resize(dim);
        d.resize(dim);
        s_hist.assign(M * dim, 0.0);
        y_hist.assign(M * dim, 0.0);
        rho.assign(M, 0.0);
        alpha_buf.assign(M, 0.0);
        grid.resize(n);
    }
};

// Expand reduced parameters q_red around box center (L/2, L/2) into full [0, L]^2 coordinates z_full.
// Layout of q_red (length dim = 3*m_orbits + has_center):
//   rx[0..m-1], ry[0..m-1], th[0..m-1], [th_center if has_center]
inline void expand_sym(
    int n, int sym_mode, int m_orbits, int has_center, double L,
    const double* q_red, double* z_full
) {
    const double hc = 0.5 * L;
    double* fx = z_full;
    double* fy = z_full + n;
    double* ft = z_full + 2 * n;

    if (sym_mode == 1) {
        for (int j = 0; j < n; ++j) {
            fx[j] = hc + q_red[j];
            fy[j] = hc + q_red[n + j];
            ft[j] = q_red[2 * n + j];
        }
        return;
    }

    const double* rx = q_red;
    const double* ry = q_red + m_orbits;
    const double* rt = q_red + 2 * m_orbits;

    if (sym_mode == 2) {
        for (int j = 0; j < m_orbits; ++j) {
            const double dx = rx[j];
            const double dy = ry[j];
            const double th = rt[j];
            const int i0 = 2 * j;
            const int i1 = i0 + 1;
            fx[i0] = hc + dx;
            fy[i0] = hc + dy;
            ft[i0] = th;
            fx[i1] = hc - dx;
            fy[i1] = hc - dy;
            ft[i1] = th + kPi;
        }
    } else if (sym_mode == 4) {
        for (int j = 0; j < m_orbits; ++j) {
            const double dx = rx[j];
            const double dy = ry[j];
            const double th = rt[j];
            const int b = 4 * j;
            fx[b + 0] = hc + dx;  fy[b + 0] = hc + dy;  ft[b + 0] = th;
            fx[b + 1] = hc - dy;  fy[b + 1] = hc + dx;  ft[b + 1] = th + 0.5 * kPi;
            fx[b + 2] = hc - dx;  fy[b + 2] = hc - dy;  ft[b + 2] = th + kPi;
            fx[b + 3] = hc + dy;  fy[b + 3] = hc - dx;  ft[b + 3] = th + 1.5 * kPi;
        }
    }
    if (has_center) {
        const int ic = sym_mode * m_orbits;
        fx[ic] = hc;
        fy[ic] = hc;
        ft[ic] = q_red[3 * m_orbits];
    }
}

inline void fold_sym_grad(
    int n, int sym_mode, int m_orbits, int has_center,
    const double* g_full, double* g_red
) {
    const double* gx = g_full;
    const double* gy = g_full + n;
    const double* gt = g_full + 2 * n;

    if (sym_mode == 1) {
        std::copy_n(g_full, 3 * n, g_red);
        return;
    }

    double* grx = g_red;
    double* gry = g_red + m_orbits;
    double* grt = g_red + 2 * m_orbits;

    if (sym_mode == 2) {
        for (int j = 0; j < m_orbits; ++j) {
            const int i0 = 2 * j;
            const int i1 = i0 + 1;
            grx[j] = gx[i0] - gx[i1];
            gry[j] = gy[i0] - gy[i1];
            grt[j] = gt[i0] + gt[i1];
        }
    } else if (sym_mode == 4) {
        for (int j = 0; j < m_orbits; ++j) {
            const int b = 4 * j;
            grx[j] = gx[b + 0] + gy[b + 1] - gx[b + 2] - gy[b + 3];
            gry[j] = gy[b + 0] - gx[b + 1] - gy[b + 2] + gx[b + 3];
            grt[j] = gt[b + 0] + gt[b + 1] + gt[b + 2] + gt[b + 3];
        }
    }
    if (has_center) {
        const int ic = sym_mode * m_orbits;
        g_red[3 * m_orbits] = gt[ic];
    }
}

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
    double side, GridWorkspace& gw, double* grad, double* per_sq_pen = nullptr
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

    return energy;
}

inline double eval_sym_energy_grad(
    SymLbfgsWorkspace& ws, const double* q_red,
    double L, double w_wall, double w_pair, double side,
    double* g_red_out, double* per_sq_pen = nullptr
) {
    expand_sym(ws.n, ws.sym_mode, ws.m_orbits, ws.has_center, L, q_red, ws.z_full.data());
    const double e = compute_energy_grad_full(
        ws.n, ws.z_full.data(), L, w_wall, w_pair, side, ws.grid, ws.g_full.data(), per_sq_pen
    );
    fold_sym_grad(ws.n, ws.sym_mode, ws.m_orbits, ws.has_center, ws.g_full.data(), g_red_out);
    return e;
}

double lbfgs_minimize_sym(
    int n, int sym_mode, double* q_red, double L,
    double w_wall, double w_pair, double side, int maxiter, SymLbfgsWorkspace& ws
) {
    ws.init(n, sym_mode);
    const int dim = ws.dim;
    constexpr int M = SymLbfgsWorkspace::M;

    double f = eval_sym_energy_grad(ws, q_red, L, w_wall, w_pair, side, ws.g.data());
    if (f < 1e-15) return f;

    int head = 0;
    int hist_cnt = 0;
    double gamma = 1.0;

    for (int it = 0; it < maxiter; ++it) {
        double g_inf = 0.0;
        for (int k = 0; k < dim; ++k) g_inf = std::max(g_inf, std::fabs(ws.g[k]));
        if (g_inf < 1e-9 || f < 1e-14) break;

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
            for (int k = 0; k < dim; ++k) ws.q_new[k] = q_red[k] + alpha * ws.d[k];
            f_new = eval_sym_energy_grad(ws, ws.q_new.data(), L, w_wall, w_pair, side, ws.g_new.data());
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
            const double skv = ws.q_new[k] - q_red[k];
            const double ykv = ws.g_new[k] - ws.g[k];
            sk[k] = skv;
            yk[k] = ykv;
            ys += skv * ykv;
            yy += ykv * ykv;
            ss += skv * skv;
        }
        std::copy_n(ws.q_new.data(), dim, q_red);
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

// Compute single-square energy against all squares in z_full EXCEPT orbit victim_orb (whose indices are skipped)
inline double eval_single_sq_energy(
    int n, const double* x, const double* y, const double* c, const double* s, const double* hx,
    double L, double w_wall, double w_pair, double side,
    double px, double py, double pth, int skip_start, int skip_end
) {
    const double half = 0.5 * side;
    const double ci = std::cos(pth), si = std::sin(pth);
    const double hxi = half * (std::fabs(ci) + std::fabs(si));
    double e = 0.0;
    double v = hxi - px; if (v > 0.0) e += w_wall * v * v;
    v = px + hxi - L;    if (v > 0.0) e += w_wall * v * v;
    v = hxi - py;        if (v > 0.0) e += w_wall * v * v;
    v = py + hxi - L;    if (v > 0.0) e += w_wall * v * v;

    for (int j = 0; j < n; ++j) {
        if (j >= skip_start && j < skip_end) continue;
        const double dx = x[j] - px;
        const double hsum = hxi + hx[j];
        if (std::fabs(dx) >= hsum) continue;
        const double dy = y[j] - py;
        if (std::fabs(dy) >= hsum) continue;
        const double rad = half * (1.0 + std::fabs(ci * c[j] + si * s[j]) + std::fabs(ci * s[j] - si * c[j]));
        const double p1 = rad - std::fabs(dx * ci + dy * si); if (p1 <= 0.0) continue;
        const double p2 = rad - std::fabs(-dx * si + dy * ci); if (p2 <= 0.0) continue;
        const double p3 = rad - std::fabs(dx * c[j] + dy * s[j]); if (p3 <= 0.0) continue;
        const double p4 = rad - std::fabs(-dx * s[j] + dy * c[j]); if (p4 <= 0.0) continue;
        e += w_pair * std::pow(std::min(std::min(p1, p2), std::min(p3, p4)), 2);
    }
    return e;
}

// Relocate a hotspot orbit into the lowest-energy empty void in [0, L]^2
bool cavity_relocate_orbit(
    SymLbfgsWorkspace& ws, double* q_red, double L, double side,
    std::mt19937_64& rng, std::vector<int>& occ, std::vector<std::pair<int, int>>& empty_cells
) {
    const int n = ws.n;
    const int m = ws.m_orbits;
    if (m <= 1) return false;
    eval_sym_energy_grad(ws, q_red, L, 2.0, 1.0, side, ws.g.data(), ws.per_pen.data());

    // Score each orbit by max square penalty
    std::vector<std::pair<double, int>> orb_scores(m);
    for (int j = 0; j < m; ++j) {
        double sc = 0.0;
        for (int k = 0; k < ws.sym_mode; ++k) {
            sc = std::max(sc, ws.per_pen[j * ws.sym_mode + k]);
        }
        orb_scores[j] = {sc, j};
    }
    std::sort(orb_scores.begin(), orb_scores.end(), [](auto& a, auto& b) { return a.first > b.first; });
    if (orb_scores[0].first < 1e-11) return false;

    const int top_k = std::min(m, 4);
    int pick_rank = std::uniform_int_distribution<int>(0, top_k - 1)(rng);
    if (orb_scores[pick_rank].first < 1e-12) pick_rank = 0;
    const int vic = orb_scores[pick_rank].second;

    const int skip_start = vic * ws.sym_mode;
    const int skip_end = skip_start + ws.sym_mode;

    const double* fx = ws.z_full.data();
    const double* fy = ws.z_full.data() + n;
    const double* ft = ws.z_full.data() + 2 * n;
    const double half = 0.5 * side;
    std::vector<double> c(n), s(n), hx(n);
    for (int i = 0; i < n; ++i) {
        c[i] = std::cos(ft[i]);
        s[i] = std::sin(ft[i]);
        hx[i] = half * (std::fabs(c[i]) + std::fabs(s[i]));
    }

    // Build 0.5-unit occupancy grid over [0, L]^2 to find least-occupied void pockets
    const double gstep = 0.5;
    const int ng = std::max(4, static_cast<int>(std::floor(L / gstep)));
    occ.assign(ng * ng, 0);
    for (int i = 0; i < n; ++i) {
        if (i >= skip_start && i < skip_end) continue;
        int ix = std::max(0, std::min(ng - 1, static_cast<int>(fx[i] / gstep)));
        int iy = std::max(0, std::min(ng - 1, static_cast<int>(fy[i] / gstep)));
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                int ux = ix + dx, uy = iy + dy;
                if (ux >= 0 && ux < ng && uy >= 0 && uy < ng) {
                    occ[ux * ng + uy] += (dx == 0 && dy == 0) ? 4 : 1;
                }
            }
        }
    }
    empty_cells.clear();
    empty_cells.reserve(ng * ng);
    const double hc = 0.5 * L;
    for (int ix = 1; ix < ng - 1; ++ix) {
        for (int iy = 1; iy < ng - 1; ++iy) {
            const double px = (ix + 0.5) * gstep;
            const double py = (iy + 0.5) * gstep;
            if (ws.sym_mode == 2 && (px - hc) < -0.2) continue;
            if (ws.sym_mode == 4 && ((px - hc) < -0.2 || (py - hc) < -0.2)) continue;
            empty_cells.push_back({occ[ix * ng + iy], ix * ng + iy});
        }
    }
    const int n_keep = std::min<int>(24, static_cast<int>(empty_cells.size()));
    if (n_keep == 0) return false;
    std::partial_sort(empty_cells.begin(), empty_cells.begin() + n_keep, empty_cells.end());

    static constexpr double kCandAngles[] = {
        0.0, 0.2617993878, 0.4363323130, 0.5235987756, 0.6108652382, 0.6981317008, 0.7853981634
    };
    std::uniform_real_distribution<double> u_jit(-0.25, 0.25);

    double best_ins_e = 1e18;
    double best_rx = q_red[vic];
    double best_ry = q_red[m + vic];
    double best_rt = q_red[2 * m + vic];

    for (int ci_idx = 0; ci_idx < n_keep + 12; ++ci_idx) {
        double px, py;
        if (ci_idx < n_keep) {
            const int cell_id = empty_cells[ci_idx].second;
            const int ix = cell_id / ng, iy = cell_id % ng;
            px = (ix + 0.5) * gstep + u_jit(rng);
            py = (iy + 0.5) * gstep + u_jit(rng);
        } else {
            px = std::uniform_real_distribution<double>(0.55, L - 0.55)(rng);
            py = std::uniform_real_distribution<double>(0.55, L - 0.55)(rng);
        }
        const double rx = px - hc, ry = py - hc;
        // Skip self-overlap near center under C2/C4
        if (ws.sym_mode >= 2 && (rx * rx + ry * ry) < 0.48) continue;

        for (double ang : kCandAngles) {
            double e = eval_single_sq_energy(
                n, fx, fy, c.data(), s.data(), hx.data(), L, 2.0, 1.0, side,
                px, py, ang, skip_start, skip_end
            );
            if (e < best_ins_e) {
                best_ins_e = e;
                best_rx = rx;
                best_ry = ry;
                best_rt = ang;
            }
        }
    }

    q_red[vic] = best_rx;
    q_red[m + vic] = best_ry;
    q_red[2 * m + vic] = best_rt;
    return true;
}

// One full ILS + L-Continuation trajectory in C++
void run_ils_trajectory(
    int n, int sym_mode, double* q_red,
    int n_L_steps, const double* L_sched, const int* ils_iters_per_L,
    double a_start, int n_melt_stages,
    uint64_t rng_seed, SymLbfgsWorkspace& ws,
    double* out_best_q_feas, double* out_best_L_feas,
    double* out_best_E, double* out_best_pen, double* out_L_bbox
) {
    ws.init(n, sym_mode);
    const int dim = ws.dim;
    std::mt19937_64 rng(rng_seed);
    std::uniform_real_distribution<double> u_ang(-0.35, 0.35);
    std::uniform_real_distribution<double> u_xy(-0.18, 0.18);
    std::uniform_real_distribution<double> u_01(0.0, 1.0);

    std::vector<double> q_cur(q_red, q_red + dim);
    std::vector<double> q_trial(dim), q_stage_best(dim);
    std::vector<int> occ;
    std::vector<std::pair<int, int>> empty_cells;

    *out_best_L_feas = 1e18;
    double best_overall_E = 1e18;
    std::vector<double> q_overall_best(q_red, q_red + dim);

    // Optional soft-core warm-up at L_sched[0]
    if (n_melt_stages > 0 && a_start < 0.999) {
        for (int ms = 0; ms < n_melt_stages; ++ms) {
            const double frac = (n_melt_stages == 1) ? 1.0 : static_cast<double>(ms) / (n_melt_stages - 1);
            const double a = a_start + frac * (1.0 - a_start);
            lbfgs_minimize_sym(n, sym_mode, q_cur.data(), L_sched[0], 2.0, 1.0, a, 220, ws);
        }
    }

    for (int ls = 0; ls < n_L_steps; ++ls) {
        const double L_tgt = L_sched[ls];
        double stage_E = lbfgs_minimize_sym(n, sym_mode, q_cur.data(), L_tgt, 2.0, 1.0, 1.0, 320, ws);
        double stage_best_E = stage_E;
        std::copy_n(q_cur.data(), dim, q_stage_best.data());

        // Quick feasibility check right after initial compress at L_tgt
        auto check_and_save_feas = [&](const double* cand_q) -> double {
            expand_sym(n, sym_mode, ws.m_orbits, ws.has_center, L_tgt, cand_q, ws.z_full.data());
            double bbox_L = 0.0;
            const double pen = compute_sat_bbox_and_pen(
                n, ws.z_full.data(), ws.z_full.data() + n, ws.z_full.data() + 2 * n, 1.0, ws.grid, &bbox_L
            );
            if (pen <= 1e-11 && bbox_L < *out_best_L_feas) {
                *out_best_L_feas = bbox_L;
                std::copy_n(cand_q, dim, out_best_q_feas);
            }
            return pen;
        };

        if (stage_best_E > 1e-11) {
            const int n_ils = ils_iters_per_L ? ils_iters_per_L[ls] : 0;
            for (int step = 0; step < n_ils; ++step) {
                std::copy_n(q_stage_best.data(), dim, q_trial.data());

                if (step % 3 != 2) {
                    // Move 1: Relocate 1 or 2 worst hotspot orbits into void cells + small local neighbor jitter
                    const int n_reloc = (step % 2 == 0) ? 1 : 2;
                    for (int r = 0; r < n_reloc; ++r) {
                        cavity_relocate_orbit(ws, q_trial.data(), L_tgt, 1.0, rng, occ, empty_cells);
                    }
                    // Also nudge 1-2 random neighbors slightly
                    const int m = ws.m_orbits;
                    for (int k = 0; k < std::min(4, m); ++k) {
                        const int j = std::uniform_int_distribution<int>(0, m - 1)(rng);
                        q_trial[j] += 0.5 * u_xy(rng);
                        q_trial[m + j] += 0.5 * u_xy(rng);
                        if (u_01(rng) < 0.5) q_trial[2 * m + j] += 0.3 * u_ang(rng);
                    }
                } else {
                    // Move 2: Local hotspot cluster melt (a = 0.90 -> 1.00) with coherent tilted rotation
                    eval_sym_energy_grad(ws, q_trial.data(), L_tgt, 2.0, 1.0, 1.0, ws.g.data(), ws.per_pen.data());
                    int hot_orb = 0;
                    double best_p = -1.0;
                    for (int j = 0; j < ws.m_orbits; ++j) {
                        double sp = 0.0;
                        for (int k = 0; k < ws.sym_mode; ++k) sp = std::max(sp, ws.per_pen[j * ws.sym_mode + k]);
                        if (sp > best_p) { best_p = sp; hot_orb = j; }
                    }
                    const int m = ws.m_orbits;
                    const double hrx = q_trial[hot_orb], hry = q_trial[m + hot_orb];
                    const double rot_jump = std::uniform_real_distribution<double>(0.25, 0.78)(rng);
                    const int sgn = (u_01(rng) < 0.5) ? 1 : -1;
                    for (int j = 0; j < m; ++j) {
                        const double dx = q_trial[j] - hrx, dy = q_trial[m + j] - hry;
                        if (dx * dx + dy * dy <= 7.5 || u_01(rng) < 0.04) {
                            q_trial[j] += u_xy(rng);
                            q_trial[m + j] += u_xy(rng);
                            q_trial[2 * m + j] = (u_01(rng) < 0.5) ? (sgn * rot_jump + 0.1 * u_ang(rng))
                                                                    : (q_trial[2 * m + j] + u_ang(rng));
                        }
                    }
                    lbfgs_minimize_sym(n, sym_mode, q_trial.data(), L_tgt, 2.0, 1.0, 0.91, 140, ws);
                }

                const double tE = lbfgs_minimize_sym(n, sym_mode, q_trial.data(), L_tgt, 2.0, 1.0, 1.0, 240, ws);
                if (tE < stage_best_E - 1e-12) {
                    stage_best_E = tE;
                    std::copy_n(q_trial.data(), dim, q_stage_best.data());
                    if (stage_best_E < 1e-12) break;
                }
            }
        }

        // High-precision wall tightening at this L_tgt
        std::copy_n(q_stage_best.data(), dim, q_cur.data());
        stage_best_E = lbfgs_minimize_sym(n, sym_mode, q_cur.data(), L_tgt, 6.0, 1.0, 1.0, 320, ws);
        const double stage_pen = check_and_save_feas(q_cur.data());

        if (ls == n_L_steps - 1 || stage_best_E <= best_overall_E) {
            best_overall_E = stage_best_E;
            std::copy_n(q_cur.data(), dim, q_overall_best.data());
            *out_best_E = stage_best_E;
            *out_best_pen = stage_pen;
            expand_sym(n, sym_mode, ws.m_orbits, ws.has_center, L_tgt, q_cur.data(), ws.z_full.data());
            compute_sat_bbox_and_pen(
                n, ws.z_full.data(), ws.z_full.data() + n, ws.z_full.data() + 2 * n, 1.0, ws.grid, out_L_bbox
            );
        }
    }

    std::copy_n(q_overall_best.data(), dim, q_red);
}

}  // namespace

extern "C" {

double eval_sym_sat_energy_grad(
    int n, int sym_mode, const double* q_red,
    double L, double w_wall, double w_pair, double side, double* g_red_out
) {
    SymLbfgsWorkspace ws;
    ws.init(n, sym_mode);
    return eval_sym_energy_grad(ws, q_red, L, w_wall, w_pair, side, g_red_out);
}

void expand_sym_coords(int n, int sym_mode, double L, const double* q_red, double* z_full_out) {
    const int has_c = (sym_mode == 1) ? 0 : (n % sym_mode);
    const int m_orb = (sym_mode == 1) ? n : ((n - has_c) / sym_mode);
    expand_sym(n, sym_mode, m_orb, has_c, L, q_red, z_full_out);
}

double eval_sat_pen_bbox(int n, const double* x, const double* y, const double* th, double side, double* out_L) {
    GridWorkspace gw;
    return compute_sat_bbox_and_pen(n, x, y, th, side, gw, out_L);
}

void batch_ils_solve(
    int batch_size, int n, int sym_mode,
    double* q_red_batch,       // [batch_size * dim], modified in-place to final/best reduced state
    double* q_feas_batch,      // [batch_size * dim], best zero-overlap reduced state if found
    int n_L_steps, const double* L_sched_batch,  // [batch_size * n_L_steps]
    const int* ils_iters_per_L,
    double a_start, int n_melt_stages,
    int n_threads, uint64_t base_seed,
    double* out_best_L_feas, double* out_E, double* out_pen, double* out_L_bbox
) {
    const int threads = std::max(1, std::min(7, n_threads));
    const int has_c = (sym_mode == 1) ? 0 : (n % sym_mode);
    const int m_orb = (sym_mode == 1) ? n : ((n - has_c) / sym_mode);
    const int dim = 3 * m_orb + has_c;
    std::atomic_int next_idx{0};

    auto worker_fn = [&]() {
        SymLbfgsWorkspace ws;
        while (true) {
            const int b = next_idx.fetch_add(1, std::memory_order_relaxed);
            if (b >= batch_size) break;
            double bL_feas = 1e18, e = 0.0, p = 0.0, lb = 0.0;
            run_ils_trajectory(
                n, sym_mode,
                q_red_batch + static_cast<size_t>(b) * dim,
                n_L_steps, L_sched_batch + static_cast<size_t>(b) * n_L_steps,
                ils_iters_per_L, a_start, n_melt_stages,
                base_seed + static_cast<uint64_t>(b) * 1000003ULL + 13ULL,
                ws, q_feas_batch + static_cast<size_t>(b) * dim,
                &bL_feas, &e, &p, &lb
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
