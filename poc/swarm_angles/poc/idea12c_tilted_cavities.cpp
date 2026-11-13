// Idea 12C: No-Snap a-Homotopy on f_W=0.75 Slabs & L-Corners,
// Exact C4-Reduced Pinched 4-Edge Frames (W = P + 0.76 on n=401),
// and Unsnapped 4frame_buf / Lrim_buf / Seam Melts (n=401, 626).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <omp.h>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

using namespace std;

static constexpr double PI = 3.14159265358979323846;

struct Sq {
    double x, y, th;
};

struct RNG {
    uint64_t s;
    explicit RNG(uint64_t seed) : s(seed ? seed : 88172645463325252ULL) {}
    inline uint64_t next() {
        uint64_t x = s;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        return s = x;
    }
    inline double u01() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
    inline double sym() { return u01() * 2.0 - 1.0; }
    inline int irand(int n) { return (int)(next() % (uint64_t)n); }
};

inline double sat_pen_pair(double xi, double yi, double ci, double si, double hxi,
                           double xj, double yj, double cj, double sj, double hxj,
                           double half_side) {
    double dx = xj - xi;
    if (fabs(dx) >= hxi + hxj) return 0.0;
    double dy = yj - yi;
    if (fabs(dy) >= hxi + hxj) return 0.0;
    double rad = half_side * (1.0 + fabs(ci * cj + si * sj) + fabs(ci * sj - si * cj));
    double p1 = rad - fabs(dx * ci + dy * si);
    if (p1 <= 0.0) return 0.0;
    double p2 = rad - fabs(-dx * si + dy * ci);
    if (p2 <= 0.0) return 0.0;
    double p3 = rad - fabs(dx * cj + dy * sj);
    if (p3 <= 0.0) return 0.0;
    double p4 = rad - fabs(-dx * sj + dy * cj);
    if (p4 <= 0.0) return 0.0;
    return min(min(p1, p2), min(p3, p4));
}

pair<double, double> exact_max_pen_and_L(const vector<Sq>& sq, double side = 1.0) {
    int n = (int)sq.size();
    double half = 0.5 * side;
    vector<double> c(n), s(n), hx(n);
    double minx = 1e18, maxx = -1e18, miny = 1e18, maxy = -1e18;
    for (int i = 0; i < n; ++i) {
        c[i] = cos(sq[i].th);
        s[i] = sin(sq[i].th);
        hx[i] = half * (fabs(c[i]) + fabs(s[i]));
        minx = min(minx, sq[i].x - hx[i]);
        maxx = max(maxx, sq[i].x + hx[i]);
        miny = min(miny, sq[i].y - hx[i]);
        maxy = max(maxy, sq[i].y + hx[i]);
    }
    double max_p = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double p = sat_pen_pair(sq[i].x, sq[i].y, c[i], s[i], hx[i],
                                    sq[j].x, sq[j].y, c[j], s[j], hx[j], half);
            if (p > max_p) max_p = p;
        }
    }
    return {max_p, max(maxx - minx, maxy - miny)};
}

struct RectSpatialSA {
    int n = 0;
    double Lx = 1.0, Ly = 1.0, side = 1.0, half = 0.5;
    double l_corner_w = -1.0;  // If > 0: inner forbidden quadrant x > W && y > W
    double cell_sz = 1.5;
    int gx_n = 4, gy_n = 4;
    vector<Sq> sq;
    vector<double> c, s, hx;
    vector<int> cell_id;
    vector<vector<int>> cells;
    vector<uint8_t> is_active;
    vector<uint8_t> tilt_lock;  // 1: keep tilted (never snap to 0, clamp |sin(2th)| when a<0.992)
    vector<int> active_ids;
    vector<int> tilted_active_ids;
    double total_E = 0.0;

    void init(const vector<Sq>& init_sq, double bx, double by, double sq_side,
              const vector<uint8_t>* active_mask = nullptr,
              const vector<uint8_t>* tlock = nullptr,
              double l_cut = -1.0) {
        sq = init_sq;
        n = (int)sq.size();
        Lx = bx; Ly = by;
        side = sq_side; half = 0.5 * side;
        l_corner_w = l_cut;
        cell_sz = max(1.5, 1.5 * side);
        gx_n = max(4, (int)ceil(Lx / cell_sz) + 2);
        gy_n = max(4, (int)ceil(Ly / cell_sz) + 2);
        c.resize(n); s.resize(n); hx.resize(n);
        cell_id.assign(n, -1);
        cells.assign(gx_n * gy_n, {});
        is_active.assign(n, 1);
        tilt_lock.assign(n, 0);
        active_ids.clear();
        tilted_active_ids.clear();
        for (int i = 0; i < n; ++i) {
            if (active_mask && (*active_mask)[i] == 0) is_active[i] = 0;
            if (tlock) tilt_lock[i] = (*tlock)[i];
            if (is_active[i]) {
                active_ids.push_back(i);
                if (tilt_lock[i]) tilted_active_ids.push_back(i);
            }
            c[i] = cos(sq[i].th); s[i] = sin(sq[i].th);
            hx[i] = half * (fabs(c[i]) + fabs(s[i]));
            int cid = get_cid(sq[i].x, sq[i].y);
            cell_id[i] = cid;
            cells[cid].push_back(i);
        }
        recompute_E();
    }

    inline int get_cid(double x, double y) const {
        int gx = max(0, min(gx_n - 1, (int)(x / cell_sz) + 1));
        int gy = max(0, min(gy_n - 1, (int)(y / cell_sz) + 1));
        return gy * gx_n + gx;
    }

    inline double wall_E_val(double xi, double yi, double hxi, uint8_t tl) const {
        double e = 0.0, v;
        if ((v = hxi - xi) > 0.0) e += v * v;
        if ((v = xi + hxi - Lx) > 0.0) e += v * v;
        if ((v = hxi - yi) > 0.0) e += v * v;
        if ((v = yi + hxi - Ly) > 0.0) e += v * v;
        if (l_corner_w > 0.0) {
            double dx_in = (xi + hxi) - l_corner_w;
            double dy_in = (yi + hxi) - l_corner_w;
            if (dx_in > 0.0 && dy_in > 0.0) {
                double d = min(dx_in, dy_in);
                e += d * d;
            }
        }
        (void)tl;
        return e;
    }

    inline double orient_E(double th, uint8_t tl) const {
        if (!tl || side >= 0.992) return 0.0;
        double s2 = fabs(sin(2.0 * th));
        if (s2 < 0.30) {
            double d = 0.30 - s2;
            return 0.08 * d * d;
        }
        return 0.0;
    }

    double local_E(int i, double xi, double yi, double th, double ci, double si, double hxi, int skip_j = -1) const {
        double e = wall_E_val(xi, yi, hxi, tilt_lock[i]) + orient_E(th, tilt_lock[i]);
        int gx = max(0, min(gx_n - 1, (int)(xi / cell_sz) + 1));
        int gy = max(0, min(gy_n - 1, (int)(yi / cell_sz) + 1));
        for (int dy = -1; dy <= 1; ++dy) {
            int yy = gy + dy;
            if (yy < 0 || yy >= gy_n) continue;
            for (int dx = -1; dx <= 1; ++dx) {
                int xx = gx + dx;
                if (xx < 0 || xx >= gx_n) continue;
                for (int j : cells[yy * gx_n + xx]) {
                    if (j == i || j == skip_j) continue;
                    double p = sat_pen_pair(xi, yi, ci, si, hxi,
                                            sq[j].x, sq[j].y, c[j], s[j], hx[j], half);
                    if (p > 0.0) e += p * p;
                }
            }
        }
        return e;
    }

    double recompute_E() {
        double e = 0.0;
        for (int i = 0; i < n; ++i) {
            e += wall_E_val(sq[i].x, sq[i].y, hx[i], tilt_lock[i]) + orient_E(sq[i].th, tilt_lock[i]);
            int gx = max(0, min(gx_n - 1, (int)(sq[i].x / cell_sz) + 1));
            int gy = max(0, min(gy_n - 1, (int)(sq[i].y / cell_sz) + 1));
            for (int dy = -1; dy <= 1; ++dy) {
                int yy = gy + dy;
                if (yy < 0 || yy >= gy_n) continue;
                for (int dx = -1; dx <= 1; ++dx) {
                    int xx = gx + dx;
                    if (xx < 0 || xx >= gx_n) continue;
                    for (int j : cells[yy * gx_n + xx]) {
                        if (j <= i) continue;
                        double p = sat_pen_pair(sq[i].x, sq[i].y, c[i], s[i], hx[i],
                                                sq[j].x, sq[j].y, c[j], s[j], hx[j], half);
                        if (p > 0.0) e += p * p;
                    }
                }
            }
        }
        total_E = e;
        return e;
    }

    inline void update_sq(int i, double nx, double ny, double nth, double nc, double ns, double nhx) {
        sq[i] = {nx, ny, nth};
        c[i] = nc; s[i] = ns; hx[i] = nhx;
        int ncid = get_cid(nx, ny);
        int ocid = cell_id[i];
        if (ncid != ocid) {
            auto& v = cells[ocid];
            for (size_t k = 0; k < v.size(); ++k) {
                if (v[k] == i) { v[k] = v.back(); v.pop_back(); break; }
            }
            cells[ncid].push_back(i);
            cell_id[i] = ncid;
        }
    }

    void run_sa(int iters, double T0, double T1, double step_xy, double step_th, RNG& rng) {
        int m_act = (int)active_ids.size();
        if (m_act == 0) return;
        double invN = 1.0 / max(1, iters);
        double log_ratio = log(max(1e-12, T1) / max(1e-12, T0));
        double best_E = total_E;
        vector<Sq> best_sq = sq;

        for (int it = 0; it < iters; ++it) {
            double frac = it * invN;
            double T = T0 * exp(frac * log_ratio);
            double sc = 1.0 - 0.85 * frac;

            if (rng.u01() < 0.06 && m_act >= 2) {
                int i = active_ids[rng.irand(m_act)];
                int j = active_ids[rng.irand(m_act)];
                if (i == j || tilt_lock[i] != tilt_lock[j]) continue;
                double pij_old = sat_pen_pair(sq[i].x, sq[i].y, c[i], s[i], hx[i],
                                              sq[j].x, sq[j].y, c[j], s[j], hx[j], half);
                double old_e = local_E(i, sq[i].x, sq[i].y, sq[i].th, c[i], s[i], hx[i], j)
                             + local_E(j, sq[j].x, sq[j].y, sq[j].th, c[j], s[j], hx[j], i) + pij_old * pij_old;
                double nxi = sq[j].x, nyi = sq[j].y, nthi = sq[i].th;
                double nxj = sq[i].x, nyj = sq[i].y, nthj = sq[j].th;
                double nci = cos(nthi), nsi = sin(nthi), nhxi = half * (fabs(nci) + fabs(nsi));
                double ncj = cos(nthj), nsj = sin(nthj), nhxj = half * (fabs(ncj) + fabs(nsj));
                double pij_new = sat_pen_pair(nxi, nyi, nci, nsi, nhxi, nxj, nyj, ncj, nsj, nhxj, half);
                double new_e = local_E(i, nxi, nyi, nthi, nci, nsi, nhxi, j)
                             + local_E(j, nxj, nyj, nthj, ncj, nsj, nhxj, i) + pij_new * pij_new;
                double dE = new_e - old_e;
                if (dE <= 0.0 || rng.u01() < exp(-dE / T)) {
                    update_sq(i, nxi, nyi, nthi, nci, nsi, nhxi);
                    update_sq(j, nxj, nyj, nthj, ncj, nsj, nhxj);
                    total_E += dE;
                    if (total_E < best_E) { best_E = total_E; best_sq = sq; }
                }
                continue;
            }

            int i = active_ids[rng.irand(m_act)];
            double old_e = local_E(i, sq[i].x, sq[i].y, sq[i].th, c[i], s[i], hx[i]);
            if (old_e == 0.0 && rng.u01() < 0.65) {
                int j = active_ids[rng.irand(m_act)];
                double ej = local_E(j, sq[j].x, sq[j].y, sq[j].th, c[j], s[j], hx[j]);
                if (ej > 0.0) { i = j; old_e = ej; }
            }

            double nx = sq[i].x + rng.sym() * step_xy * sc;
            double ny = sq[i].y + rng.sym() * step_xy * sc;
            double nth = sq[i].th;
            double mv = rng.u01();
            if (tilt_lock[i]) {
                if (mv < 0.14 && !tilted_active_ids.empty()) {
                    int k = tilted_active_ids[rng.irand((int)tilted_active_ids.size())];
                    nth = (rng.u01() < 0.35) ? (0.5 * PI - sq[k].th) : sq[k].th;
                    nth += rng.sym() * 0.015;
                } else if (mv < 0.72) {
                    nth += rng.sym() * step_th * sc;
                }
            } else {
                if (mv < 0.14) {
                    int k = active_ids[rng.irand(m_act)];
                    double r2 = rng.u01();
                    if (r2 < 0.25) nth = 0.0;
                    else if (r2 < 0.45) nth = 0.5 * PI - sq[k].th + rng.sym() * 0.015;
                    else nth = sq[k].th + rng.sym() * 0.015;
                } else if (mv < 0.72) {
                    nth += rng.sym() * step_th * sc;
                }
            }
            nx = max(0.32, min(Lx - 0.32, nx));
            ny = max(0.32, min(Ly - 0.32, ny));
            if (l_corner_w > 0.0 && nx > l_corner_w - 0.32 && ny > l_corner_w - 0.32) {
                if (nx - l_corner_w > ny - l_corner_w) ny = l_corner_w - 0.35;
                else nx = l_corner_w - 0.35;
            }
            double nc = cos(nth), ns = sin(nth), nhx = half * (fabs(nc) + fabs(ns));
            double new_e = local_E(i, nx, ny, nth, nc, ns, nhx);
            double dE = new_e - old_e;
            if (dE <= 0.0 || rng.u01() < exp(-dE / T)) {
                update_sq(i, nx, ny, nth, nc, ns, nhx);
                total_E += dE;
                if (total_E < best_E) { best_E = total_E; best_sq = sq; }
            }
        }
        if (best_E < total_E) {
            vector<uint8_t> mask = is_active, tl = tilt_lock;
            init(best_sq, Lx, Ly, side, &mask, &tl, l_corner_w);
        }
    }
};

double adam_relax_rect(vector<Sq>& sq, double Lx, double Ly, double side, int max_steps,
                       double lr0 = 0.02, bool freeze_angles = false, double eps_stop = 1e-13,
                       const vector<uint8_t>* active_mask = nullptr, double l_cut = -1.0) {
    int n = (int)sq.size();
    double half = 0.5 * side;
    double cell_sz = max(1.6, 1.6 * side);
    int gn_x = max(4, (int)ceil(Lx / cell_sz) + 2);
    int gn_y = max(4, (int)ceil(Ly / cell_sz) + 2);
    vector<vector<int>> cells(gn_x * gn_y);
    vector<double> m(3 * n, 0.0), v(3 * n, 0.0);
    double b1 = 0.88, b2 = 0.99, eps = 1e-8;
    double b1t = 1.0, b2t = 1.0;

    vector<Sq> best_sq = sq;
    double best_E = 1e18;

    for (int step = 0; step < max_steps; ++step) {
        if (step % 15 == 0) {
            for (auto& cell : cells) cell.clear();
            for (int i = 0; i < n; ++i) {
                int gx = max(0, min(gn_x - 1, (int)(sq[i].x / cell_sz) + 1));
                int gy = max(0, min(gn_y - 1, (int)(sq[i].y / cell_sz) + 1));
                cells[gy * gn_x + gx].push_back(i);
            }
        }
        vector<double> c(n), s(n), hx(n), dhx(n);
        vector<double> gx(n, 0.0), gy(n, 0.0), gth(n, 0.0);
        for (int i = 0; i < n; ++i) {
            c[i] = cos(sq[i].th); s[i] = sin(sq[i].th);
            double sc_c = (c[i] >= 0 ? 1.0 : -1.0);
            double sc_s = (s[i] >= 0 ? 1.0 : -1.0);
            hx[i] = half * (fabs(c[i]) + fabs(s[i]));
            dhx[i] = half * (sc_c * (-s[i]) + sc_s * c[i]);
        }
        double E = 0.0;
        for (int i = 0; i < n; ++i) {
            double v1;
            if ((v1 = hx[i] - sq[i].x) > 0.0) { E += v1 * v1; gx[i] -= 2 * v1; gth[i] += 2 * v1 * dhx[i]; }
            if ((v1 = sq[i].x + hx[i] - Lx) > 0.0) { E += v1 * v1; gx[i] += 2 * v1; gth[i] += 2 * v1 * dhx[i]; }
            if ((v1 = hx[i] - sq[i].y) > 0.0) { E += v1 * v1; gy[i] -= 2 * v1; gth[i] += 2 * v1 * dhx[i]; }
            if ((v1 = sq[i].y + hx[i] - Ly) > 0.0) { E += v1 * v1; gy[i] += 2 * v1; gth[i] += 2 * v1 * dhx[i]; }
            if (l_cut > 0.0) {
                double dx_in = sq[i].x + hx[i] - l_cut;
                double dy_in = sq[i].y + hx[i] - l_cut;
                if (dx_in > 0.0 && dy_in > 0.0) {
                    if (dx_in < dy_in) {
                        E += dx_in * dx_in; gx[i] += 2 * dx_in; gth[i] += 2 * dx_in * dhx[i];
                    } else {
                        E += dy_in * dy_in; gy[i] += 2 * dy_in; gth[i] += 2 * dy_in * dhx[i];
                    }
                }
            }

            int cx0 = max(0, min(gn_x - 1, (int)(sq[i].x / cell_sz) + 1));
            int cy0 = max(0, min(gn_y - 1, (int)(sq[i].y / cell_sz) + 1));
            for (int dyc = -1; dyc <= 1; ++dyc) {
                int yy = cy0 + dyc;
                if (yy < 0 || yy >= gn_y) continue;
                for (int dxc = -1; dxc <= 1; ++dxc) {
                    int xx = cx0 + dxc;
                    if (xx < 0 || xx >= gn_x) continue;
                    for (int j : cells[yy * gn_x + xx]) {
                        if (j <= i) continue;
                        double dx = sq[j].x - sq[i].x;
                        if (fabs(dx) >= hx[i] + hx[j]) continue;
                        double dy = sq[j].y - sq[i].y;
                        if (fabs(dy) >= hx[i] + hx[j]) continue;

                        double a = c[i] * c[j] + s[i] * s[j];
                        double b = c[i] * s[j] - s[i] * c[j];
                        double sa = (a >= 0 ? 1.0 : -1.0);
                        double sb = (b >= 0 ? 1.0 : -1.0);
                        double rad = half * (1.0 + fabs(a) + fabs(b));

                        double p_axes[4] = {
                            dx * c[i] + dy * s[i], -dx * s[i] + dy * c[i],
                            dx * c[j] + dy * s[j], -dx * s[j] + dy * c[j]
                        };
                        double pens[4] = { rad - fabs(p_axes[0]), rad - fabs(p_axes[1]),
                                           rad - fabs(p_axes[2]), rad - fabs(p_axes[3]) };
                        int kmin = 0;
                        if (pens[1] < pens[kmin]) kmin = 1;
                        if (pens[2] < pens[kmin]) kmin = 2;
                        if (pens[3] < pens[kmin]) kmin = 3;
                        double pen = pens[kmin];
                        if (pen <= 0.0) continue;

                        E += pen * pen;
                        double drad_dthi = half * (sa * b - sb * a);
                        double drad_dthj = half * (-sa * b + sb * a);
                        double sgn = (p_axes[kmin] >= 0 ? 1.0 : -1.0);
                        double dpx = 0, dpy = 0, dthi_ax = 0, dthj_ax = 0;
                        if (kmin == 0) { dpx = c[i]; dpy = s[i]; dthi_ax = -dx * s[i] + dy * c[i]; }
                        else if (kmin == 1) { dpx = -s[i]; dpy = c[i]; dthi_ax = -dx * c[i] - dy * s[i]; }
                        else if (kmin == 2) { dpx = c[j]; dpy = s[j]; dthj_ax = -dx * s[j] + dy * c[j]; }
                        else { dpx = -s[j]; dpy = c[j]; dthj_ax = -dx * c[j] - dy * s[j]; }
                        double w2 = 2.0 * pen;
                        gx[i] += w2 * sgn * dpx; gy[i] += w2 * sgn * dpy;
                        gx[j] -= w2 * sgn * dpx; gy[j] -= w2 * sgn * dpy;
                        gth[i] += w2 * (drad_dthi - sgn * dthi_ax);
                        gth[j] += w2 * (drad_dthj - sgn * dthj_ax);
                    }
                }
            }
        }
        if (E < best_E) { best_E = E; best_sq = sq; }
        if (E < eps_stop) break;

        b1t *= b1; b2t *= b2;
        double lr = lr0 * (0.1 + 0.9 * 0.5 * (1.0 + cos(PI * step / max(1, max_steps))));
        for (int i = 0; i < n; ++i) {
            if (active_mask && (*active_mask)[i] == 0) continue;
            double gs[3] = {gx[i], gy[i], freeze_angles ? 0.0 : gth[i] * 0.6};
            double* ptrs[3] = {&sq[i].x, &sq[i].y, &sq[i].th};
            for (int k = 0; k < 3; ++k) {
                int idx = k * n + i;
                m[idx] = b1 * m[idx] + (1.0 - b1) * gs[k];
                v[idx] = b2 * v[idx] + (1.0 - b2) * gs[k] * gs[k];
                double mh = m[idx] / (1.0 - b1t);
                double vh = v[idx] / (1.0 - b2t);
                *ptrs[k] -= lr * mh / (sqrt(vh) + eps);
            }
        }
    }
    sq = best_sq;
    return best_E;
}

double polish_and_shrink_sq(vector<Sq>& sq, double L_start) {
    double cur_L = L_start;
    auto [p0, bbox0] = exact_max_pen_and_L(sq, 1.0);
    if (p0 > 1e-11) {
        adam_relax_rect(sq, cur_L, cur_L, 1.0, 500, 0.018, true, 1e-15);
        adam_relax_rect(sq, cur_L, cur_L, 1.0, 1500, 0.013, false, 1e-20);
        auto [p1, b1] = exact_max_pen_and_L(sq, 1.0);
        if (p1 > 1e-11) return 1e18;
        cur_L = min(cur_L, b1);
    } else {
        cur_L = bbox0;
    }
    vector<Sq> best_sq = sq;
    double best_L = cur_L;
    auto shift_origin = [&](vector<Sq>& v) {
        double mx = 1e18, my = 1e18;
        for (const auto& q : v) {
            double hx = 0.5 * (fabs(cos(q.th)) + fabs(sin(q.th)));
            mx = min(mx, q.x - hx); my = min(my, q.y - hx);
        }
        for (auto& q : v) { q.x -= mx; q.y -= my; }
    };
    shift_origin(best_sq);

    double step = 0.002;
    for (int round = 0; round < 18; ++round) {
        vector<Sq> trial = best_sq;
        double target_L = best_L - step;
        double halfL = 0.5 * best_L;
        double sc = (target_L - 1.0) / max(1.0, best_L - 1.0);
        for (auto& q : trial) {
            q.x = 0.5 * target_L + (q.x - halfL) * sc;
            q.y = 0.5 * target_L + (q.y - halfL) * sc;
        }
        adam_relax_rect(trial, target_L, target_L, 1.0, 750, 0.01, false, 1e-22);
        auto [p, b] = exact_max_pen_and_L(trial, 1.0);
        if (p <= 1e-13 && b < best_L - 1e-12) {
            best_L = b; best_sq = trial; shift_origin(best_sq); step *= 1.25;
        } else {
            step *= 0.45;
        }
        if (step < 1e-10) break;
    }
    sq = best_sq;
    return best_L;
}

vector<Sq> load_merged_n(int target_n) {
    ifstream f("poc/pell_corrugation/data/border-scan/merged.csv");
    string line;
    getline(f, line);
    vector<Sq> res;
    while (getline(f, line)) {
        stringstream ss(line); string tok; vector<string> cols;
        while (getline(ss, tok, ',')) cols.push_back(tok);
        if (cols.size() < 6) continue;
        if (stoi(cols[1]) == target_n) res.push_back({stod(cols[3]), stod(cols[4]), stod(cols[5])});
    }
    if (res.empty()) return res;
    double minx = 1e18, miny = 1e18;
    for (const auto& q : res) {
        double hx = 0.5 * (fabs(cos(q.th)) + fabs(sin(q.th)));
        minx = min(minx, q.x - hx); miny = min(miny, q.y - hx);
    }
    for (auto& q : res) { q.x -= minx; q.y -= miny; }
    return res;
}

void save_csv(const string& path, int n, const vector<Sq>& sq) {
    ofstream out(path);
    out << setprecision(17) << "id,n,square_id,x,y,angle\n";
    for (int i = 0; i < n; ++i) {
        out << i << "," << n << "," << i << "," << sq[i].x << "," << sq[i].y << "," << sq[i].th << "\n";
    }
}

struct ProbeJob {
    string name;
    double Lx, Ly;
    vector<Sq> init_sq;
    vector<uint8_t> mask;
    vector<uint8_t> tilt_lock;
    int sweeps;
    bool global_polish;
    double l_cut = -1.0;
};

pair<double, vector<Sq>> run_homotopy_job(const ProbeJob& job, uint64_t seed) {
    RNG rng(seed);
    vector<Sq> sq = job.init_sq;
    double a_sched[8] = {0.86, 0.91, 0.945, 0.972, 0.988, 0.995, 0.999, 1.0};
    RectSpatialSA sa;
    for (int st = 0; st < 8; ++st) {
        double a = a_sched[st];
        sa.init(sq, job.Lx, job.Ly, a, &job.mask, &job.tilt_lock, job.l_cut);
        double T0 = (st == 0) ? 0.018 : 0.004 * (1.01 - a);
        double T1 = max(1e-8, 5e-5 * (1.01 - a));
        sa.run_sa(job.sweeps, T0, T1, 0.30 * (1.05 - a) + 0.015, 0.42 * (1.05 - a) + 0.025, rng);
        sq = sa.sq;
        adam_relax_rect(sq, job.Lx, job.Ly, a, (st < 6 ? 140 : 320), 0.015 * (1.05 - a) + 0.003,
                        false, 1e-13, &job.mask, job.l_cut);
    }
    double E = adam_relax_rect(sq, job.Lx, job.Ly, 1.0, 1200, 0.012, false, 1e-20,
                               job.global_polish ? nullptr : &job.mask, job.l_cut);
    return {E, sq};
}

// --- Exact C4-Symmetric Homotopy Solver for n = 401 = M^2 + 4*Q ---
struct C4Job {
    string name;
    double L;
    vector<Sq> fixed_core;  // frozen M^2 or M_in^2 core squares (already C4-symmetric)
    vector<Sq> arm_sq;      // Q independent active squares in arm 0
    vector<uint8_t> arm_tlock;
    int sweeps;
};

static inline Sq rot90(const Sq& q, double L, int r) {
    double cx = 0.5 * L, cy = 0.5 * L;
    double dx = q.x - cx, dy = q.y - cy;
    if (r == 0) return q;
    if (r == 1) return {cx - dy, cy + dx, q.th + 0.5 * PI};
    if (r == 2) return {cx - dx, cy - dy, q.th + PI};
    return {cx + dy, cy - dx, q.th + 1.5 * PI};
}

pair<double, vector<Sq>> expand_and_eval_c4(const C4Job& job, const vector<Sq>& arm, double side) {
    vector<Sq> all = job.fixed_core;
    for (int r = 0; r < 4; ++r)
        for (const auto& q : arm) all.push_back(rot90(q, job.L, r));
    int m = (int)all.size();
    double half = 0.5 * side;
    double E = 0.0;
    for (int i = 0; i < m; ++i) {
        double c = cos(all[i].th), s = sin(all[i].th), hx = half * (fabs(c) + fabs(s));
        double v;
        if ((v = hx - all[i].x) > 0) E += v * v;
        if ((v = all[i].x + hx - job.L) > 0) E += v * v;
        if ((v = hx - all[i].y) > 0) E += v * v;
        if ((v = all[i].y + hx - job.L) > 0) E += v * v;
    }
    vector<double> cv(m), sv(m), hxv(m);
    for (int i = 0; i < m; ++i) {
        cv[i] = cos(all[i].th); sv[i] = sin(all[i].th);
        hxv[i] = half * (fabs(cv[i]) + fabs(sv[i]));
    }
    for (int i = 0; i < m; ++i) {
        for (int j = i + 1; j < m; ++j) {
            double p = sat_pen_pair(all[i].x, all[i].y, cv[i], sv[i], hxv[i],
                                    all[j].x, all[j].y, cv[j], sv[j], hxv[j], half);
            if (p > 0.0) E += p * p;
        }
    }
    return {E, all};
}

pair<double, vector<Sq>> run_c4_homotopy(const C4Job& job, uint64_t seed) {
    RNG rng(seed);
    vector<Sq> arm = job.arm_sq;
    int Q = (int)arm.size();
    int M2 = (int)job.fixed_core.size();
    double a_sched[8] = {0.86, 0.91, 0.945, 0.972, 0.988, 0.995, 0.999, 1.0};

    vector<double> fc_c(M2), fc_s(M2), fc_hx(M2);
    for (int i = 0; i < M2; ++i) {
        fc_c[i] = cos(job.fixed_core[i].th);
        fc_s[i] = sin(job.fixed_core[i].th);
        fc_hx[i] = 0.5 * (fabs(fc_c[i]) + fabs(fc_s[i]));
    }

    for (int st = 0; st < 8; ++st) {
        double a = a_sched[st], half = 0.5 * a;
        for (int i = 0; i < M2; ++i) fc_hx[i] = half * (fabs(fc_c[i]) + fabs(fc_s[i]));

        auto q_energy = [&](int k, const Sq& qk) -> double {
            double e = 0.0;
            Sq copies[4];
            double ck[4], sk[4], hxk[4];
            for (int r = 0; r < 4; ++r) {
                copies[r] = rot90(qk, job.L, r);
                ck[r] = cos(copies[r].th); sk[r] = sin(copies[r].th);
                hxk[r] = half * (fabs(ck[r]) + fabs(sk[r]));
                double v;
                if ((v = hxk[r] - copies[r].x) > 0) e += v * v;
                if ((v = copies[r].x + hxk[r] - job.L) > 0) e += v * v;
                if ((v = hxk[r] - copies[r].y) > 0) e += v * v;
                if ((v = copies[r].y + hxk[r] - job.L) > 0) e += v * v;
                if (job.arm_tlock[k] && a < 0.992) {
                    double s2 = fabs(sin(2.0 * copies[r].th));
                    if (s2 < 0.30) { double d = 0.30 - s2; e += 0.08 * d * d; }
                }
            }
            // Self-rotations of k
            for (int r1 = 0; r1 < 4; ++r1) {
                for (int r2 = r1 + 1; r2 < 4; ++r2) {
                    double p = sat_pen_pair(copies[r1].x, copies[r1].y, ck[r1], sk[r1], hxk[r1],
                                            copies[r2].x, copies[r2].y, ck[r2], sk[r2], hxk[r2], half);
                    if (p > 0) e += p * p;
                }
            }
            // Fixed core (using rotated check: copy 0 against all core is 1/4 of sum -> scale by 4)
            for (int c_i = 0; c_i < M2; ++c_i) {
                double p = sat_pen_pair(copies[0].x, copies[0].y, ck[0], sk[0], hxk[0],
                                        job.fixed_core[c_i].x, job.fixed_core[c_i].y,
                                        fc_c[c_i], fc_s[c_i], fc_hx[c_i], half);
                if (p > 0) e += 4.0 * p * p;
            }
            // Other arm squares j != k (copies[0] vs all 4 rotations of j, x4 symmetry)
            for (int j = 0; j < Q; ++j) {
                if (j == k) continue;
                for (int r = 0; r < 4; ++r) {
                    Sq qj = rot90(arm[j], job.L, r);
                    double cj = cos(qj.th), sj = sin(qj.th), hxj = half * (fabs(cj) + fabs(sj));
                    double p = sat_pen_pair(copies[0].x, copies[0].y, ck[0], sk[0], hxk[0],
                                            qj.x, qj.y, cj, sj, hxj, half);
                    if (p > 0) e += 4.0 * p * p;
                }
            }
            return e;
        };

        double T0 = (st == 0) ? 0.05 : 0.01 * (1.01 - a);
        double T1 = max(1e-8, 1e-4 * (1.01 - a));
        double log_ratio = log(T1 / T0);
        double sxy = 0.28 * (1.05 - a) + 0.015;
        double sth = 0.40 * (1.05 - a) + 0.025;
        vector<Sq> best_arm = arm;
        double best_full = 1e18;

        for (int it = 0; it < job.sweeps; ++it) {
            double frac = it / (double)max(1, job.sweeps);
            double T = T0 * exp(frac * log_ratio);
            double sc = 1.0 - 0.85 * frac;
            int k = rng.irand(Q);
            double old_e = q_energy(k, arm[k]);
            if (old_e == 0.0 && rng.u01() < 0.65) {
                int j = rng.irand(Q);
                double ej = q_energy(j, arm[j]);
                if (ej > 0.0) { k = j; old_e = ej; }
            }
            Sq nq = arm[k];
            nq.x = max(0.32, min(job.L - 0.32, nq.x + rng.sym() * sxy * sc));
            nq.y = max(0.32, min(job.L - 0.32, nq.y + rng.sym() * sxy * sc));
            double mv = rng.u01();
            if (job.arm_tlock[k] && mv < 0.14) {
                int k2 = rng.irand(Q);
                nq.th = arm[k2].th + rng.sym() * 0.015;
            } else if (mv < 0.72) {
                nq.th += rng.sym() * sth * sc;
            }
            double new_e = q_energy(k, nq);
            double dE = new_e - old_e;
            if (dE <= 0.0 || rng.u01() < exp(-dE / T)) {
                arm[k] = nq;
            }
            if ((it + 1) % max(1, job.sweeps / 5) == 0) {
                auto [ef, _] = expand_and_eval_c4(job, arm, a);
                if (ef < best_full) { best_full = ef; best_arm = arm; }
            }
        }
        auto [ef_end, _] = expand_and_eval_c4(job, arm, a);
        if (best_full < ef_end) arm = best_arm;

        // Joint unconstrained Adam on expanded configuration at current a, then pull arm back
        auto [__, full_sq] = expand_and_eval_c4(job, arm, a);
        vector<uint8_t> mask(full_sq.size(), 0);
        for (int i = M2; i < M2 + Q; ++i) mask[i] = 1;
        // Relax ALL 4 arms symmetrically by relaxing full_sq then averaging back arm 0
        vector<uint8_t> mask_all(full_sq.size(), 1);
        for (int i = 0; i < M2; ++i) mask_all[i] = 0;
        adam_relax_rect(full_sq, job.L, job.L, a, (st < 6 ? 120 : 250), 0.014 * (1.05 - a) + 0.003,
                        false, 1e-13, &mask_all);
        for (int i = 0; i < Q; ++i) arm[i] = full_sq[M2 + i];
    }
    auto [_, full_sq] = expand_and_eval_c4(job, arm, 1.0);
    double E = adam_relax_rect(full_sq, job.L, job.L, 1.0, 1400, 0.012, false, 1e-20, nullptr);
    return {E, full_sq};
}

int main() {
    omp_set_num_threads(7);
    auto t0 = chrono::steady_clock::now();

    cout << "=== 12C PART 1: Pure f_W=0.75 Slabs (P+0.75)x(H+0.25) & L-Corner Capacity (No 0-Snap) ===\n" << flush;
    for (int P : {3, 4, 5, 6, 7}) {
        for (int H : {14, 18, 22}) {
            double Wx = P + 0.75, Hy = H + 0.25;
            int n_slab = P * H + 1;
            double best_E = 1e18;
            #pragma omp parallel for schedule(dynamic, 1)
            for (int w = 0; w < 14; ++w) {
                RNG rng(120001ULL + P * 997 + H * 37 + w * 7919);
                vector<Sq> sq;
                double th = max(0.20, min(0.75, acos(min(0.99, (P + 0.15) / (P + 1.0))) + (w % 7 - 3) * 0.035));
                double c = cos(th), s = sin(th);
                int wcols = P + 1;
                double u0 = 0.5 * (c + s);
                double du = (Wx - (c + s)) / max(1, wcols - 1);
                double pitch = 1.0 / c;
                int rows_needed = (n_slab + wcols - 1) / wcols;
                for (int r = 0; r < rows_needed && (int)sq.size() < n_slab; ++r) {
                    for (int col = 0; col < wcols && (int)sq.size() < n_slab; ++col) {
                        double px = u0 + col * du;
                        double py = 0.65 + r * pitch + (col - 0.5 * (wcols - 1)) * (du * tan(th) * 0.6);
                        py = max(0.55, min(Hy - 0.55, py));
                        sq.push_back({px, py, ((w % 2 == 0) ? th : -th) + rng.sym() * 0.06});
                    }
                }
                vector<uint8_t> mask(n_slab, 1), tl(n_slab, 1);
                ProbeJob job{"slab75", Wx, Hy, sq, mask, tl, 240000, false, -1.0};
                auto [E, _] = run_homotopy_job(job, 220007ULL + P * 131 + H * 19 + w * 911);
                #pragma omp critical
                { best_E = min(best_E, E); }
            }
            cout << "  Slab75 P=" << P << " (" << Wx << "x" << Hy << ", need " << n_slab << "): min_E="
                 << scientific << setprecision(3) << best_E << fixed << "\n" << flush;
        }
        // Cyclic L-Corner of arm width W = P + 0.75 inside L = 18 + 0.502 (need 2*P*18 - P^2 + 1? No: 18^2 - (18-P-1)^2 + 1)
        int H_arm = 18;
        double Wc = P + 0.75, Lb = H_arm + 0.502;
        int n_lcorner = H_arm * H_arm + 1 - (H_arm - P - 1) * (H_arm - P - 1);
        double best_LE = 1e18;
        #pragma omp parallel for schedule(dynamic, 1)
        for (int w = 0; w < 14; ++w) {
            RNG rng(330001ULL + P * 997 + w * 7919);
            vector<Sq> sq;
            double th = max(0.20, min(0.75, acos(min(0.99, (P + 0.15) / (P + 1.0))) + (w % 7 - 3) * 0.04));
            for (int k = 0; k < n_lcorner; ++k) {
                bool bottom = (k % 2 == 0);
                double u = (k / 2 + 0.5) / max(1, n_lcorner / 2);
                double px = bottom ? (0.5 + u * (Lb - 1.0)) : (0.5 + rng.u01() * (Wc - 1.0));
                double py = bottom ? (0.5 + rng.u01() * (Wc - 1.0)) : (0.5 + u * (Lb - 1.0));
                sq.push_back({px, py, (bottom ? th : 0.5 * PI - th) + rng.sym() * 0.12});
            }
            vector<uint8_t> mask(n_lcorner, 1), tl(n_lcorner, 1);
            ProbeJob job{"lcut", Lb, Lb, sq, mask, tl, 280000, false, Wc};
            auto [E, _] = run_homotopy_job(job, 440007ULL + P * 131 + w * 911);
            #pragma omp critical
            { best_LE = min(best_LE, E); }
        }
        cout << "  LCorner75 P=" << P << " (W=" << Wc << ", L=" << Lb << ", need " << n_lcorner << "): min_E="
             << scientific << setprecision(3) << best_LE << fixed << "\n" << flush;
    }

    cout << "\n=== 12C PART 2: Exact C4-Symmetric Pinched Frame & Tilted Core on n=401 ===\n" << flush;
    double L401_a = 20.5345339, L401_b = 20.5300;
    double win401_L = 20.535533905933;
    vector<Sq> win401_sq;
    string win401_src = "none";

    for (int M : {15, 13, 11, 9, 7, 5, 1}) {
        int Q = (401 - M * M) / 4;
        double best_E = 1e18;
        for (int core_mode = 0; core_mode < 2; ++core_mode) {
            if (core_mode == 1 && (M < 5 || M > 11)) continue;  // 45deg core only M in {5,7,9,11}
            double best_mode_E = 1e18;
            #pragma omp parallel for schedule(dynamic, 1)
            for (int w = 0; w < 14; ++w) {
                double Lt = (w < 10) ? L401_a : L401_b;
                RNG rng(550001ULL + M * 1009 + core_mode * 97 + w * 7919);
                vector<Sq> core;
                if (core_mode == 0) {
                    // Centered M x M upright core
                    double off = 0.5 * (Lt - M);
                    for (int j = 0; j < M; ++j)
                        for (int i = 0; i < M; ++i)
                            core.push_back({off + i + 0.5, off + j + 0.5, 0.0});
                } else {
                    // Centered M x M @ 45deg diamond
                    double cx = 0.5 * Lt, cy = 0.5 * Lt;
                    double d = 1.0 / sqrt(2.0);
                    double halfM = 0.5 * (M - 1);
                    for (int j = 0; j < M; ++j) {
                        for (int i = 0; i < M; ++i) {
                            double u = i - halfM, v = j - halfM;
                            core.push_back({cx + (u - v) * d, cy + (u + v) * d, 0.25 * PI});
                        }
                    }
                }
                double W_arm = 0.5 * (Lt - M);
                double th = (18.0 + (w % 7) * 4.0) * PI / 180.0;
                vector<Sq> arm;
                vector<uint8_t> tlock(Q, 1);
                // Seed Q squares along bottom-left-to-bottom-right arm [0, Lt - W_arm] x [0, W_arm]
                int wcols = max(1, (int)round(W_arm));
                for (int q = 0; q < Q; ++q) {
                    double frac = (q + 0.5) / Q;
                    double px = 0.5 + frac * max(1.0, Lt - W_arm - 0.5);
                    double py = 0.5 + ((q % wcols) + 0.5) / wcols * max(0.5, W_arm - 1.0);
                    if (core_mode == 1) {
                        // Around diamond corner [0..Lt/2] x [0..Lt/2]
                        double r = sqrt((q + 0.5) / Q);
                        double a_ang = 0.5 * PI * rng.u01();
                        px = 0.6 + 0.45 * Lt * (1.0 - 0.45 * r) * cos(a_ang);
                        py = 0.6 + 0.45 * Lt * (1.0 - 0.45 * r) * sin(a_ang);
                    }
                    arm.push_back({max(0.5, min(Lt - 0.5, px)), max(0.5, min(Lt - 0.5, py)),
                                   (core_mode == 1 && w % 2 == 1 ? 0.0 : th) + rng.sym() * 0.12});
                    if (core_mode == 1 && w % 2 == 1) tlock[q] = 0;
                }
                C4Job job{"c4", Lt, core, arm, tlock, 160000};
                auto [E, rsq] = run_c4_homotopy(job, 660001ULL + M * 131 + w * 911);
                auto [mp, mL] = exact_max_pen_and_L(rsq, 1.0);
                #pragma omp critical
                {
                    best_mode_E = min(best_mode_E, E);
                    best_E = min(best_E, E);
                    if (mp <= 1e-11 || E < 1e-9) {
                        vector<Sq> pol = rsq;
                        double wL = polish_and_shrink_sq(pol, Lt);
                        if (wL < win401_L - 1e-11) {
                            win401_L = wL; win401_sq = pol;
                            win401_src = "C4_M" + to_string(M) + "_m" + to_string(core_mode);
                        }
                    }
                }
            }
            cout << "  C4 M=" << M << " mode=" << (core_mode == 0 ? "upright" : "45deg  ")
                 << " (Q=" << Q << "): min_E=" << scientific << setprecision(3) << best_mode_E << fixed << "\n" << flush;
        }
    }

    cout << "\n=== 12C PART 3: Unsnapped 4frame_buf, Lrim_buf & Melted Corridors (401 & 626) ===\n" << flush;
    vector<string> fam_names;
    vector<ProbeJob> jobs3;
    auto add_b = [&](const string& nm, int cnt, auto fn) {
        fam_names.push_back(nm);
        for (int w = 0; w < cnt; ++w) jobs3.push_back(fn(nm, w));
    };

    double f401 = L401_a - 20.0;
    for (int W : {3, 4, 5, 6}) {
        add_b("4frame_ns_W" + to_string(W), 7, [&](const string& nm, int w) {
            RNG rng(7100 + W * 31 + w);
            vector<Sq> sq; vector<uint8_t> mask, tl;
            double th = (18.0 + w * 3.8) * PI / 180.0;
            // Core M_even = 20 - 2*W? Wait: to get W = P + 0.76, freeze odd core (19 - 2*(W-1)) = 21 - 2*W
            int M_core = 21 - 2 * W;
            int lo = W - 1, hi = lo + M_core;
            double off_c = 0.5 * (L401_a - M_core) - lo;
            for (int j = 0; j < 20; ++j) for (int i = 0; i < 20; ++i) {
                bool in_core = (i >= lo && i < hi && j >= lo && j < hi);
                bool deep = (i >= lo + 1 && i < hi - 1 && j >= lo + 1 && j < hi - 1);
                double px = in_core ? (i + off_c + 0.5) : (i + 0.5) * (L401_a / 20.0);
                double py = in_core ? (j + off_c + 0.5) : (j + 0.5) * (L401_a / 20.0);
                double a = in_core ? 0.0 : (((i < lo && j >= lo) || (i >= hi && j < hi)) ? (th + 0.5 * PI) : th);
                sq.push_back({px, py, a + (in_core ? 0.0 : rng.sym() * 0.1)});
                mask.push_back(deep ? 0 : 1);
                tl.push_back(in_core ? 0 : 1);
            }
            sq.push_back({0.5 * W, 0.5 * L401_a, th + 0.5 * PI});
            mask.push_back(1); tl.push_back(1);
            return ProbeJob{nm, L401_a, L401_a, sq, mask, tl, 350000, true, -1.0};
        });
    }

    auto s401 = load_merged_n(401);
    if (s401.size() == 401) {
        add_b("401_seam_melt", 14, [&](const string& nm, int w) {
            RNG rng(8200 + w * 97);
            vector<Sq> sq = s401;
            double sc = (L401_a - 1e-4) / exact_max_pen_and_L(sq, 1.0).second;
            vector<uint8_t> mask(401, 1), tl(401, 0);
            for (int i = 0; i < 401; ++i) {
                sq[i].x *= sc; sq[i].y *= sc;
                // Freeze deep SW & NE upright corners (far from diagonal)
                bool deep_sw = (sq[i].x < 5.2 && sq[i].y < 5.2);
                bool deep_ne = (sq[i].x > 15.3 && sq[i].y > 15.3);
                if (deep_sw || deep_ne) {
                    mask[i] = 0;
                } else {
                    double deg = fmod(fabs(sq[i].th) * 180.0 / PI, 90.0);
                    if (deg > 10 && deg < 80) {
                        tl[i] = 1;
                        sq[i].th += ((w % 7) - 3) * 0.08 + rng.sym() * 0.04;
                    } else {
                        sq[i].th += rng.sym() * 0.08;
                    }
                    sq[i].x += rng.sym() * 0.14;
                    sq[i].y += rng.sym() * 0.14;
                }
            }
            return ProbeJob{nm, L401_a, L401_a, sq, mask, tl, 420000, true, -1.0};
        });
    }

    auto s626 = load_merged_n(626);
    double L626_t = 25.5198;
    double win626_L = 25.520815280171;
    vector<Sq> win626_sq;
    if (s626.size() == 626) {
        add_b("626_seam_melt", 14, [&](const string& nm, int w) {
            RNG rng(9300 + w * 97);
            vector<Sq> sq = s626;
            double sc = (L626_t - 1e-4) / exact_max_pen_and_L(sq, 1.0).second;
            vector<uint8_t> mask(626, 1), tl(626, 0);
            double cx = 0.5 * L626_t, cy = 0.5 * L626_t;
            for (int i = 0; i < 626; ++i) {
                sq[i].x *= sc; sq[i].y *= sc;
                double r = hypot(sq[i].x - cx, sq[i].y - cy);
                double deg = fmod(fabs(sq[i].th) * 180.0 / PI, 90.0);
                bool deep_core = (r < 4.8);
                bool deep_corner = (sq[i].x < 4.2 && sq[i].y < 4.2) || (sq[i].x > L626_t - 4.2 && sq[i].y > L626_t - 4.2)
                                || (sq[i].x < 4.2 && sq[i].y > L626_t - 4.2) || (sq[i].x > L626_t - 4.2 && sq[i].y < 4.2);
                if (deep_core || deep_corner) {
                    mask[i] = 0;
                } else {
                    if (deg > 10 && deg < 80) {
                        tl[i] = 1;
                        sq[i].th += ((w % 7) - 3) * 0.06 + rng.sym() * 0.04;
                    } else {
                        sq[i].th += rng.sym() * 0.08;
                    }
                    sq[i].x += rng.sym() * 0.14;
                    sq[i].y += rng.sym() * 0.14;
                }
            }
            return ProbeJob{nm, L626_t, L626_t, sq, mask, tl, 380000, true, -1.0};
        });
    }

    map<string, double> fam_E;
    #pragma omp parallel for schedule(dynamic, 1)
    for (int ji = 0; ji < (int)jobs3.size(); ++ji) {
        auto [E, sq] = run_homotopy_job(jobs3[ji], 800001ULL + ji * 7919ULL);
        auto [mp, mL] = exact_max_pen_and_L(sq, 1.0);
        #pragma omp critical
        {
            const string& nm = jobs3[ji].name;
            if (!fam_E.count(nm) || E < fam_E[nm]) fam_E[nm] = E;
            if (mp <= 1e-11 || E < 1e-9) {
                vector<Sq> pol = sq;
                if (sq.size() == 401) {
                    double wL = polish_and_shrink_sq(pol, L401_a);
                    if (wL < win401_L - 1e-11) { win401_L = wL; win401_sq = pol; win401_src = nm; }
                } else if (sq.size() == 626) {
                    double wL = polish_and_shrink_sq(pol, L626_t);
                    if (wL < win626_L - 1e-11) { win626_L = wL; win626_sq = pol; }
                }
            }
        }
    }
    for (const string& fn : fam_names) {
        cout << "  " << setw(16) << left << fn << ": min_E=" << scientific << setprecision(3) << fam_E[fn] << fixed << "\n";
    }

    if (win401_L < 20.535533905933 - 1e-11) {
        system("mkdir -p poc/swarm_angles/data");
        save_csv("poc/swarm_angles/data/cand12c_n401.csv", 401, win401_sq);
        cout << "*** WIN 401 L=" << setprecision(12) << win401_L << " (" << win401_src << ") ***\n";
    }
    if (win626_L < 25.520815280171 - 1e-11) {
        system("mkdir -p poc/swarm_angles/data");
        save_csv("poc/swarm_angles/data/cand12c_n626.csv", 626, win626_sq);
        cout << "*** WIN 626 L=" << setprecision(12) << win626_L << " ***\n";
    }
    double secs = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
    cout << "Idea 12C finished in " << fixed << setprecision(1) << secs << "s\n";
    return 0;
}
