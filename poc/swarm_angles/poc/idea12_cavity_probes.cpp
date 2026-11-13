// Idea 12A: Pure Cavity Capacity Probes (1-Edge Rectangular Slabs, L-Rim, 4-Edge Pinched Frames)
// with Hard vs 2-Layer Boundary-Buffered Homotopy + Deep n=401 Global a-Homotopy.
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
                           double half_side, double min_gap = 0.0) {
    double dx = xj - xi;
    if (fabs(dx) >= hxi + hxj + min_gap) return 0.0;
    double dy = yj - yi;
    if (fabs(dy) >= hxi + hxj + min_gap) return 0.0;
    double rad = half_side * (1.0 + fabs(ci * cj + si * sj) + fabs(ci * sj - si * cj)) + min_gap;
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
                                    sq[j].x, sq[j].y, c[j], s[j], hx[j], half, 0.0);
            if (p > max_p) max_p = p;
        }
    }
    return {max_p, max(maxx - minx, maxy - miny)};
}

struct RectSpatialSA {
    int n = 0;
    double Lx = 1.0, Ly = 1.0, side = 1.0, half = 0.5;
    double cell_sz = 1.5;
    int gx_n = 4, gy_n = 4;
    vector<Sq> sq;
    vector<double> c, s, hx;
    vector<int> cell_id;
    vector<vector<int>> cells;
    vector<uint8_t> is_active;
    vector<int> active_ids;
    double total_E = 0.0;

    void init(const vector<Sq>& init_sq, double bx, double by, double sq_side,
              const vector<uint8_t>* active_mask = nullptr) {
        sq = init_sq;
        n = (int)sq.size();
        Lx = bx; Ly = by;
        side = sq_side;
        half = 0.5 * side;
        cell_sz = max(1.5, 1.5 * side);
        gx_n = max(4, (int)ceil(Lx / cell_sz) + 2);
        gy_n = max(4, (int)ceil(Ly / cell_sz) + 2);
        c.resize(n); s.resize(n); hx.resize(n);
        cell_id.assign(n, -1);
        cells.assign(gx_n * gy_n, {});
        is_active.assign(n, 1);
        active_ids.clear();
        for (int i = 0; i < n; ++i) {
            if (active_mask && (*active_mask)[i] == 0) is_active[i] = 0;
            if (is_active[i]) active_ids.push_back(i);
            c[i] = cos(sq[i].th);
            s[i] = sin(sq[i].th);
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

    inline double wall_E_val(double xi, double yi, double hxi) const {
        double e = 0.0, v;
        if ((v = hxi - xi) > 0.0) e += v * v;
        if ((v = xi + hxi - Lx) > 0.0) e += v * v;
        if ((v = hxi - yi) > 0.0) e += v * v;
        if ((v = yi + hxi - Ly) > 0.0) e += v * v;
        return e;
    }

    double local_E(int i, double xi, double yi, double ci, double si, double hxi, int skip_j = -1) const {
        double e = wall_E_val(xi, yi, hxi);
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
                                            sq[j].x, sq[j].y, c[j], s[j], hx[j], half, 0.0);
                    if (p > 0.0) e += p * p;
                }
            }
        }
        return e;
    }

    double recompute_E() {
        double e = 0.0;
        for (int i = 0; i < n; ++i) {
            e += wall_E_val(sq[i].x, sq[i].y, hx[i]);
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
                                                sq[j].x, sq[j].y, c[j], s[j], hx[j], half, 0.0);
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

    void run_sa(int iters, double T0, double T1, double step_xy, double step_th, RNG& rng, bool allow_align = true) {
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

            if (rng.u01() < 0.07 && m_act >= 2) {
                int i = active_ids[rng.irand(m_act)];
                int j = active_ids[rng.irand(m_act)];
                if (i == j) continue;
                double pij_old = sat_pen_pair(sq[i].x, sq[i].y, c[i], s[i], hx[i],
                                              sq[j].x, sq[j].y, c[j], s[j], hx[j], half, 0.0);
                double old_e = local_E(i, sq[i].x, sq[i].y, c[i], s[i], hx[i], j)
                             + local_E(j, sq[j].x, sq[j].y, c[j], s[j], hx[j], i) + pij_old * pij_old;
                bool swap_pos = (rng.u01() < 0.5);
                double nxi = swap_pos ? sq[j].x : sq[i].x, nyi = swap_pos ? sq[j].y : sq[i].y;
                double nthi = swap_pos ? sq[i].th : sq[j].th;
                double nxj = swap_pos ? sq[i].x : sq[j].x, nyj = swap_pos ? sq[i].y : sq[j].y;
                double nthj = swap_pos ? sq[j].th : sq[i].th;
                double nci = cos(nthi), nsi = sin(nthi), nhxi = half * (fabs(nci) + fabs(nsi));
                double ncj = cos(nthj), nsj = sin(nthj), nhxj = half * (fabs(ncj) + fabs(nsj));
                double pij_new = sat_pen_pair(nxi, nyi, nci, nsi, nhxi, nxj, nyj, ncj, nsj, nhxj, half, 0.0);
                double new_e = local_E(i, nxi, nyi, nci, nsi, nhxi, j)
                             + local_E(j, nxj, nyj, ncj, nsj, nhxj, i) + pij_new * pij_new;
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
            double old_e = local_E(i, sq[i].x, sq[i].y, c[i], s[i], hx[i]);
            if (old_e == 0.0 && rng.u01() < 0.65) {
                int j = active_ids[rng.irand(m_act)];
                double ej = local_E(j, sq[j].x, sq[j].y, c[j], s[j], hx[j]);
                if (ej > 0.0) { i = j; old_e = ej; }
            }

            double nx = sq[i].x + rng.sym() * step_xy * sc;
            double ny = sq[i].y + rng.sym() * step_xy * sc;
            double nth = sq[i].th;
            double mv = rng.u01();
            if (allow_align && mv < 0.16) {
                int k = active_ids[rng.irand(m_act)];
                double r2 = rng.u01();
                if (r2 < 0.25) nth = 0.0;
                else if (r2 < 0.40) nth = 0.5 * PI - sq[k].th + rng.sym() * 0.015;
                else nth = sq[k].th + rng.sym() * 0.015;
            } else if (mv < 0.72) {
                nth += rng.sym() * step_th * sc;
            }
            nx = max(0.35, min(Lx - 0.35, nx));
            ny = max(0.35, min(Ly - 0.35, ny));
            double nc = cos(nth), ns = sin(nth), nhx = half * (fabs(nc) + fabs(ns));
            double new_e = local_E(i, nx, ny, nc, ns, nhx);
            double dE = new_e - old_e;
            if (dE <= 0.0 || rng.u01() < exp(-dE / T)) {
                update_sq(i, nx, ny, nth, nc, ns, nhx);
                total_E += dE;
                if (total_E < best_E) { best_E = total_E; best_sq = sq; }
            }
        }
        if (best_E < total_E) {
            vector<uint8_t> mask = is_active;
            init(best_sq, Lx, Ly, side, &mask);
        }
    }
};

double adam_relax_rect(vector<Sq>& sq, double Lx, double Ly, double side, int max_steps,
                       double lr0 = 0.02, bool freeze_angles = false, double eps_stop = 1e-13,
                       const vector<uint8_t>* active_mask = nullptr) {
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
        adam_relax_rect(sq, cur_L, cur_L, 1.0, 450, 0.02, true, 1e-15);
        adam_relax_rect(sq, cur_L, cur_L, 1.0, 1400, 0.014, false, 1e-20);
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
    int sweeps;
    bool global_polish;
};

pair<double, vector<Sq>> run_homotopy_job(const ProbeJob& job, uint64_t seed) {
    RNG rng(seed);
    vector<Sq> sq = job.init_sq;
    double a_sched[8] = {0.86, 0.91, 0.945, 0.972, 0.988, 0.995, 0.999, 1.0};
    RectSpatialSA sa;
    for (int st = 0; st < 8; ++st) {
        double a = a_sched[st];
        sa.init(sq, job.Lx, job.Ly, a, &job.mask);
        double T0 = (st == 0) ? 0.018 : 0.004 * (1.01 - a);
        double T1 = max(1e-8, 5e-5 * (1.01 - a));
        sa.run_sa(job.sweeps, T0, T1, 0.30 * (1.05 - a) + 0.015, 0.42 * (1.05 - a) + 0.025, rng, true);
        sq = sa.sq;
        adam_relax_rect(sq, job.Lx, job.Ly, a, (st < 6 ? 140 : 320), 0.015 * (1.05 - a) + 0.003,
                        false, 1e-13, &job.mask);
    }
    double E = adam_relax_rect(sq, job.Lx, job.Ly, 1.0, 1200, 0.012, false, 1e-20,
                               job.global_polish ? nullptr : &job.mask);
    return {E, sq};
}

int main() {
    omp_set_num_threads(7);
    auto t0 = chrono::steady_clock::now();

    cout << "=== 12A PART 1: Pure 1-Edge Slab Capacity (W+f) x (k+f) for W*k+1 squares ===\n" << flush;
    for (int W : {2, 3, 4, 5, 6, 7, 8}) {
        int k = 20;
        double f = 0.530;
        double Wx = W + f, Hy = k + f;
        int n_slab = W * k + 1;
        double best_E = 1e18;
        #pragma omp parallel for schedule(dynamic, 1)
        for (int w = 0; w < 14; ++w) {
            RNG rng(500001ULL + W * 997 + w * 7919);
            vector<Sq> sq;
            double deg = (16.0 + (w % 7) * 4.5) * PI / 180.0;
            for (int j = 0; j < k; ++j)
                for (int i = 0; i < W; ++i)
                    sq.push_back({(i + 0.5) * (Wx / W), (j + 0.5) * (Hy / k),
                                  ((w % 2 == 0) ? deg : ((i + j) % 2 ? deg : -deg)) + rng.sym() * 0.15});
            sq.push_back({0.5 * Wx, 0.5 * Hy, deg});
            vector<uint8_t> mask(n_slab, 1);
            ProbeJob job{"slab", Wx, Hy, sq, mask, 280000, false};
            auto [E, rsq] = run_homotopy_job(job, 600007ULL + W * 131 + w * 911);
            #pragma omp critical
            { best_E = min(best_E, E); }
        }
        cout << "  Slab W=" << W << " (" << Wx << "x" << Hy << ", need " << n_slab << "): min_E="
             << scientific << setprecision(3) << best_E << fixed << "\n" << flush;
    }

    cout << "\n=== 12A PART 2: L-Rim & 4-Edge Frame & Diag Buffer Cavities (k=20 L=20.5345) ===\n" << flush;
    double L401 = 20.5345339, f401 = L401 - 20;
    vector<string> fam_names;
    vector<ProbeJob> cav_jobs;

    auto add_cav_batch = [&](const string& fam, int count, auto builder) {
        fam_names.push_back(fam);
        for (int w = 0; w < count; ++w) cav_jobs.push_back(builder(fam, w));
    };

    // L-Rim Hard & Buffered (W=3,4,5)
    for (int W : {3, 4, 5}) {
        add_cav_batch("Lrim_hard_W" + to_string(W), 7, [&](const string& nm, int w) {
            RNG rng(100 + W * 31 + w);
            vector<Sq> sq; vector<uint8_t> mask;
            double th = (20.0 + w * 3.5) * PI / 180.0;
            for (int j = 0; j < 20; ++j) for (int i = 0; i < 20; ++i) {
                bool frozen = (i >= W && j >= W);
                double px = frozen ? (i + f401 + 0.5) : (i + 0.5) * (W + f401) / W;
                double py = frozen ? (j + f401 + 0.5) : (j + 0.5) * (W + f401) / W;
                if (!frozen && i >= W) px = i + f401 + 0.5;
                if (!frozen && j >= W) py = j + f401 + 0.5;
                sq.push_back({min(L401 - 0.5, px), min(L401 - 0.5, py),
                              frozen ? 0.0 : (j < W && i >= W ? th : 0.5 * PI - th) + rng.sym() * 0.1});
                mask.push_back(frozen ? 0 : 1);
            }
            sq.push_back({0.5 * (W + f401), 0.5 * L401, 0.5 * PI - th}); mask.push_back(1);
            return ProbeJob{nm, L401, L401, sq, mask, 350000, true};
        });
        add_cav_batch("Lrim_buf_W" + to_string(W), 7, [&](const string& nm, int w) {
            RNG rng(200 + W * 31 + w);
            vector<Sq> sq; vector<uint8_t> mask;
            double th = (20.0 + w * 3.5) * PI / 180.0;
            for (int j = 0; j < 20; ++j) for (int i = 0; i < 20; ++i) {
                bool deep_frozen = (i >= W + 2 && j >= W + 2);
                bool in_core = (i >= W && j >= W);
                double px = in_core ? (i + f401 + 0.5) : (i < W ? (i + 0.5) * (W + f401) / W : i + f401 + 0.5);
                double py = in_core ? (j + f401 + 0.5) : (j < W ? (j + 0.5) * (W + f401) / W : j + f401 + 0.5);
                sq.push_back({min(L401 - 0.5, px), min(L401 - 0.5, py),
                              in_core ? 0.0 : (j < W && i >= W ? th : 0.5 * PI - th) + rng.sym() * 0.1});
                mask.push_back(deep_frozen ? 0 : 1);
            }
            sq.push_back({0.5 * (W + f401), 0.5 * L401, 0.5 * PI - th}); mask.push_back(1);
            return ProbeJob{nm, L401, L401, sq, mask, 350000, true};
        });
    }

    // 4-Edge Pinched Frame Hard & Buffered (W=2,3)
    for (int W : {2, 3}) {
        add_cav_batch("4frame_hard_W" + to_string(W), 7, [&](const string& nm, int w) {
            RNG rng(300 + W * 31 + w);
            vector<Sq> sq; vector<uint8_t> mask;
            double th = (18.0 + w * 3.8) * PI / 180.0;
            for (int j = 0; j < 20; ++j) for (int i = 0; i < 20; ++i) {
                bool frozen = (i >= W && i < 20 - W && j >= W && j < 20 - W);
                double px = frozen ? (i + 0.5 * f401 + 0.5) : (i + 0.5) * (L401 / 20.0);
                double py = frozen ? (j + 0.5 * f401 + 0.5) : (j + 0.5) * (L401 / 20.0);
                double a = 0.0;
                if (!frozen) a = ((i < W && j >= W) || (i >= 20 - W && j < 20 - W)) ? (th + 0.5 * PI) : th;
                sq.push_back({px, py, a + (frozen ? 0.0 : rng.sym() * 0.1)});
                mask.push_back(frozen ? 0 : 1);
            }
            sq.push_back({0.5 * W + 0.2, 0.5 * L401, th + 0.5 * PI}); mask.push_back(1);
            return ProbeJob{nm, L401, L401, sq, mask, 350000, true};
        });
        add_cav_batch("4frame_buf_W" + to_string(W), 7, [&](const string& nm, int w) {
            RNG rng(400 + W * 31 + w);
            vector<Sq> sq; vector<uint8_t> mask;
            double th = (18.0 + w * 3.8) * PI / 180.0;
            for (int j = 0; j < 20; ++j) for (int i = 0; i < 20; ++i) {
                bool deep = (i >= W + 2 && i < 18 - W && j >= W + 2 && j < 18 - W);
                bool core = (i >= W && i < 20 - W && j >= W && j < 20 - W);
                double px = core ? (i + 0.5 * f401 + 0.5) : (i + 0.5) * (L401 / 20.0);
                double py = core ? (j + 0.5 * f401 + 0.5) : (j + 0.5) * (L401 / 20.0);
                double a = core ? 0.0 : (((i < W && j >= W) || (i >= 20 - W && j < 20 - W)) ? (th + 0.5 * PI) : th);
                sq.push_back({px, py, a + (core ? 0.0 : rng.sym() * 0.1)});
                mask.push_back(deep ? 0 : 1);
            }
            sq.push_back({0.5 * W + 0.2, 0.5 * L401, th + 0.5 * PI}); mask.push_back(1);
            return ProbeJob{nm, L401, L401, sq, mask, 350000, true};
        });
    }

    // Diagonal buffered cavity (w=3,4,5)
    for (int bw : {3, 4, 5}) {
        add_cav_batch("diag_buf_w" + to_string(bw), 7, [&](const string& nm, int w) {
            RNG rng(500 + bw * 31 + w);
            vector<Sq> sq; vector<uint8_t> mask;
            double th = (24.0 + w * 3.5) * PI / 180.0;
            for (int j = 0; j < 20; ++j) for (int i = 0; i < 20; ++i) {
                int d = i + j - 19;
                bool deep = abs(d) > bw / 2 + 3;
                bool band = abs(d) <= bw / 2 + 1;
                double px = i + 0.5 + (d >= 0 ? (band ? 0.5 * f401 : f401) : 0.0);
                double py = j + 0.5 + (d >= 0 ? (band ? 0.5 * f401 : f401) : 0.0);
                sq.push_back({min(L401 - 0.5, px), min(L401 - 0.5, py), band ? th + rng.sym() * 0.15 : 0.0});
                mask.push_back(deep ? 0 : 1);
            }
            sq.push_back({0.5 * L401, 0.5 * L401, th}); mask.push_back(1);
            return ProbeJob{nm, L401, L401, sq, mask, 350000, true};
        });
    }

    map<string, double> fam_min_E;
    double win401_L = 20.535533905933;
    vector<Sq> win401_sq;
    string win401_src = "none";

    #pragma omp parallel for schedule(dynamic, 1)
    for (int ji = 0; ji < (int)cav_jobs.size(); ++ji) {
        auto [E, sq] = run_homotopy_job(cav_jobs[ji], 700001ULL + ji * 7919ULL);
        auto [mp, mL] = exact_max_pen_and_L(sq, 1.0);
        #pragma omp critical
        {
            const string& nm = cav_jobs[ji].name;
            if (!fam_min_E.count(nm) || E < fam_min_E[nm]) fam_min_E[nm] = E;
            if (mp <= 1e-11 || E < 1e-9) {
                vector<Sq> pol = sq;
                double wL = polish_and_shrink_sq(pol, L401);
                if (wL < win401_L - 1e-11) {
                    win401_L = wL; win401_sq = pol; win401_src = nm;
                }
            }
        }
    }
    for (const string& fn : fam_names) {
        cout << "  " << setw(16) << left << fn << ": min_E=" << scientific << setprecision(3) << fam_min_E[fn] << fixed << "\n";
    }

    cout << "\n=== 12A PART 3: Deep n=401 Global Homotopy (28 workers x 900k sweeps/stage) ===\n" << flush;
    auto s401 = load_merged_n(401);
    double g_best_E = 1e18;
    #pragma omp parallel for schedule(dynamic, 1)
    for (int w = 0; w < 28; ++w) {
        RNG rng(880001ULL + w * 10007ULL);
        double Lt = (w < 21) ? L401 : 20.5300;
        double f = Lt - 20.0;
        vector<Sq> sq;
        int md = w % 4;
        if (md == 0 && !s401.empty()) {
            sq = s401;
            double sc = (Lt - 1e-4) / exact_max_pen_and_L(sq, 1.0).second;
            for (auto& q : sq) {
                q.x *= sc; q.y *= sc;
                double deg = fmod(fabs(q.th) * 180.0 / PI, 90.0);
                q.th += (deg > 10 && deg < 80 ? ((w / 4) - 3) * 0.08 : rng.sym() * 0.08);
                q.x += rng.sym() * 0.15; q.y += rng.sym() * 0.15;
            }
        } else if (md == 1) {
            // 4-quadrant chiral swirl
            double th = (18.0 + (w % 7) * 3.8) * PI / 180.0;
            for (int j = 0; j < 20; ++j) for (int i = 0; i < 20; ++i) {
                double px = (i + 0.5) * (Lt / 20.0), py = (j + 0.5) * (Lt / 20.0);
                double a = ((i < 10) ^ (j < 10)) ? th : 0.5 * PI - th;
                sq.push_back({px, py, a + rng.sym() * 0.15});
            }
            sq.push_back({0.5 * Lt, 0.5 * Lt, 0.25 * PI});
        } else if (md == 2) {
            // Split 4 corners (0,0),(f,0),(0,f),(f,f) leaving central cross empty + tilted swirl
            double th = (22.0 + (w % 6) * 3.5) * PI / 180.0;
            for (int j = 0; j < 20; ++j) for (int i = 0; i < 20; ++i) {
                double px = i + 0.5 + (i >= 10 ? f : 0.0);
                double py = j + 0.5 + (j >= 10 ? f : 0.0);
                double a = (abs(i - 9.5) < 3.0 || abs(j - 9.5) < 3.0) ? th : 0.0;
                sq.push_back({px, py, a + rng.sym() * 0.1});
            }
            sq.push_back({0.5 * Lt, 0.5 * Lt, th});
        } else {
            // Random vortex field
            for (int j = 0; j < 20; ++j) for (int i = 0; i < 20; ++i)
                sq.push_back({(i + 0.5) * (Lt / 20.0) + rng.sym() * 0.1,
                              (j + 0.5) * (Lt / 20.0) + rng.sym() * 0.1,
                              rng.u01() * 0.5 * PI});
            sq.push_back({0.5 * Lt, 0.5 * Lt, 0.4});
        }
        vector<uint8_t> mask(401, 1);
        ProbeJob job{"global", Lt, Lt, sq, mask, 850000, true};
        auto [E, rsq] = run_homotopy_job(job, 990001ULL + w * 7919ULL);
        auto [mp, mL] = exact_max_pen_and_L(rsq, 1.0);
        #pragma omp critical
        {
            g_best_E = min(g_best_E, E);
            if (mp <= 1e-11 || E < 1e-9) {
                vector<Sq> pol = rsq;
                double wL = polish_and_shrink_sq(pol, Lt);
                if (wL < win401_L - 1e-11) {
                    win401_L = wL; win401_sq = pol; win401_src = "global_w" + to_string(w);
                }
            }
        }
    }
    cout << "  Deep global n=401 min_E=" << scientific << setprecision(3) << g_best_E
         << fixed << setprecision(9) << " | WIN_L=" << win401_L << " (" << win401_src << ")\n";

    if (win401_L < 20.535533905933 - 1e-11) {
        system("mkdir -p poc/swarm_angles/data");
        save_csv("poc/swarm_angles/data/cand12a_n401.csv", 401, win401_sq);
        cout << "SAVED poc/swarm_angles/data/cand12a_n401.csv!\n";
    }
    double secs = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
    cout << "Idea 12A finished in " << fixed << setprecision(1) << secs << "s\n";
    return 0;
}
