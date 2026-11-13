// Idea 11: Frozen-Core Cavity a-Homotopy (11A), Soft-Interface Grid9 MIS + Joint Adam
// with n=847 45-deg+Pocket Hybrids & Clipped n=971/908 Pockets (11B),
// and Calibrated DP Per-Column Jogging Diagonal Band (11C) for D=1 roots.
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

// Fast O(1) spatial-hash SA supporting frozen sub-lattices (active_ids only moves!)
struct SpatialSA {
    int n = 0;
    double L = 1.0, side = 1.0, half = 0.5;
    double cell_sz = 1.5;
    int grid_n = 4;
    vector<Sq> sq;
    vector<double> c, s, hx;
    vector<int> cell_id;
    vector<vector<int>> cells;
    vector<uint8_t> is_active;
    vector<int> active_ids;
    double total_E = 0.0;

    void init(const vector<Sq>& init_sq, double box_L, double sq_side,
              const vector<uint8_t>* active_mask = nullptr) {
        sq = init_sq;
        n = (int)sq.size();
        L = box_L;
        side = sq_side;
        half = 0.5 * side;
        cell_sz = max(1.5, 1.5 * side);
        grid_n = max(4, (int)ceil(L / cell_sz) + 2);
        c.resize(n);
        s.resize(n);
        hx.resize(n);
        cell_id.assign(n, -1);
        cells.assign(grid_n * grid_n, {});
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
        int gx = max(0, min(grid_n - 1, (int)(x / cell_sz) + 1));
        int gy = max(0, min(grid_n - 1, (int)(y / cell_sz) + 1));
        return gy * grid_n + gx;
    }

    inline double wall_E_val(double xi, double yi, double hxi) const {
        double e = 0.0, v;
        if ((v = hxi - xi) > 0.0) e += v * v;
        if ((v = xi + hxi - L) > 0.0) e += v * v;
        if ((v = hxi - yi) > 0.0) e += v * v;
        if ((v = yi + hxi - L) > 0.0) e += v * v;
        return e;
    }

    double local_E(int i, double xi, double yi, double ci, double si, double hxi, int skip_j = -1) const {
        double e = wall_E_val(xi, yi, hxi);
        int gx = max(0, min(grid_n - 1, (int)(xi / cell_sz) + 1));
        int gy = max(0, min(grid_n - 1, (int)(yi / cell_sz) + 1));
        for (int dy = -1; dy <= 1; ++dy) {
            int yy = gy + dy;
            if (yy < 0 || yy >= grid_n) continue;
            for (int dx = -1; dx <= 1; ++dx) {
                int xx = gx + dx;
                if (xx < 0 || xx >= grid_n) continue;
                for (int j : cells[yy * grid_n + xx]) {
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
            int gx = max(0, min(grid_n - 1, (int)(sq[i].x / cell_sz) + 1));
            int gy = max(0, min(grid_n - 1, (int)(sq[i].y / cell_sz) + 1));
            for (int dy = -1; dy <= 1; ++dy) {
                int yy = gy + dy;
                if (yy < 0 || yy >= grid_n) continue;
                for (int dx = -1; dx <= 1; ++dx) {
                    int xx = gx + dx;
                    if (xx < 0 || xx >= grid_n) continue;
                    for (int j : cells[yy * grid_n + xx]) {
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
        c[i] = nc;
        s[i] = ns;
        hx[i] = nhx;
        int ncid = get_cid(nx, ny);
        int ocid = cell_id[i];
        if (ncid != ocid) {
            auto& v = cells[ocid];
            for (size_t k = 0; k < v.size(); ++k) {
                if (v[k] == i) {
                    v[k] = v.back();
                    v.pop_back();
                    break;
                }
            }
            cells[ncid].push_back(i);
            cell_id[i] = ncid;
        }
    }

    void run_sa_active(int iters, double T0, double T1, double step_xy, double step_th, RNG& rng,
                       bool allow_align = true) {
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

            double r_move = rng.u01();
            if (r_move < 0.06 && m_act >= 2) {
                // Pair swap of positions/orientations between two active squares (teleport defects)
                int i = active_ids[rng.irand(m_act)];
                int j = active_ids[rng.irand(m_act)];
                if (i == j) continue;
                double pij_old = sat_pen_pair(sq[i].x, sq[i].y, c[i], s[i], hx[i],
                                              sq[j].x, sq[j].y, c[j], s[j], hx[j], half, 0.0);
                double old_e = local_E(i, sq[i].x, sq[i].y, c[i], s[i], hx[i], j)
                             + local_E(j, sq[j].x, sq[j].y, c[j], s[j], hx[j], i)
                             + pij_old * pij_old;
                // Either swap centers or swap orientations
                bool swap_pos = (rng.u01() < 0.5);
                double nxi = swap_pos ? sq[j].x : sq[i].x;
                double nyi = swap_pos ? sq[j].y : sq[i].y;
                double nthi = swap_pos ? sq[i].th : sq[j].th;
                double nxj = swap_pos ? sq[i].x : sq[j].x;
                double nyj = swap_pos ? sq[i].y : sq[j].y;
                double nthj = swap_pos ? sq[j].th : sq[i].th;
                double nci = cos(nthi), nsi = sin(nthi), nhxi = half * (fabs(nci) + fabs(nsi));
                double ncj = cos(nthj), nsj = sin(nthj), nhxj = half * (fabs(ncj) + fabs(nsj));
                double pij_new = sat_pen_pair(nxi, nyi, nci, nsi, nhxi,
                                              nxj, nyj, ncj, nsj, nhxj, half, 0.0);
                double new_e = local_E(i, nxi, nyi, nci, nsi, nhxi, j)
                             + local_E(j, nxj, nyj, ncj, nsj, nhxj, i)
                             + pij_new * pij_new;
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
                if (r2 < 0.22) nth = 0.0;
                else if (r2 < 0.38) nth = 0.5 * PI - sq[k].th + rng.sym() * 0.015;
                else nth = sq[k].th + rng.sym() * 0.015;
            } else if (mv < 0.72) {
                nth += rng.sym() * step_th * sc;
            }
            nx = max(0.35, min(L - 0.35, nx));
            ny = max(0.35, min(L - 0.35, ny));
            double nc = cos(nth), ns = sin(nth), nhx = half * (fabs(nc) + fabs(ns));
            double new_e = local_E(i, nx, ny, nc, ns, nhx);
            double dE = new_e - old_e;
            if (dE <= 0.0 || rng.u01() < exp(-dE / T)) {
                update_sq(i, nx, ny, nth, nc, ns, nhx);
                total_E += dE;
                if (total_E < best_E) {
                    best_E = total_E;
                    best_sq = sq;
                }
            }
        }
        if (best_E < total_E) {
            vector<uint8_t> mask = is_active;
            init(best_sq, L, side, &mask);
        }
    }
};

// Adam continuous gradient optimizer with spatial hash and optional active_mask
double adam_relax(vector<Sq>& sq, double L, double side, int max_steps,
                  double lr0 = 0.02, bool freeze_angles = false, double eps_stop = 1e-13,
                  const vector<uint8_t>* active_mask = nullptr) {
    int n = (int)sq.size();
    double half = 0.5 * side;
    double cell_sz = max(1.6, 1.6 * side);
    int gn = max(4, (int)ceil(L / cell_sz) + 2);
    vector<vector<int>> cells(gn * gn);
    vector<double> m(3 * n, 0.0), v(3 * n, 0.0);
    double b1 = 0.88, b2 = 0.99, eps = 1e-8;
    double b1t = 1.0, b2t = 1.0;

    vector<Sq> best_sq = sq;
    double best_E = 1e18;

    for (int step = 0; step < max_steps; ++step) {
        if (step % 15 == 0) {
            for (auto& cell : cells) cell.clear();
            for (int i = 0; i < n; ++i) {
                int gx = max(0, min(gn - 1, (int)(sq[i].x / cell_sz) + 1));
                int gy = max(0, min(gn - 1, (int)(sq[i].y / cell_sz) + 1));
                cells[gy * gn + gx].push_back(i);
            }
        }
        vector<double> c(n), s(n), hx(n), dhx(n);
        vector<double> gx(n, 0.0), gy(n, 0.0), gth(n, 0.0);
        for (int i = 0; i < n; ++i) {
            c[i] = cos(sq[i].th);
            s[i] = sin(sq[i].th);
            double sc_c = (c[i] >= 0 ? 1.0 : -1.0);
            double sc_s = (s[i] >= 0 ? 1.0 : -1.0);
            hx[i] = half * (fabs(c[i]) + fabs(s[i]));
            dhx[i] = half * (sc_c * (-s[i]) + sc_s * c[i]);
        }
        double E = 0.0;
        for (int i = 0; i < n; ++i) {
            double v1;
            if ((v1 = hx[i] - sq[i].x) > 0.0) { E += v1 * v1; gx[i] -= 2 * v1; gth[i] += 2 * v1 * dhx[i]; }
            if ((v1 = sq[i].x + hx[i] - L) > 0.0) { E += v1 * v1; gx[i] += 2 * v1; gth[i] += 2 * v1 * dhx[i]; }
            if ((v1 = hx[i] - sq[i].y) > 0.0) { E += v1 * v1; gy[i] -= 2 * v1; gth[i] += 2 * v1 * dhx[i]; }
            if ((v1 = sq[i].y + hx[i] - L) > 0.0) { E += v1 * v1; gy[i] += 2 * v1; gth[i] += 2 * v1 * dhx[i]; }

            int cx0 = max(0, min(gn - 1, (int)(sq[i].x / cell_sz) + 1));
            int cy0 = max(0, min(gn - 1, (int)(sq[i].y / cell_sz) + 1));
            for (int dyc = -1; dyc <= 1; ++dyc) {
                int yy = cy0 + dyc;
                if (yy < 0 || yy >= gn) continue;
                for (int dxc = -1; dxc <= 1; ++dxc) {
                    int xx = cx0 + dxc;
                    if (xx < 0 || xx >= gn) continue;
                    for (int j : cells[yy * gn + xx]) {
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
                            dx * c[i] + dy * s[i],
                            -dx * s[i] + dy * c[i],
                            dx * c[j] + dy * s[j],
                            -dx * s[j] + dy * c[j]
                        };
                        double pens[4] = {
                            rad - fabs(p_axes[0]),
                            rad - fabs(p_axes[1]),
                            rad - fabs(p_axes[2]),
                            rad - fabs(p_axes[3])
                        };
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
                        gx[i] += w2 * sgn * dpx;
                        gy[i] += w2 * sgn * dpy;
                        gx[j] -= w2 * sgn * dpx;
                        gy[j] -= w2 * sgn * dpy;
                        gth[i] += w2 * (drad_dthi - sgn * dthi_ax);
                        gth[j] += w2 * (drad_dthj - sgn * dthj_ax);
                    }
                }
            }
        }
        if (E < best_E) { best_E = E; best_sq = sq; }
        if (E < eps_stop) break;

        b1t *= b1;
        b2t *= b2;
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

double polish_and_shrink(vector<Sq>& sq, double L_start) {
    double cur_L = L_start;
    auto [p0, bbox0] = exact_max_pen_and_L(sq, 1.0);
    if (p0 > 1e-11) {
        // Two-phase Adam: let upright translate first, then joint angle+pos
        adam_relax(sq, cur_L, 1.0, 450, 0.02, true, 1e-15);
        adam_relax(sq, cur_L, 1.0, 1400, 0.014, false, 1e-20);
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
            mx = min(mx, q.x - hx);
            my = min(my, q.y - hx);
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
        adam_relax(trial, target_L, 1.0, 750, 0.01, false, 1e-22);
        auto [p, b] = exact_max_pen_and_L(trial, 1.0);
        if (p <= 1e-13 && b < best_L - 1e-12) {
            best_L = b;
            best_sq = trial;
            shift_origin(best_sq);
            step *= 1.25;
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
        stringstream ss(line);
        string tok;
        vector<string> cols;
        while (getline(ss, tok, ',')) cols.push_back(tok);
        if (cols.size() < 6) continue;
        if (stoi(cols[1]) == target_n) res.push_back({stod(cols[3]), stod(cols[4]), stod(cols[5])});
    }
    if (res.empty()) return res;
    double minx = 1e18, miny = 1e18;
    for (const auto& q : res) {
        double hx = 0.5 * (fabs(cos(q.th)) + fabs(sin(q.th)));
        minx = min(minx, q.x - hx);
        miny = min(miny, q.y - hx);
    }
    for (auto& q : res) { q.x -= minx; q.y -= miny; }
    return res;
}

vector<Sq> lborder_propagate(const vector<Sq>& base, int k_from, int k_to) {
    vector<Sq> sq = base;
    auto [p, Lcur] = exact_max_pen_and_L(sq, 1.0);
    for (int k = k_from; k < k_to; ++k) {
        for (int j = 0; j < k; ++j) sq.push_back({Lcur + 0.5, j + 0.5, 0.0});
        for (int i = 0; i <= k; ++i) sq.push_back({i + 0.5, Lcur + 0.5, 0.0});
        Lcur += 1.0;
    }
    return sq;
}

void save_csv(const string& path, int n, const vector<Sq>& sq) {
    system("mkdir -p poc/swarm_angles/data");
    ofstream out(path);
    out << setprecision(17) << "id,n,square_id,x,y,angle\n";
    for (int i = 0; i < n; ++i) {
        out << i << "," << n << "," << i << "," << sq[i].x << "," << sq[i].y << "," << sq[i].th << "\n";
    }
}

// =========================================================================
// PART 11A: Frozen-Core Cavity / L-Corridor a-Homotopy
// =========================================================================
struct CavitySeed {
    vector<Sq> sq;
    vector<uint8_t> active_mask;  // 1 = moves during cavity SA, 0 = frozen crystal
    double L_tgt;
    string label;
};

vector<CavitySeed> build_cavity_seeds(int n, double L_probe, double L_fprobe,
                                      const vector<Sq>& base_seed, int workers) {
    int k = (int)floor(sqrt((double)(n - 1)) + 0.5);
    vector<CavitySeed> out;
    for (int w = 0; w < workers; ++w) {
        RNG rng(910001ULL + (uint64_t)n * 1999ULL + (uint64_t)w * 7919ULL);
        double Lt = (w % 4 == 3) ? L_fprobe : L_probe;
        double f = Lt - k;
        int type = w % 7;

        vector<Sq> sq;
        vector<uint8_t> mask;
        string lbl;

        if (type == 0) {
            // Corner W x W cavity at SW [0, W+f]^2; freeze remaining (k^2 - W^2) at shifted lattice
            int Ws[4] = {5, 7, 9, 11};
            int W = Ws[(w / 7) % 4];
            if (W >= k - 3) W = k - 4;
            double th_dom = (20.0 + ((w / 7) % 5) * 4.5) * PI / 180.0;
            for (int j = 0; j < k; ++j) {
                for (int i = 0; i < k; ++i) {
                    if (i < W && j < W) {
                        double px = (i + 0.5) * (W + f) / W;
                        double py = (j + 0.5) * (W + f) / W;
                        sq.push_back({px, py, th_dom + rng.sym() * 0.25});
                        mask.push_back(1);
                    } else {
                        sq.push_back({i + f + 0.5, j + f + 0.5, 0.0});
                        mask.push_back(0);
                    }
                }
            }
            sq.push_back({0.5 * (W + f), 0.5 * (W + f), th_dom});
            mask.push_back(1);
            lbl = "cav_corner_W" + to_string(W);
        } else if (type == 1) {
            // n=971 L-Rim cavity: freeze (k-W) x (k-W) NE core at (f+W, f+W), melt SW bottom+left rim
            int Ws[4] = {3, 4, 5, 6};
            int W = Ws[(w / 7) % 4];
            double th_dom = (20.0 + ((w / 7) % 5) * 3.8) * PI / 180.0;
            for (int j = 0; j < k; ++j) {
                for (int i = 0; i < k; ++i) {
                    if (i >= W && j >= W) {
                        sq.push_back({i + f + 0.5, j + f + 0.5, 0.0});
                        mask.push_back(0);
                    } else {
                        double px = (i + 0.5) * ((i < W) ? (W + f) / W : (Lt / k));
                        double py = (j + 0.5) * ((j < W) ? (W + f) / W : (Lt / k));
                        double th = (j < W && i >= W) ? th_dom : (i < W && j >= W ? 0.5 * PI - th_dom : 0.0);
                        sq.push_back({min(Lt - 0.5, px), min(Lt - 0.5, py), th + rng.sym() * 0.1});
                        mask.push_back(1);
                    }
                }
            }
            sq.push_back({0.5 * (W + f), 0.5 * Lt, 0.5 * PI - th_dom});
            mask.push_back(1);
            lbl = "cav_Lrim_W" + to_string(W);
        } else if (type == 2) {
            // Diagonal corridor cavity: freeze SW(0,0) + NE(f,f) triangles, melt central band (~70..170 sq)
            int band_w = 3 + ((w / 7) % 4);
            int c_cut = k - 1 - band_w / 2;
            double th_dom = (24.0 + ((w / 7) % 5) * 4.5) * PI / 180.0;
            for (int j = 0; j < k; ++j) {
                for (int i = 0; i < k; ++i) {
                    int d = i + j - (k - 1);
                    if (abs(d) <= band_w / 2 + 1) {
                        double px = (i + 0.5 + (d > 0 ? 0.5 * f : 0.0));
                        double py = (j + 0.5 + (d > 0 ? 0.5 * f : 0.0));
                        sq.push_back({min(Lt - 0.5, px), min(Lt - 0.5, py), th_dom + rng.sym() * 0.15});
                        mask.push_back(1);
                    } else if (d < 0) {
                        sq.push_back({i + 0.5, j + 0.5, 0.0});
                        mask.push_back(0);
                    } else {
                        sq.push_back({i + f + 0.5, j + f + 0.5, 0.0});
                        mask.push_back(0);
                    }
                }
            }
            sq.push_back({0.5 * Lt, 0.5 * Lt, th_dom});
            mask.push_back(1);
            lbl = "cav_diag_w" + to_string(band_w);
        } else if (type == 3) {
            // 4-Edge Pinched Frame around centered upright (k - 2W) x (k - 2W) core!
            int Ws[3] = {2, 3, 4};
            int W = Ws[(w / 7) % 3];
            double th_dom = (18.0 + ((w / 7) % 5) * 4.5) * PI / 180.0;
            for (int j = 0; j < k; ++j) {
                for (int i = 0; i < k; ++i) {
                    if (i >= W && i < k - W && j >= W && j < k - W) {
                        sq.push_back({i + 0.5 * f + 0.5, j + 0.5 * f + 0.5, 0.0});
                        mask.push_back(0);
                    } else {
                        double px = (i + 0.5) * (Lt / k);
                        double py = (j + 0.5) * (Lt / k);
                        double th = th_dom;
                        if (i < W && j >= W) th = th_dom + 0.5 * PI;
                        else if (i >= k - W && j < k - W) th = th_dom + 0.5 * PI;
                        sq.push_back({px, py, th + rng.sym() * 0.12});
                        mask.push_back(1);
                    }
                }
            }
            sq.push_back({0.5 * W + 0.25 * f, 0.5 * Lt, th_dom + 0.5 * PI});
            mask.push_back(1);
            lbl = "cav_4frame_W" + to_string(W);
        } else if (type == 4) {
            // 1-Edge Slab cavity: freeze right (k-W) x k slab at (f+W, 0), melt left W*k + 1 strip
            int Ws[4] = {3, 4, 5, 6};
            int W = Ws[(w / 7) % 4];
            double th_dom = (19.0 + ((w / 7) % 5) * 4.0) * PI / 180.0;
            for (int j = 0; j < k; ++j) {
                for (int i = 0; i < k; ++i) {
                    if (i >= W) {
                        sq.push_back({i + f + 0.5, j + 0.5, 0.0});
                        mask.push_back(0);
                    } else {
                        double px = (i + 0.5) * ((W + f) / W);
                        double py = (j + 0.5) * (Lt / k);
                        sq.push_back({px, py, th_dom + rng.sym() * 0.15});
                        mask.push_back(1);
                    }
                }
            }
            sq.push_back({0.5 * (W + f), 0.5 * Lt, th_dom});
            mask.push_back(1);
            lbl = "cav_1slab_W" + to_string(W);
        } else if (type == 5 && !base_seed.empty()) {
            // Seeded-Core Corner/Edge Melt on existing solution (n=401 / 626 / 842)
            sq = base_seed;
            // Rescale slightly to fit Lt
            double sc = (Lt - 1e-4) / max(1.0, exact_max_pen_and_L(sq, 1.0).second);
            for (auto& q : sq) { q.x *= sc; q.y *= sc; }
            mask.assign(sq.size(), 0);
            int sub = (w / 7) % 3;
            double dth = ((w / 7) % 7 - 3) * 4.0 * PI / 180.0;
            for (size_t i = 0; i < sq.size(); ++i) {
                bool act = false;
                if (sub == 0) act = (sq[i].x < 7.5 && sq[i].y < 7.5);
                else if (sub == 1) act = (sq[i].x < 4.5 || sq[i].y < 4.5);
                else act = (sq[i].x < 6.0 && sq[i].y > Lt - 7.0);
                if (act) {
                    mask[i] = 1;
                    sq[i].th += dth + rng.sym() * 0.25;
                }
            }
            lbl = "cav_seed_sub" + to_string(sub);
        } else {
            // Seeded whole-interface + opposite corner pocket melt (n=847 hybrid inspiration)
            if (!base_seed.empty()) {
                sq = base_seed;
                double sc = (Lt - 1e-4) / max(1.0, exact_max_pen_and_L(sq, 1.0).second);
                for (auto& q : sq) { q.x *= sc; q.y *= sc; }
                mask.assign(sq.size(), 0);
                int active_cnt = 0;
                for (size_t i = 0; i < sq.size(); ++i) {
                    double deg = fmod(fabs(sq[i].th) * 180.0 / PI, 90.0);
                    bool in_corner = (sq[i].x < 6.5 && sq[i].y > Lt - 6.5);
                    if (deg > 10.0 && deg < 80.0) {
                        // Keep interior 45-deg block or perturb angle
                        double dist_c = hypot(sq[i].x - 0.5 * Lt, sq[i].y - 0.5 * Lt);
                        if (dist_c > 3.0) { mask[i] = 1; sq[i].th += rng.sym() * 0.15; active_cnt++; }
                    } else if (in_corner || rng.u01() < 0.25) {
                        mask[i] = 1;
                        sq[i].th += (in_corner ? 0.62 : rng.sym() * 0.18);
                        active_cnt++;
                    }
                }
                if (active_cnt < 30) {
                    for (size_t i = 0; i < min((size_t)90, sq.size()); ++i) mask[i] = 1;
                }
                lbl = "cav_847melt";
            } else {
                // Fallback corner melt
                for (int j = 0; j < k; ++j)
                    for (int i = 0; i < k; ++i) {
                        bool act = (i < 7 && j < 7);
                        sq.push_back({act ? (i + 0.5) : (i + f + 0.5), act ? (j + 0.5) : (j + f + 0.5), act ? 0.5 : 0.0});
                        mask.push_back(act ? 1 : 0);
                    }
                sq.push_back({3.5, 3.5, 0.5});
                mask.push_back(1);
                lbl = "cav_fallback";
            }
        }
        out.push_back({move(sq), move(mask), Lt, lbl});
    }
    return out;
}

pair<double, vector<Sq>> run_11a_cavity(int n, double L_probe, double L_fprobe,
                                        const vector<Sq>& base_seed, int workers = 35,
                                        string* best_lbl_out = nullptr) {
    auto seeds = build_cavity_seeds(n, L_probe, L_fprobe, base_seed, workers);
    double best_E_global = 1e18;
    vector<Sq> best_sq_global;
    double best_win_L = 1e18;
    vector<Sq> win_sq_global;
    string best_lbl = "none";

    #pragma omp parallel for schedule(dynamic, 1)
    for (int w = 0; w < workers; ++w) {
        RNG rng(200003ULL + (uint64_t)n * 1337ULL + (uint64_t)w * 10007ULL);
        vector<Sq> sq = seeds[w].sq;
        const auto& mask = seeds[w].active_mask;
        double Lt = seeds[w].L_tgt;

        // Progressive a-Homotopy strictly inside frozen cavity (a: 0.88 -> 1.00)
        double a_sched[8] = {0.88, 0.92, 0.955, 0.978, 0.991, 0.997, 0.9994, 1.0};
        SpatialSA sa;
        for (int st = 0; st < 8; ++st) {
            double a = a_sched[st];
            sa.init(sq, Lt, a, &mask);
            int sa_steps = (st < 4) ? 260000 : 320000;
            double T0 = (st == 0) ? 0.015 : 0.004 * (1.01 - a);
            double T1 = max(1e-8, 5e-5 * (1.01 - a));
            sa.run_sa_active(sa_steps, T0, T1, 0.30 * (1.05 - a) + 0.015, 0.40 * (1.05 - a) + 0.025, rng, true);
            sq = sa.sq;
            // Relax active squares first
            adam_relax(sq, Lt, a, (st < 6 ? 150 : 350), 0.015 * (1.05 - a) + 0.003, false, 1e-13, &mask);
        }
        // Unfreeze ALL n squares for joint boundary absorption!
        double E_fin = adam_relax(sq, Lt, 1.0, 1400, 0.012, false, 1e-18, nullptr);
        auto [mp, mL] = exact_max_pen_and_L(sq, 1.0);

        #pragma omp critical
        {
            if (E_fin < best_E_global) {
                best_E_global = E_fin;
                best_sq_global = sq;
                best_lbl = seeds[w].label;
            }
            if (mp <= 1e-11 || E_fin < 1e-9) {
                vector<Sq> pol = sq;
                double wL = polish_and_shrink(pol, Lt);
                if (wL < best_win_L) {
                    best_win_L = wL;
                    win_sq_global = pol;
                    best_lbl = seeds[w].label;
                }
            }
        }
    }
    if (best_lbl_out) *best_lbl_out = best_lbl;
    if (best_win_L < 1e17) return {best_win_L, win_sq_global};
    return {-best_E_global, best_sq_global};
}

// =========================================================================
// PART 11B: Soft-Interface Grid9 MIS + Joint Adam
// =========================================================================
struct Grid9 {
    double L;
    int k;
    vector<double> gx, gy;
    vector<int> mode;
    vector<int> diag_sw, diag_nw, diag_se, diag_ne;

    void init(double box_L) {
        L = box_L;
        k = (int)floor(L);
        double f = L - k;
        double offs[3] = {0.0, 0.5 * f, f};
        gx.clear(); gy.clear(); mode.clear();
        diag_sw.clear(); diag_nw.clear(); diag_se.clear(); diag_ne.clear();
        for (int ox = 0; ox < 3; ++ox) {
            for (int oy = 0; oy < 3; ++oy) {
                for (int i = 0; i < k; ++i) {
                    double px = offs[ox] + i + 0.5;
                    if (px + 0.5 > L + 1e-7) continue;
                    for (int j = 0; j < k; ++j) {
                        double py = offs[oy] + j + 0.5;
                        if (py + 0.5 > L + 1e-7) continue;
                        gx.push_back(px);
                        gy.push_back(py);
                        mode.push_back(ox * 3 + oy);
                        diag_sw.push_back(i + j);
                        diag_nw.push_back(i + (k - 1 - j));
                        diag_se.push_back((k - 1 - i) + j);
                        diag_ne.push_back((k - 1 - i) + (k - 1 - j));
                    }
                }
            }
        }
    }

    int solve_mis_tabu(const vector<Sq>& tilted, vector<Sq>& out_up,
                       double min_gap_tilt = 0.0, double min_gap_up = 0.0) const {
        int m = (int)gx.size();
        int nt = (int)tilted.size();
        vector<double> tc(nt), ts(nt), thx(nt);
        for (int t = 0; t < nt; ++t) {
            tc[t] = cos(tilted[t].th);
            ts[t] = sin(tilted[t].th);
            thx[t] = 0.5 * (fabs(tc[t]) + fabs(ts[t]));
        }

        vector<int> cand;
        cand.reserve(m);
        for (int u = 0; u < m; ++u) {
            double ux = gx[u], uy = gy[u];
            bool ok = true;
            for (int t = 0; t < nt; ++t) {
                double rad = 0.5 + thx[t] + min_gap_tilt;
                double dx = ux - tilted[t].x;
                if (fabs(dx) >= rad) continue;
                double dy = uy - tilted[t].y;
                if (fabs(dy) >= rad) continue;
                if (fabs(dx * tc[t] + dy * ts[t]) < rad && fabs(-dx * ts[t] + dy * tc[t]) < rad) {
                    ok = false; break;
                }
            }
            if (ok) cand.push_back(u);
        }

        int nc = (int)cand.size();
        if (nc == 0) { out_up.clear(); return 0; }
        sort(cand.begin(), cand.end(), [&](int a, int b) { return gx[a] < gx[b]; });

        double thr = 1.0 + min_gap_up - 1e-12;
        vector<vector<int>> adj(nc);
        for (int a = 0; a < nc; ++a) {
            int ua = cand[a];
            double xa = gx[ua], ya = gy[ua];
            for (int b = a + 1; b < nc; ++b) {
                int ub = cand[b];
                if (gx[ub] - xa >= thr) break;
                if (fabs(ya - gy[ub]) < thr) {
                    adj[a].push_back(b);
                    adj[b].push_back(a);
                }
            }
        }

        auto greedy_order = [&](const vector<int>& order) -> vector<int> {
            vector<int> blk(nc, 0), sel;
            for (int u : order) {
                if (blk[u] == 0) {
                    sel.push_back(u);
                    for (int v : adj[u]) blk[v]++;
                }
            }
            return sel;
        };

        vector<int> best_sel;
        vector<int> ord(nc);
        iota(ord.begin(), ord.end(), 0);

        for (int om = 0; om < 10; ++om) {
            vector<double> sc(nc);
            for (int a = 0; a < nc; ++a) {
                int u = cand[a];
                int md = mode[u];
                int ox = md / 3, oy = md % 3;
                double d = 0.0;
                if (ox == 0 && oy == 0) d = diag_sw[u];
                else if (ox == 0 && oy == 2) d = diag_nw[u];
                else if (ox == 2 && oy == 0) d = diag_se[u];
                else if (ox == 2 && oy == 2) d = diag_ne[u];
                else d = k + 0.5 * min({diag_sw[u], diag_nw[u], diag_se[u], diag_ne[u]});
                double mid_pen = (ox == 1 || oy == 1) ? 20.0 : 0.0;

                if (om == 0) sc[a] = d + mid_pen;
                else if (om == 1) {
                    double px = gx[u], py = gy[u];
                    int q = (ox == 0 && oy == 0) ? (int)(px) :
                            (ox == 2 && oy == 0) ? (int)(py) :
                            (ox == 2 && oy == 2) ? (int)(L - px) : (int)(L - py);
                    sc[a] = q + 0.01 * d + mid_pen;
                } else if (om == 2) {
                    double px = gx[u], py = gy[u];
                    int q = (ox == 0 && oy == 0) ? (int)(py) :
                            (ox == 0 && oy == 2) ? (int)(px) :
                            (ox == 2 && oy == 2) ? (int)(L - py) : (int)(L - px);
                    sc[a] = q + 0.01 * d + mid_pen;
                } else if (om == 3) sc[a] = ((md == 0 || md == 8) ? 0.0 : 40.0) + d + mid_pen;
                else if (om == 4) sc[a] = ((md == 2 || md == 6) ? 0.0 : 40.0) + d + mid_pen;
                else if (om == 5) sc[a] = min({gx[u], L - gx[u], gy[u], L - gy[u]}) + mid_pen;
                else if (om == 6) sc[a] = min(gx[u], gy[u]) + mid_pen;
                else if (om == 7) sc[a] = min(L - gx[u], L - gy[u]) + mid_pen;
                else if (om == 8) sc[a] = (double)adj[a].size() + 0.01 * d + mid_pen;
                else sc[a] = d;
            }
            sort(ord.begin(), ord.end(), [&](int i, int j) { return sc[i] < sc[j]; });
            auto sel = greedy_order(ord);
            if (sel.size() > best_sel.size()) best_sel = move(sel);
        }

        // Fast (1, 2)-swap local search
        vector<int> in_s(nc, 0), tight_cnt(nc, 0);
        for (int u : best_sel) in_s[u] = 1;
        for (int u = 0; u < nc; ++u) {
            if (in_s[u]) for (int v : adj[u]) tight_cnt[v]++;
        }
        int cur_sz = (int)best_sel.size();
        bool improved = true;
        int rounds = 0;
        while (improved && rounds < 12) {
            improved = false;
            rounds++;
            for (int u = 0; u < nc; ++u) {
                if (!in_s[u]) continue;
                vector<int> freed;
                for (int v : adj[u]) if (!in_s[v] && tight_cnt[v] == 1) freed.push_back(v);
                if (freed.size() < 2) continue;
                int add1 = -1, add2 = -1;
                for (size_t i = 0; i < freed.size() && add1 < 0; ++i) {
                    for (size_t j = i + 1; j < freed.size(); ++j) {
                        int a = freed[i], b = freed[j];
                        bool confl = false;
                        for (int w : adj[a]) if (w == b) { confl = true; break; }
                        if (!confl) { add1 = a; add2 = b; break; }
                    }
                }
                if (add1 >= 0) {
                    in_s[u] = 0;
                    for (int w : adj[u]) tight_cnt[w]--;
                    in_s[add1] = 1; for (int w : adj[add1]) tight_cnt[w]++;
                    in_s[add2] = 1; for (int w : adj[add2]) tight_cnt[w]++;
                    cur_sz++;
                    improved = true;
                }
            }
        }

        out_up.clear();
        for (int u = 0; u < nc; ++u) {
            if (in_s[u]) out_up.push_back({gx[cand[u]], gy[cand[u]], 0.0});
        }
        return cur_sz;
    }
};

// Build dual-clipped / inset n=971 / n=908 reflected pockets
vector<Sq> build_971_clipped(double L, double th, double wp, double x_lo, double x_hi, double y_lo,
                             double u_cut, int mode) {
    double c = cos(th), s = sin(th), hx = 0.5 * (fabs(c) + fabs(s));
    vector<Sq> p1;
    int max_i = (int)ceil(L * 1.5);
    for (int iu = -max_i; iu <= max_i; ++iu) {
        for (int iv = -max_i; iv <= max_i; ++iv) {
            double u = u_cut + iu;
            if (u < u_cut - 1e-7) continue;
            double v = iv;
            double x = u * c - v * s;
            double y = u * s + v * c;
            if (x - hx >= x_lo - 1e-7 && x + hx <= x_hi + 1e-7 &&
                y - hx >= y_lo - 1e-7 && y + hx <= y_lo + wp + 1e-7) {
                p1.push_back({x, y, th});
            }
        }
    }
    if (p1.empty()) return {};
    vector<Sq> out = p1;
    if (mode == 0) {
        for (const auto& q : p1) out.push_back({q.y, q.x, 0.5 * PI - th});
    } else if (mode == 1) {
        for (const auto& q : p1) out.push_back({L - q.y, L - q.x, 0.5 * PI - th});
    } else if (mode == 2) {
        for (const auto& q : p1) out.push_back({L - q.x, L - q.y, th});
    } else if (mode == 3) {
        for (const auto& q : p1) {
            out.push_back({L - q.y, q.x, th + 0.5 * PI});
            out.push_back({L - q.x, L - q.y, th});
            out.push_back({q.y, L - q.x, th + 0.5 * PI});
        }
    }
    auto [pen, bL] = exact_max_pen_and_L(out, 1.0);
    if (pen > 1e-6 || bL > L + 1e-5) return {};
    return out;
}

// Build n=847 hybrid: m x m 45-deg diamond + 1 (or 2) small non-45 pocket(s)
vector<Sq> build_847_hybrid(double L, int m, double d_cx, double d_cy,
                            int pw, int ph, double p_th, double px0, double py0, bool mirror2) {
    vector<Sq> out;
    double th45 = 0.25 * PI;
    double c45 = cos(th45), s45 = sin(th45), hx45 = sqrt(0.5);
    double u0 = -0.5 * (m - 1), v0 = -0.5 * (m - 1);
    for (int iu = 0; iu < m; ++iu) {
        for (int iv = 0; iv < m; ++iv) {
            double u = u0 + iu, v = v0 + iv;
            double x = d_cx + u * c45 - v * s45;
            double y = d_cy + u * s45 + v * c45;
            if (x - hx45 < -1e-7 || x + hx45 > L + 1e-7 || y - hx45 < -1e-7 || y + hx45 > L + 1e-7)
                return {};
            out.push_back({x, y, th45});
        }
    }
    double c = cos(p_th), s = sin(p_th), hx = 0.5 * (fabs(c) + fabs(s));
    vector<Sq> pkt;
    for (int iu = 0; iu < pw; ++iu) {
        for (int iv = 0; iv < ph; ++iv) {
            double u = iu, v = iv;
            double x = px0 + u * c - v * s;
            double y = py0 + u * s + v * c;
            if (x - hx < -1e-7 || x + hx > L + 1e-7 || y - hx < -1e-7 || y + hx > L + 1e-7)
                continue;
            pkt.push_back({x, y, p_th});
        }
    }
    if (pkt.size() < 4) return {};
    for (const auto& q : pkt) out.push_back(q);
    if (mirror2) {
        for (const auto& q : pkt) out.push_back({L - q.y, L - q.x, 0.5 * PI - p_th});
    }
    auto [pen, bL] = exact_max_pen_and_L(out, 1.0);
    if (pen > 1e-6 || bL > L + 1e-5) return {};
    return out;
}

vector<Sq> build_chamfer_core(double L, int U, int V, int cut, double th, double cx, double cy) {
    double c = cos(th), s = sin(th), hx = 0.5 * (fabs(c) + fabs(s));
    vector<Sq> out;
    double u0 = -0.5 * (U - 1), v0 = -0.5 * (V - 1);
    for (int iu = 0; iu < U; ++iu) {
        for (int iv = 0; iv < V; ++iv) {
            int du = min(iu, U - 1 - iu);
            int dv = min(iv, V - 1 - iv);
            if (du + dv < cut) continue;
            double u = u0 + iu, v = v0 + iv;
            double x = cx + u * c - v * s;
            double y = cy + u * s + v * c;
            if (x - hx < -1e-7 || x + hx > L + 1e-7 || y - hx < -1e-7 || y + hx > L + 1e-7) return {};
            out.push_back({x, y, th});
        }
    }
    return out;
}

// =========================================================================
// PART 11C: Calibrated Per-Column DP Jogging Diagonal Band
// =========================================================================
vector<Sq> build_dp_jog_band(double L, int k, int a_nw, int a_se, int w,
                             double th, double bend_end, double u_off, double min_gap_stair = 0.0) {
    double f = L - k;
    // Build NW staircase (0, f) and SE staircase (f, 0) squares near interface
    vector<Sq> stair;
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
            int d_nw = i + (k - 1 - j);
            if (d_nw < a_nw && d_nw >= a_nw - 3) {
                stair.push_back({i + 0.5, j + f + 0.5, 0.0});
            }
            int d_se = (k - 1 - i) + j;
            if (d_se < a_se && d_se >= a_se - 3) {
                stair.push_back({i + f + 0.5, j + 0.5, 0.0});
            }
        }
    }
    // Check if NW and SE themselves fit walls
    double c0 = cos(th), s0 = sin(th);
    double tan_diff = tan(0.25 * PI - th);  // v-shift rate along 45-deg main diagonal!

    int max_slices = (int)ceil(L * 1.45) + 4;
    double u_start = 0.7 * (c0 + s0) + u_off;

    // For each slice sl, precompute feasible v0 values around nominal 45-deg centerline
    const int NV = 97;
    vector<vector<double>> feas_v(max_slices);
    vector<double> slice_th(max_slices, th);

    for (int sl = 0; sl < max_slices; ++sl) {
        double u = u_start + sl * 1.0;
        // Nominal intersection of line u = x*c0 + y*s0 with diagonal x = y:
        // x_nom = u / (c0 + s0), y_nom = x_nom => v_nom = x_nom * (c0 - s0)
        double x_nom = u / (c0 + s0);
        double v_center = x_nom * (c0 - s0);
        double sth = th;
        if (sl <= 2) sth = th - bend_end;
        else if (sl >= max_slices - 4) sth = th + bend_end;
        slice_th[sl] = sth;
        double sc = cos(sth), ss = sin(sth), shx = 0.5 * (fabs(sc) + fabs(ss));

        for (int vi = 0; vi < NV; ++vi) {
            double v0 = v_center - 0.5 * (w - 1) + (vi - NV / 2) * 0.025;
            bool ok = true;
            for (int r = 0; r < w; ++r) {
                double vr = v0 + r;
                double px = u * c0 - vr * s0;
                double py = u * s0 + vr * c0;
                if (px - shx < -1e-7 || px + shx > L + 1e-7 || py - shx < -1e-7 || py + shx > L + 1e-7) {
                    ok = false; break;
                }
                for (const auto& st : stair) {
                    double p = sat_pen_pair(px, py, sc, ss, shx, st.x, st.y, 1.0, 0.0, 0.5, 0.5, min_gap_stair);
                    if (p > 1e-7) { ok = false; break; }
                }
                if (!ok) break;
            }
            if (ok) feas_v[sl].push_back(v0);
        }
    }

    // DP for longest contiguous chain of slices with |v_{sl} - v_{sl-1}| <= 0.48
    int best_len = 0, best_end_sl = -1, best_end_vi = -1;
    vector<vector<int>> dp_len(max_slices), dp_prev(max_slices);
    for (int sl = 0; sl < max_slices; ++sl) {
        int m = (int)feas_v[sl].size();
        dp_len[sl].assign(m, 1);
        dp_prev[sl].assign(m, -1);
        for (int vi = 0; vi < m; ++vi) {
            if (sl > 0) {
                double vcur = feas_v[sl][vi];
                for (int pj = 0; pj < (int)feas_v[sl - 1].size(); ++pj) {
                    double dv = fabs(vcur - feas_v[sl - 1][pj]);
                    if (dv <= 0.48 && dp_len[sl - 1][pj] + 1 > dp_len[sl][vi]) {
                        dp_len[sl][vi] = dp_len[sl - 1][pj] + 1;
                        dp_prev[sl][vi] = pj;
                    }
                }
            }
            if (dp_len[sl][vi] > best_len) {
                best_len = dp_len[sl][vi];
                best_end_sl = sl;
                best_end_vi = vi;
            }
        }
    }
    if (best_len < 3) return {};

    vector<Sq> band;
    int csl = best_end_sl, cvi = best_end_vi;
    while (csl >= 0 && cvi >= 0) {
        double u = u_start + csl * 1.0;
        double v0 = feas_v[csl][cvi];
        double sth = slice_th[csl];
        for (int r = 0; r < w; ++r) {
            double vr = v0 + r;
            band.push_back({u * c0 - vr * s0, u * s0 + vr * c0, sth});
        }
        int pvi = dp_prev[csl][cvi];
        csl--;
        cvi = pvi;
    }
    reverse(band.begin(), band.end());
    // Exact check if bend_end caused tiny end overlap -> reject if overlaps
    if (bend_end != 0.0) {
        auto [pen, bL] = exact_max_pen_and_L(band, 1.0);
        if (pen > 1e-6) return {};
    }
    return band;
}

// Helper: smart trim `cap - n` worst-overlapping upright squares before Adam
vector<Sq> smart_trim_and_build(int n, double L, const vector<Sq>& tilted, const vector<Sq>& up) {
    int nt = (int)tilted.size();
    vector<Sq> res = tilted;
    if (nt + (int)up.size() <= n) {
        res.insert(res.end(), up.begin(), up.end());
        return res;
    }
    vector<pair<double, int>> scored;
    scored.reserve(up.size());
    for (int i = 0; i < (int)up.size(); ++i) {
        double pen_sum = 0.0;
        for (int t = 0; t < nt; ++t) {
            double c = cos(tilted[t].th), s = sin(tilted[t].th);
            double hx = 0.5 * (fabs(c) + fabs(s));
            double p = sat_pen_pair(up[i].x, up[i].y, 1.0, 0.0, 0.5,
                                    tilted[t].x, tilted[t].y, c, s, hx, 0.5, 0.0);
            if (p > 0.0) pen_sum += p * p;
        }
        scored.push_back({pen_sum, i});
    }
    sort(scored.begin(), scored.end());
    for (int i = 0; i < n - nt; ++i) res.push_back(up[scored[i].second]);
    return res;
}

int main() {
    omp_set_num_threads(7);
    auto t0 = chrono::steady_clock::now();

    // 1) Calibrate 11C DP Jogging Band against n=403 @ 20.564935730 (should get >= 403!)
    {
        double L403 = 20.564935730;
        Grid9 g403;
        g403.init(L403);
        int best403 = 0;
        for (int a = 15; a <= 18; ++a) {
            auto b = build_dp_jog_band(L403, 20, a, a, 3, 34.58 * PI / 180.0, 0.0, 0.0, 0.0);
            if (!b.empty()) {
                vector<Sq> up;
                int nup = g403.solve_mis_tabu(b, up, -0.05, 0.0);
                best403 = max(best403, (int)b.size() + nup);
            }
        }
        cout << "Calib 11C DP band on n=403 (L=" << fixed << setprecision(6) << L403
             << "): best_cap=" << best403 << "/403\n" << flush;
    }

    struct Target {
        int n;
        double L_base, L_front;
    };
    vector<Target> targets = {
        {401, 20.535533905933, 20.526135333409},
        {626, 25.520815280171, 25.511048764219},
        {842, 29.520815280171, 29.502152707903},
        {442, 21.535533905933, 21.522922939732},
        {530, 23.535533905933, 23.517304919934},
    };

    auto s401 = load_merged_n(401);
    auto s626 = load_merged_n(626);
    map<int, vector<Sq>> seeds;
    seeds[401] = s401;
    seeds[442] = lborder_propagate(s401, 20, 21);
    seeds[530] = lborder_propagate(s401, 20, 23);
    seeds[626] = s626;
    seeds[842] = lborder_propagate(s626, 25, 29);

    ofstream jout("poc/swarm_angles/docs/idea11_results.json");
    jout << "[\n";

    for (size_t ti = 0; ti < targets.size(); ++ti) {
        const auto& T = targets[ti];
        int n = T.n;
        int k = (int)floor(sqrt((double)(n - 1)) + 0.5);
        double L_probe = T.L_base - 0.001;
        double L_fprobe = T.L_front + 0.002;
        auto tn = chrono::steady_clock::now();

        double best_win_L = T.L_base;
        vector<Sq> best_win_sq;
        string best_src = "none";

        // --- 11A: Frozen-Core Cavity a-Homotopy ---
        string lbl_11a;
        auto [res11a, sq11a] = run_11a_cavity(n, L_probe, L_fprobe, seeds[n], 35, &lbl_11a);
        double min_E_11a = (res11a < 0 ? -res11a : 0.0);
        if (res11a > 0 && res11a < best_win_L - 1e-11) {
            best_win_L = res11a;
            best_win_sq = sq11a;
            best_src = "11A_" + lbl_11a;
        }

        // --- Build Candidates for 11B & 11C ---
        struct Cand { vector<Sq> tilted; string fam; string tag; };
        vector<Cand> cands;

        // 11B_971: Clipped & inset reflected oblique pockets
        for (double deg : {18.0, 22.0, 25.2, 28.5, 32.0, 36.87}) {
            double th = deg * PI / 180.0;
            for (double wp : {2.2, 3.2, 4.4, 5.8, 7.0}) {
                for (double x_lo : {0.0, 2.0, 6.0, 9.5}) {
                    if (x_lo >= L_probe - 5.0) continue;
                    for (double dx_hi : {0.0, 2.0, 3.5}) {
                        double x_hi = L_probe - dx_hi;
                        if (x_hi - x_lo < 4.0) continue;
                        for (double y_lo : {0.0, 1.0}) {
                            for (int md : {-1, 0, 1, 3}) {
                                if (md == 3 && x_lo < 1.5 && dx_hi > 0.5) continue;
                                auto tq = build_971_clipped(L_probe, th, wp, x_lo, x_hi, y_lo, x_lo * cos(th), md);
                                if (!tq.empty() && (int)tq.size() < n) {
                                    cands.push_back({move(tq), "11B_971", "971 d=" + to_string((int)deg) + " w=" + to_string((int)(wp*10)) + " m=" + to_string(md)});
                                }
                            }
                        }
                    }
                }
            }
        }

        // 11B_847: 45-deg diamond + small non-45 corner/wall pocket(s)
        for (int m_dim = max(6, k - 12); m_dim <= min(k - 4, 21); ++m_dim) {
            double half_span = 0.5 * m_dim * sqrt(2.0);
            if (2.0 * half_span > L_probe - 0.3) continue;
            for (double shift_off : {0.0, 1.5, 3.0, 4.5}) {
                double d_cx = L_probe - half_span - 0.5 - shift_off;
                double d_cy = half_span + 0.5 + shift_off * 0.3;
                if (d_cx - half_span < 0.0 || d_cy + half_span > L_probe) continue;
                for (double pdeg : {24.0, 30.0, 34.0, 37.4, 41.0}) {
                    for (int pw : {2, 3, 4}) {
                        for (int ph : {3, 4, 5}) {
                            auto tq = build_847_hybrid(L_probe, m_dim, d_cx, d_cy, pw, ph,
                                                       pdeg * PI / 180.0, 1.2, L_probe - 4.8, false);
                            if (!tq.empty() && (int)tq.size() < n) {
                                cands.push_back({move(tq), "11B_847", "847 m=" + to_string(m_dim) + " pd=" + to_string((int)pdeg)});
                            }
                        }
                    }
                }
            }
        }

        // 11B_core: Larger U x V chamfered non-45 & 45 cores
        for (double deg : {26.0, 31.0, 36.0, 40.0, 43.0, 45.0}) {
            double th = deg * PI / 180.0;
            for (int U = max(8, k - 11); U <= min(k - 2, 22); ++U) {
                for (int V : {U, U - 1}) {
                    for (int cut : {0, 1, 2, 3}) {
                        for (double ox : {0.0, 0.35}) {
                            auto tq = build_chamfer_core(L_probe, U, V, cut, th, 0.5 * L_probe - ox, 0.5 * L_probe);
                            if (!tq.empty() && (int)tq.size() < n) {
                                cands.push_back({move(tq), "11B_core", "core " + to_string(U) + "x" + to_string(V) + " c=" + to_string(cut) + " d=" + to_string((int)deg)});
                            }
                        }
                    }
                }
            }
        }

        // 11C_dp: DP Jogging Diagonal Bands across (w, deg, a_nw, a_se)
        for (double deg : {24.0, 28.5, 33.0, 34.58, 37.5, 41.5, 45.0}) {
            double th = deg * PI / 180.0;
            for (int w : {1, 2, 3, 4, 5, 6}) {
                for (int a_base = k - w - 3; a_base <= k - w + 1; ++a_base) {
                    if (a_base < 4) continue;
                    for (int d_a : {0, 1}) {
                        for (double u_off : {0.0, 0.45}) {
                            for (double bend : {0.0, 0.05}) {
                                auto tq = build_dp_jog_band(L_probe, k, a_base, a_base - d_a, w, th, bend, u_off, -0.02);
                                if (!tq.empty() && (int)tq.size() < n) {
                                    cands.push_back({move(tq), "11C_dp", "dp w=" + to_string(w) + " d=" + to_string((int)deg) + " a=" + to_string(a_base)});
                                }
                            }
                        }
                    }
                }
            }
        }

        Grid9 g9;
        g9.init(L_probe);
        vector<int> active;
        for (int ci = 0; ci < (int)cands.size(); ++ci) {
            double area_rem = L_probe * L_probe - (double)cands[ci].tilted.size();
            if ((int)cands[ci].tilted.size() + (int)area_rem >= n - 2) active.push_back(ci);
        }

        map<string, int> fam_best_strict, fam_best_soft;
        map<string, string> fam_tag_strict, fam_tag_soft;

        #pragma omp parallel for schedule(dynamic, 4)
        for (int ai = 0; ai < (int)active.size(); ++ai) {
            int ci = active[ai];
            vector<Sq> up0, up_soft;
            int nup0 = g9.solve_mis_tabu(cands[ci].tilted, up0, 0.0, 0.0);
            int cap0 = (int)cands[ci].tilted.size() + nup0;

            int cap_s = cap0;
            int soft_best_nup = nup0;
            // Test soft tilted-border gaps (-0.06, -0.11) when strict cap0 is close (>= n - 5)
            if (cap0 >= n - 5) {
                for (double mg : {-0.06, -0.11}) {
                    vector<Sq> ups;
                    int nups = g9.solve_mis_tabu(cands[ci].tilted, ups, mg, 0.0);
                    int cs = (int)cands[ci].tilted.size() + nups;
                    if (cs > cap_s) { cap_s = cs; up_soft = move(ups); soft_best_nup = nups; }
                    if (cs >= n) {
                        auto cand_sq = smart_trim_and_build(n, L_probe, cands[ci].tilted, up_soft);
                        double wL = polish_and_shrink(cand_sq, L_probe);
                        #pragma omp critical
                        {
                            if (wL < best_win_L - 1e-11) {
                                best_win_L = wL;
                                best_win_sq = cand_sq;
                                best_src = cands[ci].fam + ":soft(" + cands[ci].tag + ")";
                            }
                        }
                    }
                }
            }
            if (cap0 >= n) {
                auto cand_sq = smart_trim_and_build(n, L_probe, cands[ci].tilted, up0);
                double wL = polish_and_shrink(cand_sq, L_probe);
                #pragma omp critical
                {
                    if (wL < best_win_L - 1e-11) {
                        best_win_L = wL;
                        best_win_sq = cand_sq;
                        best_src = cands[ci].fam + ":strict(" + cands[ci].tag + ")";
                    }
                }
            }

            #pragma omp critical
            {
                const string& fam = cands[ci].fam;
                if (cap0 > fam_best_strict[fam]) {
                    fam_best_strict[fam] = cap0;
                    fam_tag_strict[fam] = cands[ci].tag + "(nt=" + to_string(cands[ci].tilted.size()) + ")";
                }
                if (cap_s > fam_best_soft[fam]) {
                    fam_best_soft[fam] = cap_s;
                    fam_tag_soft[fam] = cands[ci].tag + "(nt=" + to_string(cands[ci].tilted.size()) + ")";
                }
            }
        }

        if (best_win_L < T.L_base - 1e-11) {
            save_csv("poc/swarm_angles/data/cand11_n" + to_string(n) + ".csv", n, best_win_sq);
        }

        double secs = chrono::duration<double>(chrono::steady_clock::now() - tn).count();
        cout << setprecision(6)
             << "n=" << n << " (base=" << T.L_base << " front=" << T.L_front << "):\n"
             << "  11A_cav min_E=" << scientific << setprecision(2) << min_E_11a << fixed
             << " (" << lbl_11a << ")\n"
             << "  11B_971 strict=" << fam_best_strict["11B_971"] << " soft=" << fam_best_soft["11B_971"] << " [" << fam_tag_strict["11B_971"] << "]\n"
             << "  11B_847 strict=" << fam_best_strict["11B_847"] << " soft=" << fam_best_soft["11B_847"] << " [" << fam_tag_strict["11B_847"] << "]\n"
             << "  11B_core strict=" << fam_best_strict["11B_core"] << " soft=" << fam_best_soft["11B_core"] << " [" << fam_tag_strict["11B_core"] << "]\n"
             << "  11C_dp  strict=" << fam_best_strict["11C_dp"] << " soft=" << fam_best_soft["11C_dp"] << " [" << fam_tag_strict["11C_dp"] << "]\n"
             << "  WIN_L=" << setprecision(9) << best_win_L << " (" << best_src << ") in "
             << setprecision(2) << secs << "s (active_cands=" << active.size() << ")\n" << flush;

        jout << "  {\"n\": " << n << ", \"L_base\": " << setprecision(12) << T.L_base
             << ", \"L_front\": " << T.L_front
             << ", \"min_11a_E\": " << scientific << min_E_11a << fixed
             << ", \"11b_971_strict\": " << fam_best_strict["11B_971"]
             << ", \"11b_971_soft\": " << fam_best_soft["11B_971"]
             << ", \"11b_847_strict\": " << fam_best_strict["11B_847"]
             << ", \"11b_847_soft\": " << fam_best_soft["11B_847"]
             << ", \"11b_core_strict\": " << fam_best_strict["11B_core"]
             << ", \"11b_core_soft\": " << fam_best_soft["11B_core"]
             << ", \"11c_dp_strict\": " << fam_best_strict["11C_dp"]
             << ", \"11c_dp_soft\": " << fam_best_soft["11C_dp"]
             << ", \"best_win_L\": " << setprecision(12) << best_win_L
             << ", \"best_src\": \"" << best_src << "\", \"seconds\": " << setprecision(2) << secs << "}"
             << (ti + 1 < targets.size() ? "," : "") << "\n";
    }
    jout << "]\n";
    jout.close();
    double tot = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
    cout << "Idea 11 finished in " << fixed << setprecision(1) << tot << "s\n";
    return 0;
}
