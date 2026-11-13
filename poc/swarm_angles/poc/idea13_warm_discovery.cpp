// Idea 13A: Warm-Box (L_warm = k + 0.558 / 0.575) No-Snap Multi-Domain Crystallization
// + Local Nematic Alignment + Patch Shift + n=37 Hinged Chevron & n=50 Edge-Slip Bands
// + Joint Adam & L-Shrink to Basin Minimum L* on n=401, 442, 530, 626.
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

struct NematicSA {
    int n = 0;
    double L = 1.0, side = 1.0, half = 0.5;
    double cell_sz = 1.5;
    int gn = 4;
    vector<Sq> sq;
    vector<double> c, s, hx;
    vector<int> cell_id;
    vector<vector<int>> cells;
    vector<uint8_t> is_active;
    vector<uint8_t> tilt_lock;
    vector<int> active_ids;
    double total_E = 0.0;

    void init(const vector<Sq>& init_sq, double box_L, double sq_side,
              const vector<uint8_t>* active_mask = nullptr,
              const vector<uint8_t>* tlock = nullptr) {
        sq = init_sq;
        n = (int)sq.size();
        L = box_L; side = sq_side; half = 0.5 * side;
        cell_sz = max(1.5, 1.5 * side);
        gn = max(4, (int)ceil(L / cell_sz) + 2);
        c.resize(n); s.resize(n); hx.resize(n);
        cell_id.assign(n, -1);
        cells.assign(gn * gn, {});
        is_active.assign(n, 1);
        tilt_lock.assign(n, 0);
        active_ids.clear();
        for (int i = 0; i < n; ++i) {
            if (active_mask && (*active_mask)[i] == 0) is_active[i] = 0;
            if (tlock) tilt_lock[i] = (*tlock)[i];
            if (is_active[i]) active_ids.push_back(i);
            c[i] = cos(sq[i].th); s[i] = sin(sq[i].th);
            hx[i] = half * (fabs(c[i]) + fabs(s[i]));
            int cid = get_cid(sq[i].x, sq[i].y);
            cell_id[i] = cid;
            cells[cid].push_back(i);
        }
        recompute_E();
    }

    inline int get_cid(double x, double y) const {
        int gx = max(0, min(gn - 1, (int)(x / cell_sz) + 1));
        int gy = max(0, min(gn - 1, (int)(y / cell_sz) + 1));
        return gy * gn + gx;
    }

    inline double wall_E(double xi, double yi, double hxi) const {
        double e = 0.0, v;
        if ((v = hxi - xi) > 0.0) e += v * v;
        if ((v = xi + hxi - L) > 0.0) e += v * v;
        if ((v = hxi - yi) > 0.0) e += v * v;
        if ((v = yi + hxi - L) > 0.0) e += v * v;
        return e;
    }

    inline double orient_E(double th, uint8_t tl) const {
        if (!tl || side >= 0.992) return 0.0;
        double s2 = fabs(sin(2.0 * th));
        if (s2 < 0.20) {
            double d = 0.20 - s2;
            return 0.06 * d * d;
        }
        return 0.0;
    }

    double local_E(int i, double xi, double yi, double th, double ci, double si, double hxi, int skip_j = -1) const {
        double e = wall_E(xi, yi, hxi) + orient_E(th, tilt_lock[i]);
        int gx = max(0, min(gn - 1, (int)(xi / cell_sz) + 1));
        int gy = max(0, min(gn - 1, (int)(yi / cell_sz) + 1));
        for (int dy = -1; dy <= 1; ++dy) {
            int yy = gy + dy;
            if (yy < 0 || yy >= gn) continue;
            for (int dx = -1; dx <= 1; ++dx) {
                int xx = gx + dx;
                if (xx < 0 || xx >= gn) continue;
                for (int j : cells[yy * gn + xx]) {
                    if (j == i || j == skip_j) continue;
                    double p = sat_pen_pair(xi, yi, ci, si, hxi,
                                            sq[j].x, sq[j].y, c[j], s[j], hx[j], half);
                    if (p > 0.0) e += p * p;
                }
            }
        }
        return e;
    }

    int pick_local_neighbor(int i, RNG& rng) const {
        int gx = max(0, min(gn - 1, (int)(sq[i].x / cell_sz) + 1));
        int gy = max(0, min(gn - 1, (int)(sq[i].y / cell_sz) + 1));
        int cand[64], nc = 0;
        for (int dy = -1; dy <= 1 && nc < 64; ++dy) {
            int yy = gy + dy;
            if (yy < 0 || yy >= gn) continue;
            for (int dx = -1; dx <= 1 && nc < 64; ++dx) {
                int xx = gx + dx;
                if (xx < 0 || xx >= gn) continue;
                for (int j : cells[yy * gn + xx]) {
                    if (j != i && nc < 64) cand[nc++] = j;
                }
            }
        }
        if (nc == 0) return -1;
        return cand[rng.irand(nc)];
    }

    double recompute_E() {
        double e = 0.0;
        for (int i = 0; i < n; ++i) {
            e += wall_E(sq[i].x, sq[i].y, hx[i]) + orient_E(sq[i].th, tilt_lock[i]);
            int gx = max(0, min(gn - 1, (int)(sq[i].x / cell_sz) + 1));
            int gy = max(0, min(gn - 1, (int)(sq[i].y / cell_sz) + 1));
            for (int dy = -1; dy <= 1; ++dy) {
                int yy = gy + dy;
                if (yy < 0 || yy >= gn) continue;
                for (int dx = -1; dx <= 1; ++dx) {
                    int xx = gx + dx;
                    if (xx < 0 || xx >= gn) continue;
                    for (int j : cells[yy * gn + xx]) {
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

    void run_sa(int iters, double T0, double T1, double step_xy, double step_th, RNG& rng, bool no_zero_snap) {
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

            double rmove = rng.u01();
            if (rmove < 0.05 && m_act >= 2) {
                // Local swap / angle-exchange with nearby square
                int i = active_ids[rng.irand(m_act)];
                int j = pick_local_neighbor(i, rng);
                if (j < 0 || !is_active[j] || i == j) continue;
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
            if (mv < 0.16) {
                // Local nematic alignment with neighbor (never forced 0-snap if no_zero_snap || tilt_lock[i])
                int nb = pick_local_neighbor(i, rng);
                if (nb >= 0) {
                    if (rng.u01() < 0.25) nth = 0.5 * PI - sq[nb].th + rng.sym() * 0.015;
                    else nth = sq[nb].th + rng.sym() * 0.015;
                }
                if (!no_zero_snap && !tilt_lock[i] && rng.u01() < 0.08) nth = 0.0;
            } else if (mv < 0.72) {
                nth += rng.sym() * step_th * sc;
            }
            nx = max(0.32, min(L - 0.32, nx));
            ny = max(0.32, min(L - 0.32, ny));
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
            init(best_sq, L, side, &mask, &tl);
        }
    }
};

double adam_relax(vector<Sq>& sq, double L, double side, int max_steps,
                  double lr0 = 0.018, bool freeze_angles = false, double eps_stop = 1e-14,
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
        adam_relax(sq, cur_L, 1.0, 500, 0.018, true, 1e-15);
        adam_relax(sq, cur_L, 1.0, 1400, 0.013, false, 1e-20);
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

    double step = 0.004;
    for (int round = 0; round < 28; ++round) {
        vector<Sq> trial = best_sq;
        double target_L = best_L - step;
        double halfL = 0.5 * best_L;
        double sc = (target_L - 1.0) / max(1.0, best_L - 1.0);
        for (auto& q : trial) {
            q.x = 0.5 * target_L + (q.x - halfL) * sc;
            q.y = 0.5 * target_L + (q.y - halfL) * sc;
        }
        adam_relax(trial, target_L, 1.0, 900, 0.010, false, 1e-22);
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

struct WarmJob {
    int n, k;
    string fam;
    double L_warm;
    vector<Sq> init_sq;
    vector<uint8_t> mask;
    vector<uint8_t> tlock;
    int sweeps;
    bool no_zero_snap;
};

pair<double, vector<Sq>> run_warm_job(const WarmJob& job, uint64_t seed) {
    RNG rng(seed);
    vector<Sq> sq = job.init_sq;
    double a_sched[7] = {0.85, 0.905, 0.945, 0.972, 0.988, 0.996, 1.0};
    NematicSA sa;
    for (int st = 0; st < 7; ++st) {
        double a = a_sched[st];
        sa.init(sq, job.L_warm, a, &job.mask, &job.tlock);
        double T0 = (st == 0) ? 0.020 : 0.004 * (1.01 - a);
        double T1 = max(1e-8, 5e-5 * (1.01 - a));
        sa.run_sa(job.sweeps, T0, T1, 0.28 * (1.05 - a) + 0.015, 0.42 * (1.05 - a) + 0.025, rng, job.no_zero_snap);
        sq = sa.sq;
        adam_relax(sq, job.L_warm, a, (st < 5 ? 140 : 320), 0.015 * (1.05 - a) + 0.003,
                   false, 1e-13, &job.mask);
    }
    double E = adam_relax(sq, job.L_warm, 1.0, 1200, 0.012, false, 1e-20, nullptr);
    return {E, sq};
}

int main() {
    omp_set_num_threads(7);
    auto t0 = chrono::steady_clock::now();

    struct Target { int n, k; double L_base, L_front; };
    vector<Target> targets = {
        {401, 20, 20.535533905933, 20.526135333409},
        {442, 21, 21.535533905933, 21.522923000000},
        {530, 23, 23.535533905933, 23.517305000000},
    };

    vector<WarmJob> jobs;
    for (const auto& tg : targets) {
        int n = tg.n, k = tg.k;
        auto base_sq = load_merged_n(n);
        double warm_offs[2] = {0.555, 0.605};

        // 1) chev_n37: 3-segment hinged diagonal band (45+d, 45, 45-d) + upright corners
        for (int w_idx = 0; w_idx < 14; ++w_idx) {
            double Lw = k + warm_offs[w_idx % 2];
            double fw = Lw - k;
            RNG rng(131001ULL + n * 97 + w_idx * 7919);
            int w_band = 3 + (w_idx % 4);  // 3..6
            double d_ang = (3.0 + (w_idx / 2) * 2.2) * PI / 180.0;
            vector<Sq> sq; vector<uint8_t> mask, tl;
            // Upright lattice skipping diagonal channel |i + j - (k - 1)| <= w_band/2 + 1
            int sw_max = k - 2 - w_band / 2;
            for (int j = 0; j < k; ++j) for (int i = 0; i < k; ++i) {
                int diag_d = i + j - (k - 1);
                if (abs(diag_d) <= (w_band + 1) / 2) continue;
                bool in_sw = (i <= k / 2 && j <= k / 2 && i + j < k - 1);
                bool in_ne = (i >= k / 2 && j >= k / 2 && i + j > k - 1);
                double ox = (in_ne || diag_d > 0) ? fw : 0.0;
                double oy = (in_ne || diag_d < 0) ? fw : 0.0;
                if (in_sw) { ox = 0.0; oy = 0.0; }
                if (in_ne) { ox = fw; oy = fw; }
                sq.push_back({i + ox + 0.5, j + oy + 0.5, 0.0});
                bool deep = abs(diag_d) > (w_band + 1) / 2 + 2;
                mask.push_back(deep ? 0 : 1);
                tl.push_back(0);
            }
            (void)sw_max;
            int need = n - (int)sq.size();
            for (int q = 0; q < need; ++q) {
                double frac = (q + 0.5) / max(1, need);
                double u = 1.2 + frac * (Lw - 2.4);
                double v_off = ((q % w_band) - 0.5 * (w_band - 1)) * 0.95;
                double px = u - v_off * 0.707;
                double py = Lw - u - v_off * 0.707;
                double a = (frac < 0.38) ? (0.25 * PI + d_ang) : (frac > 0.62 ? (0.25 * PI - d_ang) : 0.25 * PI);
                sq.push_back({max(0.5, min(Lw - 0.5, px)), max(0.5, min(Lw - 0.5, py)), a + rng.sym() * 0.08});
                mask.push_back(1); tl.push_back(1);
            }
            jobs.push_back({n, k, "chev_n37", Lw, sq, mask, tl, 420000, true});
        }

        // 2) slip_n50: 36.87deg / 34deg edge-slipped sheared band + corners
        for (int w_idx = 0; w_idx < 14; ++w_idx) {
            double Lw = k + warm_offs[w_idx % 2];
            double fw = Lw - k;
            RNG rng(132001ULL + n * 97 + w_idx * 7919);
            double th = (24.0 + (w_idx % 7) * 2.8) * PI / 180.0;
            int w_band = 3 + (w_idx % 3);
            vector<Sq> sq; vector<uint8_t> mask, tl;
            for (int j = 0; j < k; ++j) for (int i = 0; i < k; ++i) {
                int diag_d = i + j - (k - 1);
                if (abs(diag_d) <= (w_band + 1) / 2) continue;
                double ox = (diag_d > 0) ? fw : 0.0;
                double oy = (diag_d < 0) ? fw : 0.0;
                if (i >= k / 2 && j >= k / 2) { ox = fw; oy = fw; }
                if (i < k / 2 && j < k / 2) { ox = 0.0; oy = 0.0; }
                sq.push_back({i + ox + 0.5, j + oy + 0.5, 0.0});
                mask.push_back(abs(diag_d) > (w_band + 1) / 2 + 2 ? 0 : 1);
                tl.push_back(0);
            }
            int need = n - (int)sq.size();
            for (int q = 0; q < need; ++q) {
                double frac = (q + 0.5) / max(1, need);
                double u = 1.0 + frac * (Lw - 2.0);
                int row = q % w_band;
                double slip = (row == 0 ? -0.18 : (row == w_band - 1 ? 0.18 : 0.0));
                double v_off = (row - 0.5 * (w_band - 1)) * 0.95;
                double px = u - v_off * sin(th) + slip * cos(th);
                double py = Lw - u + v_off * cos(th) + slip * sin(th);
                sq.push_back({max(0.5, min(Lw - 0.5, px)), max(0.5, min(Lw - 0.5, py)), th + rng.sym() * 0.08});
                mask.push_back(1); tl.push_back(1);
            }
            jobs.push_back({n, k, "slip_n50", Lw, sq, mask, tl, 420000, true});
        }

        // 3) small_tilt: full-box chiral/rotated multi-grain 4..20 deg (no 0-snap)
        for (int w_idx = 0; w_idx < 14; ++w_idx) {
            double Lw = k + warm_offs[w_idx % 2];
            RNG rng(133001ULL + n * 97 + w_idx * 7919);
            double th = (4.0 + (w_idx % 7) * 2.6) * PI / 180.0;
            vector<Sq> sq; vector<uint8_t> mask(n, 1), tl(n, 1);
            for (int j = 0; j < k; ++j) for (int i = 0; i < k; ++i) {
                double px = (i + 0.5) * (Lw / k), py = (j + 0.5) * (Lw / k);
                bool q2 = ((i < k / 2) ^ (j < k / 2));
                double a = (w_idx < 7) ? (q2 ? th : -th) : (q2 ? th : 0.5 * PI - th);
                sq.push_back({px, py, a + rng.sym() * 0.06});
            }
            sq.push_back({0.5 * Lw, 0.5 * Lw, 0.25 * PI});
            jobs.push_back({n, k, "small_tilt", Lw, sq, mask, tl, 450000, true});
        }

        // 4) 2blk_hinge: perturb merged.csv 45deg blocks by +-delta + asymmetric seam melt
        if ((int)base_sq.size() == n) {
            for (int w_idx = 0; w_idx < 14; ++w_idx) {
                double Lw = k + warm_offs[w_idx % 2];
                RNG rng(134001ULL + n * 97 + w_idx * 7919);
                auto [mp0, L0] = exact_max_pen_and_L(base_sq, 1.0);
                (void)mp0;
                double sc = (Lw - 1e-4) / L0;
                vector<Sq> sq = base_sq;
                vector<uint8_t> mask(n, 1), tl(n, 0);
                double d_ang = (2.0 + (w_idx % 7) * 2.2) * PI / 180.0;
                for (int i = 0; i < n; ++i) {
                    sq[i].x *= sc; sq[i].y *= sc;
                    bool deep_sw = (sq[i].x < 5.0 && sq[i].y < 5.0);
                    bool deep_ne = (sq[i].x > Lw - 5.0 && sq[i].y > Lw - 5.0);
                    if (deep_sw || deep_ne) {
                        mask[i] = 0;
                    } else {
                        double deg = fmod(fabs(sq[i].th) * 180.0 / PI, 90.0);
                        if (deg > 10 && deg < 80) {
                            tl[i] = 1;
                            sq[i].th += (sq[i].x < sq[i].y ? d_ang : -d_ang) + rng.sym() * 0.04;
                        } else {
                            sq[i].th += rng.sym() * 0.08;
                        }
                        sq[i].x += rng.sym() * 0.14;
                        sq[i].y += rng.sym() * 0.14;
                    }
                }
                jobs.push_back({n, k, "2blk_hinge", Lw, sq, mask, tl, 420000, true});
            }
        }

        // 5) voronoi_chiral: 4-grain de novo Voronoi crystals with patch-neighbor nematic alignment
        for (int w_idx = 0; w_idx < 14; ++w_idx) {
            double Lw = k + warm_offs[w_idx % 2];
            RNG rng(135001ULL + n * 97 + w_idx * 7919);
            int ng = 4 + (w_idx % 3);
            vector<double> gx(ng), gy(ng), gth(ng);
            for (int g = 0; g < ng; ++g) {
                gx[g] = (0.15 + 0.7 * rng.u01()) * Lw;
                gy[g] = (0.15 + 0.7 * rng.u01()) * Lw;
                gth[g] = (0.10 + 0.35 * rng.u01()) * PI;
            }
            vector<Sq> sq; vector<uint8_t> mask(n, 1), tl(n, 1);
            for (int j = 0; j < k; ++j) for (int i = 0; i < k; ++i) {
                double px = (i + 0.5) * (Lw / k) + rng.sym() * 0.12;
                double py = (j + 0.5) * (Lw / k) + rng.sym() * 0.12;
                int bg = 0; double bd = 1e18;
                for (int g = 0; g < ng; ++g) {
                    double d = (px - gx[g]) * (px - gx[g]) + (py - gy[g]) * (py - gy[g]);
                    if (d < bd) { bd = d; bg = g; }
                }
                sq.push_back({max(0.4, min(Lw - 0.4, px)), max(0.4, min(Lw - 0.4, py)), gth[bg] + rng.sym() * 0.08});
            }
            sq.push_back({0.5 * Lw, 0.5 * Lw, gth[0]});
            jobs.push_back({n, k, "voronoi", Lw, sq, mask, tl, 450000, true});
        }
    }

    // Also n=626 hinge melt (14 workers)
    auto s626 = load_merged_n(626);
    if (s626.size() == 626) {
        for (int w_idx = 0; w_idx < 14; ++w_idx) {
            double Lw = (w_idx < 7) ? 25.535 : 25.548;
            RNG rng(136001ULL + w_idx * 7919);
            auto [mp0, L0] = exact_max_pen_and_L(s626, 1.0);
            (void)mp0;
            double sc = (Lw - 1e-4) / L0;
            vector<Sq> sq = s626;
            vector<uint8_t> mask(626, 1), tl(626, 0);
            double d_ang = (1.5 + (w_idx % 7) * 1.8) * PI / 180.0;
            for (int i = 0; i < 626; ++i) {
                sq[i].x *= sc; sq[i].y *= sc;
                bool corner = (sq[i].x < 4.0 && sq[i].y < 4.0) || (sq[i].x > Lw - 4.0 && sq[i].y > Lw - 4.0)
                           || (sq[i].x < 4.0 && sq[i].y > Lw - 4.0) || (sq[i].x > Lw - 4.0 && sq[i].y < 4.0);
                if (corner) { mask[i] = 0; }
                else {
                    double deg = fmod(fabs(sq[i].th) * 180.0 / PI, 90.0);
                    if (deg > 10 && deg < 80) {
                        tl[i] = 1;
                        sq[i].th += (sq[i].x + sq[i].y < Lw ? d_ang : -d_ang) + rng.sym() * 0.04;
                    } else sq[i].th += rng.sym() * 0.06;
                    sq[i].x += rng.sym() * 0.12; sq[i].y += rng.sym() * 0.12;
                }
            }
            jobs.push_back({626, 25, "626_hinge", Lw, sq, mask, tl, 360000, true});
        }
    }

    cout << "Idea 13A running " << jobs.size() << " warm-box discovery jobs on 7 threads...\n" << flush;

    struct FamStats {
        int total = 0, feas = 0;
        double min_E = 1e18;
        double best_L = 1e18;
    };
    map<pair<int, string>, FamStats> stats;
    map<int, pair<double, vector<Sq>>> best_by_n;

    #pragma omp parallel for schedule(dynamic, 1)
    for (int ji = 0; ji < (int)jobs.size(); ++ji) {
        auto [E, rsq] = run_warm_job(jobs[ji], 990001ULL + ji * 7919ULL);
        auto [mp, mL] = exact_max_pen_and_L(rsq, 1.0);
        double shrunk_L = 1e18;
        vector<Sq> shrunk_sq;
        if (mp <= 1e-10 || E < 1e-9) {
            shrunk_sq = rsq;
            shrunk_L = polish_and_shrink_sq(shrunk_sq, jobs[ji].L_warm);
        }
        #pragma omp critical
        {
            auto& st = stats[{jobs[ji].n, jobs[ji].fam}];
            st.total++;
            st.min_E = min(st.min_E, E);
            if (shrunk_L < 1e17) {
                st.feas++;
                if (shrunk_L < st.best_L) st.best_L = shrunk_L;
                if (!best_by_n.count(jobs[ji].n) || shrunk_L < best_by_n[jobs[ji].n].first) {
                    best_by_n[jobs[ji].n] = {shrunk_L, shrunk_sq};
                }
            }
        }
    }

    for (const auto& kv : stats) {
        cout << "  n=" << kv.first.first << " fam=" << setw(12) << left << kv.first.second
             << " feas=" << kv.second.feas << "/" << kv.second.total
             << " min_E=" << scientific << setprecision(2) << kv.second.min_E << fixed
             << " best_L*=" << setprecision(9) << (kv.second.best_L < 1e17 ? kv.second.best_L : -1.0) << "\n";
    }
    for (const auto& kv : best_by_n) {
        string spath = "/home/lucas/.gemini/antigravity-cli/brain/95f98074-e284-43bd-8857-8c64d023144c/scratch/best13_n" + to_string(kv.first) + ".csv";
        save_csv(spath, kv.first, kv.second.second);
        cout << "  BEST n=" << kv.first << ": L*=" << setprecision(12) << kv.second.first << " saved to " << spath << "\n";
    }
    double secs = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
    cout << "Idea 13A finished in " << fixed << setprecision(1) << secs << "s\n";
    return 0;
}
