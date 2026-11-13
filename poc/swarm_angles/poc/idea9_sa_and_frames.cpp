// Fast C++/OpenMP Idea 9 solver:
// 9A: Pinwheel + Tabu MIS calibration & upgrade on Grid4C
// 9B: Small/medium-angle (U x V) cores, 4-edge C4/C2 pinched frames, & n=364/404-type skewed bands
// 9C: Spatial-Hash Cluster+Patch SA + Adam L-Shrinker from merged.csv & def=1+1 seeds
//
// Compile & run (sandboxed CPU ONLY, max 7 threads):
//   g++ -O3 -std=c++17 -fopenmp poc/swarm_angles/poc/idea9_sa_and_frames.cpp \
//     -o /home/lucas/.gemini/antigravity-cli/brain/95f98074-e284-43bd-8857-8c64d023144c/scratch/idea9
//   OMP_NUM_THREADS=7 /home/lucas/.gemini/antigravity-cli/brain/95f98074-e284-43bd-8857-8c64d023144c/scratch/idea9

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <omp.h>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

static constexpr double PI = 3.14159265358979323846;

struct Square {
    double x, y, th;
};

struct PreSq {
    double x, y, c, s, hx;
};

inline PreSq make_pre(double x, double y, double th) {
    double c = std::cos(th), s = std::sin(th);
    return {x, y, c, s, 0.5 * (std::abs(c) + std::abs(s))};
}

// Positive if overlapping (clearance gap = -min_depth; >0 means overlap)
inline double sat_overlap_depth(const PreSq& a, const PreSq& b) {
    double dx = b.x - a.x, dy = b.y - a.y;
    if (std::abs(dx) >= a.hx + b.hx || std::abs(dy) >= a.hx + b.hx) return -1.0;

    double best_clear = -1e18; // maximum separation among the 4 axes (<0 means overlapping)
    double axes[4][2] = {{a.c, a.s}, {-a.s, a.c}, {b.c, b.s}, {-b.s, b.c}};
    for (int k = 0; k < 4; ++k) {
        double ux = axes[k][0], uy = axes[k][1];
        double d = std::abs(dx * ux + dy * uy);
        double ra = 0.5 * (std::abs(ux * a.c + uy * a.s) + std::abs(-ux * a.s + uy * a.c));
        double rb = 0.5 * (std::abs(ux * b.c + uy * b.s) + std::abs(-ux * b.s + uy * b.c));
        double clr = d - (ra + rb);
        if (clr >= 0.0) return -1.0;
        if (clr > best_clear) best_clear = clr;
    }
    return -best_clear; // > 0
}

inline bool sat_clear_pair(const PreSq& a, const PreSq& b, double min_gap) {
    double dx = b.x - a.x, dy = b.y - a.y;
    if (std::abs(dx) >= a.hx + b.hx + min_gap || std::abs(dy) >= a.hx + b.hx + min_gap) return true;
    double axes[4][2] = {{a.c, a.s}, {-a.s, a.c}, {b.c, b.s}, {-b.s, b.c}};
    for (int k = 0; k < 4; ++k) {
        double ux = axes[k][0], uy = axes[k][1];
        double d = std::abs(dx * ux + dy * uy);
        double ra = 0.5 * (std::abs(ux * a.c + uy * a.s) + std::abs(-ux * a.s + uy * a.c));
        double rb = 0.5 * (std::abs(ux * b.c + uy * b.s) + std::abs(-ux * b.s + uy * b.c));
        if (d >= ra + rb + min_gap) return true;
    }
    return false;
}

std::pair<double, double> sat_max_overlap_and_bbox(const std::vector<Square>& sq) {
    int n = (int)sq.size();
    std::vector<PreSq> ps(n);
    double minx = 1e18, maxx = -1e18, miny = 1e18, maxy = -1e18;
    for (int i = 0; i < n; ++i) {
        ps[i] = make_pre(sq[i].x, sq[i].y, sq[i].th);
        minx = std::min(minx, ps[i].x - ps[i].hx); maxx = std::max(maxx, ps[i].x + ps[i].hx);
        miny = std::min(miny, ps[i].y - ps[i].hx); maxy = std::max(maxy, ps[i].y + ps[i].hx);
    }
    double max_ov = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double d = sat_overlap_depth(ps[i], ps[j]);
            if (d > max_ov) max_ov = d;
        }
    }
    return {max_ov, std::max(maxx - minx, maxy - miny)};
}

// SAT energy + analytic gradients wrt (x, y, th)
double sat_energy_grad(
    const std::vector<double>& x,
    const std::vector<double>& y,
    const std::vector<double>& th,
    double L,
    double w_pair,
    double w_wall,
    std::vector<double>& gx,
    std::vector<double>& gy,
    std::vector<double>& gth
) {
    int n = (int)x.size();
    std::fill(gx.begin(), gx.end(), 0.0);
    std::fill(gy.begin(), gy.end(), 0.0);
    std::fill(gth.begin(), gth.end(), 0.0);
    double energy = 0.0;

    std::vector<double> c(n), s(n), sc(n), ss(n), hx(n), dhx(n);
    for (int i = 0; i < n; ++i) {
        c[i] = std::cos(th[i]);
        s[i] = std::sin(th[i]);
        sc[i] = (c[i] >= 0.0 ? 1.0 : -1.0);
        ss[i] = (s[i] >= 0.0 ? 1.0 : -1.0);
        hx[i] = 0.5 * (std::abs(c[i]) + std::abs(s[i]));
        dhx[i] = 0.5 * (-sc[i] * s[i] + ss[i] * c[i]);
        double xi = x[i], yi = y[i], h = hx[i], dh = dhx[i];
        if (xi - h < 0.0) { double d = h - xi; energy += w_wall * d * d; double g = 2.0 * w_wall * d; gx[i] -= g; gth[i] += g * dh; }
        if (xi + h > L)   { double d = xi + h - L; energy += w_wall * d * d; double g = 2.0 * w_wall * d; gx[i] += g; gth[i] += g * dh; }
        if (yi - h < 0.0) { double d = h - yi; energy += w_wall * d * d; double g = 2.0 * w_wall * d; gy[i] -= g; gth[i] += g * dh; }
        if (yi + h > L)   { double d = yi + h - L; energy += w_wall * d * d; double g = 2.0 * w_wall * d; gy[i] += g; gth[i] += g * dh; }
    }

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double dx = x[j] - x[i], dy = y[j] - y[i];
            if (std::abs(dx) >= hx[i] + hx[j] || std::abs(dy) >= hx[i] + hx[j]) continue;

            double best_pen = 1e18;
            int best_k = -1;
            double best_p1 = 0, best_p2 = 0, best_s = 0, best_ux = 0, best_uy = 0;

            for (int k = 0; k < 4; ++k) {
                double ux = (k == 0) ? c[i] : (k == 1) ? -s[i] : (k == 2) ? c[j] : -s[j];
                double uy = (k == 0) ? s[i] : (k == 1) ? c[i] : (k == 2) ? s[j] : c[j];
                double d_proj = dx * ux + dy * uy;
                double s_d = (d_proj >= 0.0) ? 1.0 : -1.0;
                double p1 = ux * c[i] + uy * s[i];
                double p2 = -ux * s[i] + uy * c[i];
                double p3 = ux * c[j] + uy * s[j];
                double p4 = -ux * s[j] + uy * c[j];
                double pen = 0.5 * (std::abs(p1) + std::abs(p2) + std::abs(p3) + std::abs(p4)) - std::abs(d_proj);
                if (pen <= 0.0) { best_pen = 0.0; break; }
                if (pen < best_pen) {
                    best_pen = pen; best_k = k; best_s = s_d; best_ux = ux; best_uy = uy;
                    best_p1 = (p1 >= 0.0 ? 1.0 : -1.0);
                    best_p2 = (p2 >= 0.0 ? 1.0 : -1.0);
                }
            }
            if (best_pen <= 0.0 || best_k < 0) continue;

            energy += w_pair * best_pen * best_pen;
            double g = 2.0 * w_pair * best_pen;
            gx[i] += g * best_s * best_ux; gx[j] -= g * best_s * best_ux;
            gy[i] += g * best_s * best_uy; gy[j] -= g * best_s * best_uy;

            double p3 = best_ux * c[j] + best_uy * s[j];
            double p4 = -best_ux * s[j] + best_uy * c[j];
            double s3 = (p3 >= 0.0 ? 1.0 : -1.0), s4 = (p4 >= 0.0 ? 1.0 : -1.0);
            double dux = (best_k == 0) ? -s[i] : (best_k == 1) ? -c[i] : (best_k == 2) ? -s[j] : -c[j];
            double duy = (best_k == 0) ? c[i]  : (best_k == 1) ? -s[i] : (best_k == 2) ? c[j]  : -s[j];
            double dd = best_s * (dx * dux + dy * duy);

            if (best_k < 2) {
                double d_j = 0.5 * (s3 * (dux * c[j] + duy * s[j]) + s4 * (-dux * s[j] + duy * c[j])) - dd;
                gth[i] += g * d_j;
                double d_other = 0.5 * (s3 * (-best_ux * s[j] + best_uy * c[j]) + s4 * (-best_ux * c[j] - best_uy * s[j]));
                gth[j] += g * d_other;
            } else {
                double d_i = 0.5 * (best_p1 * (dux * c[i] + duy * s[i]) + best_p2 * (-dux * s[i] + duy * c[i])) - dd;
                gth[j] += g * d_i;
                double d_other = 0.5 * (best_p1 * (-best_ux * s[i] + best_uy * c[i]) + best_p2 * (-best_ux * c[i] - best_uy * s[i]));
                gth[i] += g * d_other;
            }
        }
    }
    return energy;
}

double polish_and_shrink(std::vector<Square>& sq, double L_start, int n_iters = 450) {
    int n = (int)sq.size();
    std::vector<double> x(n), y(n), th(n), gx(n), gy(n), gth(n);
    std::vector<double> mx(n, 0), vx(n, 0), my(n, 0), vy(n, 0), mt(n, 0), vt(n, 0);
    for (int i = 0; i < n; ++i) { x[i] = sq[i].x; y[i] = sq[i].y; th[i] = sq[i].th; }

    auto run_adam = [&](double L, int iters, double lr) {
        std::fill(mx.begin(), mx.end(), 0.0); std::fill(vx.begin(), vx.end(), 0.0);
        std::fill(my.begin(), my.end(), 0.0); std::fill(vy.begin(), vy.end(), 0.0);
        std::fill(mt.begin(), mt.end(), 0.0); std::fill(vt.begin(), vt.end(), 0.0);
        for (int t = 1; t <= iters; ++t) {
            double e = sat_energy_grad(x, y, th, L, 1.0, 1.5, gx, gy, gth);
            if (e < 1e-26) break;
            double b1 = 0.9, b2 = 0.999, eps = 1e-8;
            double bc1 = 1.0 - std::pow(b1, t), bc2 = 1.0 - std::pow(b2, t);
            double cur_lr = lr * (0.15 + 0.85 * (1.0 - double(t) / iters));
            for (int i = 0; i < n; ++i) {
                mx[i] = b1 * mx[i] + (1 - b1) * gx[i]; vx[i] = b2 * vx[i] + (1 - b2) * gx[i] * gx[i];
                my[i] = b1 * my[i] + (1 - b1) * gy[i]; vy[i] = b2 * vy[i] + (1 - b2) * gy[i] * gy[i];
                mt[i] = b1 * mt[i] + (1 - b1) * gth[i]; vt[i] = b2 * vt[i] + (1 - b2) * gth[i] * gth[i];
                x[i] -= cur_lr * (mx[i] / bc1) / (std::sqrt(vx[i] / bc2) + eps);
                y[i] -= cur_lr * (my[i] / bc1) / (std::sqrt(vy[i] / bc2) + eps);
                th[i] -= cur_lr * 0.5 * (mt[i] / bc1) / (std::sqrt(vt[i] / bc2) + eps);
            }
        }
    };

    double cur_L = L_start;
    run_adam(cur_L, n_iters, 2e-3);
    std::vector<Square> cur(n);
    for (int i = 0; i < n; ++i) cur[i] = {x[i], y[i], th[i]};
    auto [pen, boxL] = sat_max_overlap_and_bbox(cur);
    if (pen > 1e-11) return 1e18;
    cur_L = std::min(cur_L, boxL);
    sq = cur;

    for (double step : {1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6}) {
        for (int tries = 0; tries < 8; ++tries) {
            double trial_L = cur_L - step;
            auto xb = x, yb = y, tb = th;
            run_adam(trial_L, 260, step * 0.6 + 8e-5);
            for (int i = 0; i < n; ++i) cur[i] = {x[i], y[i], th[i]};
            auto [p2, b2] = sat_max_overlap_and_bbox(cur);
            if (p2 <= 1e-11 && b2 <= cur_L - 1e-11) { cur_L = b2; sq = cur; }
            else { x = xb; y = yb; th = tb; break; }
        }
    }
    return cur_L;
}

// 9A: Grid4C with Pinwheel + Strip + Sweep + Tabu/Plateau MIS
struct Grid4C {
    double L;
    int k, N;
    std::vector<double> x, y;
    std::vector<int> corner, d, ci, cj, cheb;
    std::vector<std::vector<int>> adj;
    static constexpr int N_ORD = 8;
    std::vector<int> fast_ord[N_ORD];

    void init(double L_, double min_gap) {
        L = L_;
        k = (int)std::floor(L);
        N = 4 * k * k;
        x.resize(N); y.resize(N); corner.resize(N);
        d.resize(N); ci.resize(N); cj.resize(N); cheb.resize(N);
        adj.assign(N, {});
        const double ox[4] = {0.5, L - 0.5, 0.5, L - 0.5};
        const double sx[4] = {1.0, -1.0, 1.0, -1.0};
        const double oy[4] = {0.5, 0.5, L - 0.5, L - 0.5};
        const double sy[4] = {1.0, 1.0, -1.0, -1.0};
        int idx = 0;
        for (int c = 0; c < 4; ++c) {
            for (int i = 0; i < k; ++i) {
                for (int j = 0; j < k; ++j) {
                    x[idx] = ox[c] + sx[c] * i;
                    y[idx] = oy[c] + sy[c] * j;
                    corner[idx] = c;
                    ci[idx] = i; cj[idx] = j;
                    d[idx] = i + j;
                    cheb[idx] = std::max(i, j);
                    ++idx;
                }
            }
        }
        double thr = 1.0 + min_gap;
        for (int i = 0; i < N; ++i) {
            for (int j = i + 1; j < N; ++j) {
                if (corner[i] != corner[j] && std::abs(x[i] - x[j]) < thr && std::abs(y[i] - y[j]) < thr) {
                    adj[i].push_back(j);
                    adj[j].push_back(i);
                }
            }
        }
        auto key_fn = [&](int m, int u) -> int {
            int c = corner[u], i = ci[u], j = cj[u];
            if (m == 0) return cheb[u] * 100 + d[u];
            if (m == 1) return d[u] * 100 + cheb[u];
            if (m == 2) return ((c == 1 || c == 2) ? (c == 1 ? i * 100 + j : j * 100 + i) : 10000 + cheb[u] * 100 + d[u]); // NW/SE wall-arm pinwheel CW
            if (m == 3) return ((c == 0 || c == 3) ? (c == 0 ? j * 100 + i : i * 100 + j) : 10000 + cheb[u] * 100 + d[u]); // SW/NE wall-arm pinwheel
            if (m == 4) { // CW 4-arm pinwheel
                int arm = (c == 0) ? j : (c == 1) ? i : (c == 3) ? j : i;
                int oth = (c == 0) ? i : (c == 1) ? j : (c == 3) ? i : j;
                return arm * 100 + oth;
            }
            if (m == 5) { // CCW 4-arm pinwheel
                int arm = (c == 0) ? i : (c == 1) ? j : (c == 3) ? i : j;
                int oth = (c == 0) ? j : (c == 1) ? i : (c == 3) ? j : i;
                return arm * 100 + oth;
            }
            if (m == 6) return cheb[u] + ((c == 1 || c == 2) ? 1000 : 0);
            return cheb[u] + ((c == 0 || c == 3) ? 1000 : 0);
        };
        for (int m = 0; m < N_ORD; ++m) {
            fast_ord[m].resize(N);
            std::iota(fast_ord[m].begin(), fast_ord[m].end(), 0);
            std::sort(fast_ord[m].begin(), fast_ord[m].end(), [&](int a, int b) {
                int ka = key_fn(m, a), kb = key_fn(m, b);
                if (ka != kb) return ka < kb;
                return corner[a] < corner[b];
            });
        }
    }

    void compute_alive(const std::vector<PreSq>& T, double min_gap, std::vector<uint8_t>& alive, int& alive_cnt) const {
        alive.assign(N, 1);
        alive_cnt = N;
        if (T.empty()) return;
        double minx = 1e18, maxx = -1e18, miny = 1e18, maxy = -1e18;
        for (const auto& t : T) {
            minx = std::min(minx, t.x - t.hx); maxx = std::max(maxx, t.x + t.hx);
            miny = std::min(miny, t.y - t.hx); maxy = std::max(maxy, t.y + t.hx);
        }
        double pad = 0.5 + min_gap + 1e-9;
        for (int i = 0; i < N; ++i) {
            double px = x[i], py = y[i];
            if (px + pad < minx || px - pad > maxx || py + pad < miny || py - pad > maxy) continue;
            PreSq up{px, py, 1.0, 0.0, 0.5};
            for (const auto& t : T) {
                if (!sat_clear_pair(up, t, min_gap)) {
                    alive[i] = 0;
                    --alive_cnt;
                    break;
                }
            }
        }
    }

    int fast_greedy_lb(const std::vector<uint8_t>& alive, std::vector<uint8_t>& blocked) const {
        int best = 0;
        blocked.resize(N);
        for (int m = 0; m < N_ORD; ++m) {
            std::fill(blocked.begin(), blocked.end(), 0);
            int cnt = 0;
            for (int u : fast_ord[m]) {
                if (alive[u] && !blocked[u]) {
                    ++cnt;
                    for (int v : adj[u]) blocked[v] = 1;
                }
            }
            if (cnt > best) best = cnt;
        }
        return best;
    }

    // Exact Degree-0/1 reduction + Pinwheel/Row-Col + Tabu + ILS + Exact B&B (M <= 48)
    std::vector<Square> solve_mis_exact(const std::vector<uint8_t>& alive_in) const {
        std::vector<uint8_t> removed(N, 0), selected(N, 0);
        std::vector<int> deg(N, 0);
        for (int i = 0; i < N; ++i) {
            if (!alive_in[i]) { removed[i] = 1; continue; }
            int d_cnt = 0;
            for (int v : adj[i]) if (alive_in[v]) ++d_cnt;
            deg[i] = d_cnt;
        }
        std::vector<int> q;
        for (int i = 0; i < N; ++i) if (!removed[i] && deg[i] <= 1) q.push_back(i);
        auto remove_node = [&](int v) {
            if (removed[v]) return;
            removed[v] = 1;
            for (int w : adj[v]) if (!removed[w] && --deg[w] <= 1) q.push_back(w);
        };
        size_t qh = 0;
        while (qh < q.size()) {
            int u = q[qh++];
            if (removed[u]) continue;
            if (deg[u] == 0) { selected[u] = 1; remove_node(u); }
            else if (deg[u] == 1) {
                int nbr = -1;
                for (int v : adj[u]) if (!removed[v]) { nbr = v; break; }
                selected[u] = 1; remove_node(u);
                if (nbr != -1) remove_node(nbr);
            }
        }

        std::vector<uint8_t> vis(N, 0);
        for (int root = 0; root < N; ++root) {
            if (removed[root] || vis[root]) continue;
            std::vector<int> comp, bfs = {root};
            vis[root] = 1;
            for (size_t bi = 0; bi < bfs.size(); ++bi) {
                int u = bfs[bi]; comp.push_back(u);
                for (int v : adj[u]) if (!removed[v] && !vis[v]) { vis[v] = 1; bfs.push_back(v); }
            }
            int M = (int)comp.size();
            std::vector<int> local_id(N, -1);
            for (int i = 0; i < M; ++i) local_id[comp[i]] = i;
            std::vector<std::vector<int>> ladj(M);
            for (int i = 0; i < M; ++i)
                for (int v : adj[comp[i]]) if (!removed[v]) ladj[i].push_back(local_id[v]);

            auto local_climb = [&](std::vector<int>& sel, std::vector<int>& ccnt, int& cnt) {
                bool imp = true;
                while (imp) {
                    imp = false;
                    for (int u = 0; u < M; ++u) if (!sel[u] && ccnt[u] == 0) {
                        sel[u] = 1; ++cnt; for (int w : ladj[u]) ++ccnt[w]; imp = true;
                    }
                    for (int u = 0; u < M; ++u) {
                        if (!sel[u]) continue;
                        int c1 = -1, c2 = -1;
                        for (int v : ladj[u]) {
                            if (!sel[v] && ccnt[v] == 1) {
                                if (c1 == -1) c1 = v;
                                else {
                                    bool lnk = false;
                                    for (int w : ladj[c1]) if (w == v) { lnk = true; break; }
                                    if (!lnk) { c2 = v; break; }
                                }
                            }
                        }
                        if (c2 != -1) {
                            sel[u] = 0; for (int w : ladj[u]) --ccnt[w];
                            sel[c1] = 1; for (int w : ladj[c1]) ++ccnt[w];
                            sel[c2] = 1; for (int w : ladj[c2]) ++ccnt[w];
                            ++cnt; imp = true;
                        }
                    }
                }
            };

            std::vector<int> best_c(M, 0), best_ccnt(M, 0);
            int best_sz = 0;
            std::vector<int> ord(M);
            std::iota(ord.begin(), ord.end(), 0);
            std::mt19937 rng(123);

            // Structured Pinwheel + Spatial + Wall-Sweep + Corner orders
            for (int trial = 0; trial < 32; ++trial) {
                if (trial < 8) {
                    std::vector<int> pos_in_ord(N, 0);
                    for (int p = 0; p < N; ++p) pos_in_ord[fast_ord[trial][p]] = p;
                    std::sort(ord.begin(), ord.end(), [&](int a, int b) {
                        return pos_in_ord[comp[a]] < pos_in_ord[comp[b]];
                    });
                } else if (trial < 14) {
                    // Geometric cuts & wall sweeps: x asc, x desc, y asc, y desc, pinwheel strip weights
                    int m = trial - 8;
                    std::sort(ord.begin(), ord.end(), [&](int a, int b) {
                        int ga = comp[a], gb = comp[b];
                        double ka = 0, kb = 0;
                        if (m == 0) { ka = x[ga]; kb = x[gb]; }
                        else if (m == 1) { ka = -x[ga]; kb = -x[gb]; }
                        else if (m == 2) { ka = y[ga]; kb = y[gb]; }
                        else if (m == 3) { ka = -y[ga]; kb = -y[gb]; }
                        else if (m == 4) {
                            ka = (corner[ga]==1? ci[ga]: corner[ga]==2? cj[ga]: cheb[ga]+50);
                            kb = (corner[gb]==1? ci[gb]: corner[gb]==2? cj[gb]: cheb[gb]+50);
                        } else {
                            ka = (corner[ga]==1? cj[ga]: corner[ga]==2? ci[ga]: cheb[ga]+50);
                            kb = (corner[gb]==1? cj[gb]: corner[gb]==2? ci[gb]: cheb[gb]+50);
                        }
                        if (ka != kb) return ka < kb;
                        return corner[ga] < corner[gb];
                    });
                } else if (trial == 14) {
                    std::sort(ord.begin(), ord.end(), [&](int a, int b) { return ladj[a].size() < ladj[b].size(); });
                } else {
                    std::shuffle(ord.begin(), ord.end(), rng);
                }
                std::vector<int> sel(M, 0), ccnt(M, 0);
                int cnt = 0;
                for (int u : ord) if (ccnt[u] == 0) { sel[u] = 1; ++cnt; for (int v : ladj[u]) ++ccnt[v]; }
                local_climb(sel, ccnt, cnt);
                if (cnt > best_sz) { best_sz = cnt; best_c = std::move(sel); best_ccnt = std::move(ccnt); }
            }

            // Tabu plateau + (1-1)/(2-3) search on large components
            if (M > 8) {
                std::vector<int> sel = best_c, ccnt = best_ccnt;
                int cnt = best_sz;
                std::vector<int> tabu(M, -1);
                for (int step = 0; step < 350; ++step) {
                    if (step % 70 == 0 && step > 0) {
                        sel = best_c; ccnt = best_ccnt; cnt = best_sz;
                        // Perturb a whole row/col strip OR local neighborhood!
                        int piv = rng() % M;
                        int target_c = corner[comp[piv]];
                        int by_row = rng() % 2;
                        int val = by_row ? ci[comp[piv]] : cj[comp[piv]];
                        for (int u = 0; u < M; ++u) {
                            if (sel[u] && corner[comp[u]] == target_c && ((by_row ? ci[comp[u]] : cj[comp[u]]) == val || (rng() % 5 == 0))) {
                                sel[u] = 0; --cnt; for (int w : ladj[u]) --ccnt[w];
                            }
                        }
                        std::vector<int> free_u;
                        for (int u = 0; u < M; ++u) if (!sel[u] && ccnt[u] == 0) free_u.push_back(u);
                        std::shuffle(free_u.begin(), free_u.end(), rng);
                        for (int u : free_u) if (ccnt[u] == 0) { sel[u] = 1; ++cnt; for (int w : ladj[u]) ++ccnt[w]; }
                        local_climb(sel, ccnt, cnt);
                        if (cnt > best_sz) { best_sz = cnt; best_c = sel; best_ccnt = ccnt; }
                        continue;
                    }
                    // Pick random non-selected non-tabu node with ccnt == 1 (plateau move!), swap in!
                    std::vector<int> cands;
                    for (int u = 0; u < M; ++u) if (!sel[u] && ccnt[u] == 1 && tabu[u] <= step) cands.push_back(u);
                    if (cands.empty()) {
                        for (int u = 0; u < M; ++u) if (!sel[u] && ccnt[u] == 2 && tabu[u] <= step) cands.push_back(u);
                    }
                    if (cands.empty()) break;
                    int add_u = cands[rng() % cands.size()];
                    // Drop blocking selected neighbors of add_u
                    std::vector<int> dropped;
                    for (int w : ladj[add_u]) if (sel[w]) {
                        sel[w] = 0; --cnt; for (int z : ladj[w]) --ccnt[z];
                        tabu[w] = step + 7;
                        dropped.push_back(w);
                    }
                    sel[add_u] = 1; ++cnt; for (int w : ladj[add_u]) ++ccnt[w];
                    // Greedily fill anything freed
                    for (int d_u : dropped) {
                        for (int v : ladj[d_u]) if (!sel[v] && ccnt[v] == 0) {
                            sel[v] = 1; ++cnt; for (int z : ladj[v]) ++ccnt[z];
                        }
                    }
                    local_climb(sel, ccnt, cnt);
                    if (cnt > best_sz) { best_sz = cnt; best_c = sel; best_ccnt = ccnt; }
                }
            }

            if (M <= 48) {
                std::vector<uint64_t> mask_nbr(M, 0);
                for (int i = 0; i < M; ++i) {
                    mask_nbr[i] |= (1ULL << i);
                    for (int v : ladj[i]) mask_nbr[i] |= (1ULL << v);
                }
                uint64_t best_mask = 0;
                bool found_better = false;
                uint64_t all_mask = (M == 64) ? ~0ULL : ((1ULL << M) - 1ULL);
                int steps = 0;
                auto dfs = [&](auto& self, uint64_t rem, uint64_t cur, int cur_sz) -> void {
                    if (++steps > 120000) return;
                    if (cur_sz + __builtin_popcountll(rem) <= best_sz) return;
                    if (rem == 0) {
                        if (cur_sz > best_sz) { best_sz = cur_sz; best_mask = cur; found_better = true; }
                        return;
                    }
                    uint64_t tmp = rem;
                    while (tmp) {
                        int u = __builtin_ctzll(tmp); tmp &= tmp - 1;
                        uint64_t nb = (mask_nbr[u] & rem) ^ (1ULL << u);
                        int dg = __builtin_popcountll(nb);
                        if (dg == 0) { self(self, rem ^ (1ULL << u), cur | (1ULL << u), cur_sz + 1); return; }
                        if (dg == 1) { self(self, rem & ~mask_nbr[u], cur | (1ULL << u), cur_sz + 1); return; }
                    }
                    int best_u = -1, max_d = -1;
                    tmp = rem;
                    while (tmp) {
                        int u = __builtin_ctzll(tmp); tmp &= tmp - 1;
                        int dg = __builtin_popcountll(mask_nbr[u] & rem);
                        if (dg > max_d) { max_d = dg; best_u = u; }
                    }
                    self(self, rem & ~mask_nbr[best_u], cur | (1ULL << best_u), cur_sz + 1);
                    self(self, rem ^ (1ULL << best_u), cur, cur_sz);
                };
                dfs(dfs, all_mask, 0ULL, 0);
                if (found_better) for (int i = 0; i < M; ++i) best_c[i] = (best_mask >> i) & 1ULL;
            }
            for (int i = 0; i < M; ++i) if (best_c[i]) selected[comp[i]] = 1;
        }

        std::vector<Square> out;
        for (int i = 0; i < N; ++i) if (selected[i]) out.push_back({x[i], y[i], 0.0});
        return out;
    }
};

// 9C: Fast Spatial-Hash Cluster+Patch Simulated Annealing
struct SpatialSA {
    int n;
    double L;
    double cell;
    int G;
    std::vector<Square> sq;
    std::vector<PreSq> ps;
    std::vector<std::vector<int>> cells;
    std::vector<int> cell_id;

    int cid(double x, double y) const {
        int gx = std::max(0, std::min(G - 1, (int)(x / cell)));
        int gy = std::max(0, std::min(G - 1, (int)(y / cell)));
        return gy * G + gx;
    }

    void init(const std::vector<Square>& init_sq, double L_) {
        sq = init_sq;
        n = (int)sq.size();
        L = L_;
        cell = 1.5;
        G = (int)std::ceil(L / cell) + 2;
        ps.resize(n);
        cell_id.resize(n);
        cells.assign(G * G, {});
        for (int i = 0; i < n; ++i) {
            ps[i] = make_pre(sq[i].x, sq[i].y, sq[i].th);
            int c = cid(sq[i].x, sq[i].y);
            cell_id[i] = c;
            cells[c].push_back(i);
        }
    }

    void move_cell(int i, const PreSq& np) {
        int oc = cell_id[i], nc = cid(np.x, np.y);
        if (oc != nc) {
            auto& v = cells[oc];
            for (size_t k = 0; k < v.size(); ++k) {
                if (v[k] == i) { v[k] = v.back(); v.pop_back(); break; }
            }
            cells[nc].push_back(i);
            cell_id[i] = nc;
        }
        ps[i] = np;
    }

    inline double wall_energy(const PreSq& p) const {
        double e = 0.0;
        if (p.x - p.hx < 0.0) { double d = p.hx - p.x; e += 1.5 * d * d; }
        if (p.x + p.hx > L)   { double d = p.x + p.hx - L; e += 1.5 * d * d; }
        if (p.y - p.hx < 0.0) { double d = p.hx - p.y; e += 1.5 * d * d; }
        if (p.y + p.hx > L)   { double d = p.y + p.hx - L; e += 1.5 * d * d; }
        return e;
    }

    double single_cost(int i, const PreSq& p) const {
        double e = wall_energy(p);
        int gx = std::max(0, std::min(G - 1, (int)(p.x / cell)));
        int gy = std::max(0, std::min(G - 1, (int)(p.y / cell)));
        int x0 = std::max(0, gx - 1), x1 = std::min(G - 1, gx + 1);
        int y0 = std::max(0, gy - 1), y1 = std::min(G - 1, gy + 1);
        for (int yy = y0; yy <= y1; ++yy) {
            for (int xx = x0; xx <= x1; ++xx) {
                for (int j : cells[yy * G + xx]) {
                    if (j == i) continue;
                    double ov = sat_overlap_depth(p, ps[j]);
                    if (ov > 0.0) e += ov * ov;
                }
            }
        }
        return e;
    }

    double total_energy() const {
        double e = 0.0;
        for (int i = 0; i < n; ++i) {
            e += wall_energy(ps[i]);
            int gx = std::max(0, std::min(G - 1, (int)(ps[i].x / cell)));
            int gy = std::max(0, std::min(G - 1, (int)(ps[i].y / cell)));
            int x0 = std::max(0, gx - 1), x1 = std::min(G - 1, gx + 1);
            int y0 = std::max(0, gy - 1), y1 = std::min(G - 1, gy + 1);
            for (int yy = y0; yy <= y1; ++yy) {
                for (int xx = x0; xx <= x1; ++xx) {
                    for (int j : cells[yy * G + xx]) {
                        if (j <= i) continue;
                        double ov = sat_overlap_depth(ps[i], ps[j]);
                        if (ov > 0.0) e += ov * ov;
                    }
                }
            }
        }
        return e;
    }

    std::pair<double, std::vector<Square>> run(int steps, uint64_t seed) {
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<double> U(0.0, 1.0);
        double cur_E = total_energy();
        double best_E = cur_E;
        auto best_sq = sq;
        double T_start = std::max(1e-4, cur_E * 0.2 + 5e-4);
        double T_end = 1e-9;
        double log_ratio = std::log(T_end / T_start);

        for (int s = 0; s < steps; ++s) {
            double prog = double(s) / steps;
            double T = T_start * std::exp(log_ratio * prog);
            double r = U(rng);

            if (r < 0.72) {
                // Single square move
                int i = rng() % n;
                double sc = 0.14 * (1.0 - 0.85 * prog);
                double nx = sq[i].x + (U(rng) * 2.0 - 1.0) * sc;
                double ny = sq[i].y + (U(rng) * 2.0 - 1.0) * sc;
                double nth = sq[i].th + (U(rng) * 2.0 - 1.0) * sc * 1.2;
                if (U(rng) < 0.08) nth = (rng() % 2 == 0) ? 0.0 : (0.1 + 0.65 * U(rng));
                PreSq np = make_pre(nx, ny, nth);
                double old_c = single_cost(i, ps[i]);
                double new_c = single_cost(i, np);
                double dE = new_c - old_c;
                if (dE <= 0.0 || U(rng) < std::exp(-dE / T)) {
                    sq[i] = {nx, ny, nth};
                    move_cell(i, np);
                    cur_E += dE;
                    if (cur_E < best_E) { best_E = cur_E; best_sq = sq; }
                }
            } else if (r < 0.90) {
                // Rigid patch rotate / shift (disc around random square p)
                int p = rng() % n;
                double rad = 1.4 + 2.2 * U(rng);
                std::vector<int> idx;
                double cx = 0, cy = 0;
                for (int i = 0; i < n; ++i) {
                    if (std::hypot(sq[i].x - sq[p].x, sq[i].y - sq[p].y) <= rad) {
                        idx.push_back(i); cx += sq[i].x; cy += sq[i].y;
                    }
                }
                if (idx.size() >= 2 && idx.size() <= 30) {
                    cx /= idx.size(); cy /= idx.size();
                    double dth = (U(rng) < 0.7) ? ((U(rng) * 2.0 - 1.0) * 0.45 * (1.0 - 0.7 * prog)) : 0.0;
                    double dx = (U(rng) * 2.0 - 1.0) * 0.12 * (1.0 - 0.7 * prog);
                    double dy = (U(rng) * 2.0 - 1.0) * 0.12 * (1.0 - 0.7 * prog);
                    double cd = std::cos(dth), sd = std::sin(dth);

                    // Compute patch old energy (wall + inter-patch+outer) carefully via single_cost minus internal double-counting
                    std::vector<uint8_t> in_patch(n, 0);
                    for (int u : idx) in_patch[u] = 1;
                    auto patch_cost = [&]() {
                        double pe = 0.0;
                        for (int u : idx) {
                            pe += wall_energy(ps[u]);
                            int gx = std::max(0, std::min(G - 1, (int)(ps[u].x / cell)));
                            int gy = std::max(0, std::min(G - 1, (int)(ps[u].y / cell)));
                            for (int yy = std::max(0, gy - 1); yy <= std::min(G - 1, gy + 1); ++yy)
                                for (int xx = std::max(0, gx - 1); xx <= std::min(G - 1, gx + 1); ++xx)
                                    for (int j : cells[yy * G + xx]) {
                                        if (j == u || (in_patch[j] && j < u)) continue;
                                        double ov = sat_overlap_depth(ps[u], ps[j]);
                                        if (ov > 0.0) pe += ov * ov;
                                    }
                        }
                        return pe;
                    };
                    double old_pe = patch_cost();
                    std::vector<Square> old_s(idx.size());
                    std::vector<PreSq> old_p(idx.size());
                    for (size_t k = 0; k < idx.size(); ++k) {
                        int u = idx[k];
                        old_s[k] = sq[u]; old_p[k] = ps[u];
                        double rx = sq[u].x - cx, ry = sq[u].y - cy;
                        double nx = cx + dx + rx * cd - ry * sd;
                        double ny = cy + dy + rx * sd + ry * cd;
                        double nth = sq[u].th + dth;
                        sq[u] = {nx, ny, nth};
                        move_cell(u, make_pre(nx, ny, nth));
                    }
                    double new_pe = patch_cost();
                    double dE = new_pe - old_pe;
                    if (dE <= 0.0 || U(rng) < std::exp(-dE / T)) {
                        cur_E += dE;
                        if (cur_E < best_E) { best_E = cur_E; best_sq = sq; }
                    } else {
                        for (size_t k = 0; k < idx.size(); ++k) {
                            sq[idx[k]] = old_s[k];
                            move_cell(idx[k], old_p[k]);
                        }
                    }
                }
            } else {
                // High-energy square teleport into minimum-cost random site
                int worst = rng() % n;
                double wc = single_cost(worst, ps[worst]);
                for (int t = 0; t < 8; ++t) {
                    int cand = rng() % n;
                    double cc = single_cost(cand, ps[cand]);
                    if (cc > wc) { wc = cc; worst = cand; }
                }
                if (wc > 1e-9) {
                    double best_nx = sq[worst].x, best_ny = sq[worst].y, best_nth = sq[worst].th, best_nc = wc;
                    for (int t = 0; t < 16; ++t) {
                        double nx = 0.6 + (L - 1.2) * U(rng);
                        double ny = 0.6 + (L - 1.2) * U(rng);
                        double nth = U(rng) * 0.5 * PI;
                        PreSq np = make_pre(nx, ny, nth);
                        double nc = single_cost(worst, np);
                        if (nc < best_nc) { best_nc = nc; best_nx = nx; best_ny = ny; best_nth = nth; }
                    }
                    double dE = best_nc - wc;
                    if (dE <= 0.0 || U(rng) < std::exp(-dE / T)) {
                        sq[worst] = {best_nx, best_ny, best_nth};
                        move_cell(worst, make_pre(best_nx, best_ny, best_nth));
                        cur_E += dE;
                        if (cur_E < best_E) { best_E = cur_E; best_sq = sq; }
                    }
                }
            }
            if (s % 50000 == 49999) cur_E = total_energy();
        }
        return {best_E, best_sq};
    }
};

void save_csv(const std::string& path, int n, const std::vector<Square>& sq) {
    double minx = 1e18, miny = 1e18;
    for (const auto& s : sq) {
        double hx = 0.5 * (std::abs(std::cos(s.th)) + std::abs(std::sin(s.th)));
        minx = std::min(minx, s.x - hx); miny = std::min(miny, s.y - hx);
    }
    std::ofstream f(path);
    f << std::setprecision(17) << "id,n,square_id,x,y,angle\n";
    for (int i = 0; i < n; ++i) {
        f << i << "," << n << "," << i << ","
          << (sq[i].x - minx + 1e-12) << "," << (sq[i].y - miny + 1e-12) << "," << sq[i].th << "\n";
    }
}

// Insert 1 extra square into the cleanest void of a (k^2)-size packing
std::vector<Square> insert_extra_square(const std::vector<Square>& sq, double L) {
    std::vector<PreSq> ps; ps.reserve(sq.size());
    for (const auto& s : sq) ps.push_back(make_pre(s.x, s.y, s.th));
    double best_c = 1e18;
    Square best_s{L * 0.5, L * 0.5, 0.3};
    double angles[6] = {0.0, 0.25, 0.40, 0.52, 0.64, 0.7854};
    for (int ix = 1; ix <= 38; ++ix) {
        double cx = 0.6 + (L - 1.2) * ix / 39.0;
        for (int iy = 1; iy <= 38; ++iy) {
            double cy = 0.6 + (L - 1.2) * iy / 39.0;
            for (double th : angles) {
                PreSq np = make_pre(cx, cy, th);
                if (cx - np.hx < 0.0 || cx + np.hx > L || cy - np.hx < 0.0 || cy + np.hx > L) continue;
                double e = 0.0;
                for (const auto& p : ps) {
                    double ov = sat_overlap_depth(np, p);
                    if (ov > 0.0) e += ov * ov;
                }
                if (e < best_c) { best_c = e; best_s = {cx, cy, th}; }
            }
        }
    }
    auto out = sq;
    out.push_back(best_s);
    return out;
}

int main() {
    omp_set_num_threads(7);

    std::map<int, std::vector<Square>> merged;
    {
        std::ifstream f("poc/pell_corrugation/data/border-scan/merged.csv");
        std::string line; std::getline(f, line);
        while (std::getline(f, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            std::stringstream ss(line);
            std::string a, b, c, xs, ys, as;
            std::getline(ss, a, ','); std::getline(ss, b, ','); std::getline(ss, c, ',');
            std::getline(ss, xs, ','); std::getline(ss, ys, ','); std::getline(ss, as, ',');
            merged[std::stoi(b)].push_back({std::stod(xs), std::stod(ys), std::stod(as)});
        }
    }
    std::map<int, double> L_front_map = {
        {401, 20.526135333409}, {442, 21.5229229397321}, {530, 23.517304919934}
    };

    // 9A Calibration on 401, 598, 626
    for (int ncal : {401, 598, 626}) {
        auto sq = merged[ncal];
        double minx = 1e18, miny = 1e18;
        for (const auto& s : sq) {
            double hx = 0.5 * (std::abs(std::cos(s.th)) + std::abs(std::sin(s.th)));
            minx = std::min(minx, s.x - hx); miny = std::min(miny, s.y - hx);
        }
        double L = 0;
        std::vector<PreSq> T;
        for (auto& s : sq) {
            s.x -= minx; s.y -= miny;
            double hx = 0.5 * (std::abs(std::cos(s.th)) + std::abs(std::sin(s.th)));
            L = std::max({L, s.x + hx, s.y + hx});
            if (std::abs(std::sin(2.0 * s.th)) > 0.05) T.push_back(make_pre(s.x, s.y, s.th));
        }
        Grid4C g; g.init(L, -1e-6);
        std::vector<uint8_t> alive; int acnt;
        g.compute_alive(T, -1e-6, alive, acnt);
        auto up = g.solve_mis_exact(alive);
        std::cout << "9A Pinwheel+Tabu ExactMIS n=" << ncal << " tilt=" << T.size()
                  << " up=" << up.size() << " tot=" << (T.size() + up.size()) << "\n" << std::flush;
    }

    std::ofstream jout("poc/swarm_angles/docs/idea9_results.json");
    jout << "[\n";
    std::vector<int> targets = {401, 442, 530};

    for (size_t ti = 0; ti < targets.size(); ++ti) {
        int n = targets[ti];
        int k = (int)std::floor(std::sqrt((double)(n - 1)));
        auto [pen_inc, L_ours] = sat_max_overlap_and_bbox(merged[n]);
        double L_front = L_front_map[n];
        double L_probe = L_ours - 0.001;
        double half = 0.5 * L_probe;
        double t0 = omp_get_wtime();

        // Normalize merged[n] to [0, L_ours]^2
        std::vector<Square> base_inc = merged[n];
        {
            double minx = 1e18, miny = 1e18;
            for (const auto& s : base_inc) {
                double hx = 0.5 * (std::abs(std::cos(s.th)) + std::abs(std::sin(s.th)));
                minx = std::min(minx, s.x - hx); miny = std::min(miny, s.y - hx);
            }
            for (auto& s : base_inc) { s.x -= minx; s.y -= miny; }
        }

        // 9B: Discrete geometric scan (Rotated Rect Cores, 4-Edge C4 Wall Pockets, Corner Wedges)
        Grid4C grid; grid.init(L_probe, 0.0);
        struct Cand { int score; std::string tag; std::vector<Square> tsq; };
        std::vector<std::vector<Cand>> thread_top(7);
        std::vector<std::vector<PreSq>> T_buf(7);
        std::vector<std::vector<uint8_t>> alive_buf(7), blocked_buf(7);

        auto push_cand = [&](int tid, int fast_sc, const std::string& tag, std::vector<Square>&& tsq) {
            auto& v = thread_top[tid];
            if ((int)v.size() < 12 || fast_sc > v.back().score) {
                v.push_back({fast_sc, tag, std::move(tsq)});
                std::sort(v.begin(), v.end(), [](const Cand& a, const Cand& b) { return a.score > b.score; });
                if (v.size() > 12) v.pop_back();
            }
        };
        auto eval_fast = [&](int tid, const std::vector<Square>& tsq) -> int {
            auto& T = T_buf[tid]; T.clear();
            for (const auto& q : tsq) T.push_back(make_pre(q.x, q.y, q.th));
            int acnt;
            grid.compute_alive(T, 0.0, alive_buf[tid], acnt);
            return (int)tsq.size() + grid.fast_greedy_lb(alive_buf[tid], blocked_buf[tid]);
        };

        std::vector<double> th_9b = {2.0, 3.76, 5.5, 7.5, 10.58, 12.96, 15.0, 18.435, 22.0, 26.565, 30.0, 36.87};
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();

            // 9B-1: Small/Medium-Angle Rotated Rectangular Cores (U x V) + shifts
            #pragma omp for schedule(dynamic, 1)
            for (size_t ai = 0; ai < th_9b.size(); ++ai) {
                double th_d = th_9b[ai], th = th_d * PI / 180.0;
                double c = std::cos(th), s = std::sin(th), hx = 0.5 * (c + s);
                for (int U = 2; U <= k - 2; ++U) {
                    for (int V = 2; V <= std::min(U, k - 2); ++V) {
                        if (U * V >= n) continue;
                        if (0.5 * (U - 1) * c + 0.5 * (V - 1) * s + hx > half - 0.5) continue;
                        for (double sx : {-1.0, 0.0, 1.0}) {
                            for (double sy : {-1.0, 0.0}) {
                                if (sx < -0.1 && sy < -0.1) continue;
                                double cx = half + sx, cy = half + sy;
                                std::vector<Square> tsq; tsq.reserve(U * V);
                                bool ok = true;
                                for (int i = 0; i < U && ok; ++i) {
                                    double ui = i - 0.5 * (U - 1);
                                    for (int j = 0; j < V; ++j) {
                                        double vj = j - 0.5 * (V - 1);
                                        double qx = cx + ui * c - vj * s, qy = cy + ui * s + vj * c;
                                        if (qx - hx < -1e-6 || qx + hx > L_probe + 1e-6 ||
                                            qy - hx < -1e-6 || qy + hx > L_probe + 1e-6) { ok = false; break; }
                                        tsq.push_back({qx, qy, th});
                                    }
                                }
                                if (!ok) continue;
                                int sc = eval_fast(tid, tsq);
                                std::ostringstream os;
                                os << "9B_rect " << U << "x" << V << " th=" << th_d << " s=(" << sx << "," << sy << ")";
                                push_cand(tid, sc, os.str(), std::move(tsq));
                            }
                        }
                    }
                }

                // 9B-2: C4 4-Edge Tilted Strips (1..3 cols x M rows hugging walls around free Grid4C interior)
                for (int w1 : {1, 2}) {
                    for (int m2 = 3; m2 <= 14; m2 += 2) {
                        for (double d_wall : {hx + 0.001, 1.0 + hx + 0.001, 2.0 + hx + 0.001}) {
                            for (double shift : {-2.5, -1.0, 0.0, 1.0, 2.5}) {
                                double rx = half - d_wall - 0.5 * (w1 - 1) * c;
                                double ry = shift;
                                if (rx <= 1.0) continue;
                                std::vector<Square> tsq;
                                bool ok = true;
                                for (int i = 0; i < w1 && ok; ++i) {
                                    double ui = i - 0.5 * (w1 - 1);
                                    for (int j = 0; j < m2 && ok; ++j) {
                                        double vj = j - 0.5 * (m2 - 1);
                                        double dx0 = rx + ui * c - vj * s, dy0 = ry + ui * s + vj * c;
                                        double pts[4][3] = {
                                            {half + dx0, half + dy0, th}, {half - dy0, half + dx0, th + 0.5 * PI},
                                            {half - dx0, half - dy0, th + PI}, {half + dy0, half - dx0, th + 1.5 * PI}
                                        };
                                        for (int r = 0; r < 4; ++r) {
                                            if (pts[r][0] - hx < -1e-6 || pts[r][0] + hx > L_probe + 1e-6 ||
                                                pts[r][1] - hx < -1e-6 || pts[r][1] + hx > L_probe + 1e-6) { ok = false; break; }
                                            tsq.push_back({pts[r][0], pts[r][1], pts[r][2]});
                                        }
                                    }
                                }
                                if (!ok) continue;
                                auto [p0, l0] = sat_max_overlap_and_bbox(tsq);
                                if (p0 > 1e-6 || l0 > L_probe + 1e-6) continue;
                                int sc = eval_fast(tid, tsq);
                                std::ostringstream os;
                                os << "9B_4edge " << w1 << "x" << m2 << " th=" << th_d << " dw=" << d_wall << " sh=" << shift;
                                push_cand(tid, sc, os.str(), std::move(tsq));
                            }
                        }
                    }
                }

                // 9B-3: Asymmetric Single / Dual Corner Pockets (w=2..4 cols of m2 sheared along sec(th) at wall/corner)
                double sec = 1.0 / c;
                for (int w : {2, 3, 4}) {
                    for (int m2 = 3; m2 <= k - 2; ++m2) {
                        for (double x0 : {0.0, 1.0, 2.0, 3.0}) {
                            for (double y0 : {0.0, 2.0, 5.0, 8.0}) {
                                if (y0 + m2 * sec > L_probe) continue;
                                for (int sym : {0, 1}) {
                                    std::vector<Square> tsq;
                                    bool ok = true;
                                    for (int col = 0; col < w && ok; ++col) {
                                        double qx = x0 + hx + col * c;
                                        for (int row = 0; row < m2; ++row) {
                                            double qy = y0 + hx + row * sec + col * s;
                                            if (qx + hx > L_probe + 1e-6 || qy + hx > L_probe + 1e-6) { ok = false; break; }
                                            tsq.push_back({qx, qy, th});
                                            if (sym == 1) tsq.push_back({L_probe - qx, L_probe - qy, th});
                                        }
                                    }
                                    if (!ok || (int)tsq.size() >= n) continue;
                                    if (sym == 1) {
                                        auto [p0, l0] = sat_max_overlap_and_bbox(tsq);
                                        if (p0 > 1e-6 || l0 > L_probe + 1e-6) continue;
                                    }
                                    int sc = eval_fast(tid, tsq);
                                    std::ostringstream os;
                                    os << "9B_pkt " << w << "x" << m2 << " th=" << th_d << " xy=(" << x0 << "," << y0 << ") s=" << sym;
                                    push_cand(tid, sc, os.str(), std::move(tsq));
                                }
                            }
                        }
                    }
                }
            }
        }

        // Exact MIS on top 25 9B candidates
        std::vector<Cand> merged_top;
        for (int t = 0; t < 7; ++t) merged_top.insert(merged_top.end(), thread_top[t].begin(), thread_top[t].end());
        std::sort(merged_top.begin(), merged_top.end(), [](const Cand& a, const Cand& b) { return a.score > b.score; });
        if (merged_top.size() > 25) merged_top.resize(25);

        int best_9b_cap = 0;
        std::string best_9b_tag;
        std::vector<Square> best_9b_full;
        for (const auto& cnd : merged_top) {
            std::vector<PreSq> T;
            for (const auto& q : cnd.tsq) T.push_back(make_pre(q.x, q.y, q.th));
            std::vector<uint8_t> alive; int acnt;
            grid.compute_alive(T, 0.0, alive, acnt);
            auto up = grid.solve_mis_exact(alive);
            int cap = (int)cnd.tsq.size() + (int)up.size();
            if (cap > best_9b_cap) {
                best_9b_cap = cap;
                best_9b_tag = cnd.tag + " up=" + std::to_string(up.size());
                best_9b_full = cnd.tsq;
                best_9b_full.insert(best_9b_full.end(), up.begin(), up.end());
            }
        }

        double best_win_L = L_ours;
        std::string best_win_src = "none";

        if (best_9b_cap >= n) {
            std::vector<Square> cand(best_9b_full.begin(), best_9b_full.begin() + n);
            double pL = polish_and_shrink(cand, L_probe, 500);
            if (pL < best_win_L - 1e-7) {
                best_win_L = pL; best_win_src = "9B:" + best_9b_tag;
                save_csv("poc/swarm_angles/data/cand_n" + std::to_string(n) + ".csv", n, cand);
            }
        }

        // 9C: Multi-Start Spatial-Hash Cluster+Patch SA + Adam Polish (42 workers!)
        std::vector<std::vector<Square>> sa_seeds;
        double scale_to_front = L_front / L_ours;
        double scale_mid = (0.5 * (L_front + L_ours)) / L_ours;

        // Seed 0..7: Scaled incumbent + cluster angle kicks
        for (int mode = 0; mode < 8; ++mode) {
            auto s = base_inc;
            double sc = (mode % 2 == 0) ? scale_mid : scale_to_front;
            for (auto& q : s) { q.x *= sc; q.y *= sc; }
            std::mt19937 g(1000 + mode);
            if (mode >= 2) {
                // Rotate tilted cluster(s) or re-angle half of tilted squares
                double dth = ((mode / 2) == 1 ? -0.18 : (mode / 2) == 2 ? 0.25 : -0.35);
                for (auto& q : s) if (std::abs(std::sin(2.0 * q.th)) > 0.05) q.th += dth;
            }
            sa_seeds.push_back(std::move(s));
        }

        // Seed 8..13: Perturbed k x k grid + 1 inserted extra square + random patch tilts
        for (int mode = 0; mode < 6; ++mode) {
            std::vector<Square> grid_k;
            double sp = (L_probe - 0.5) / (k - 0.5);
            for (int i = 0; i < k; ++i)
                for (int j = 0; j < k; ++j)
                    grid_k.push_back({0.5 + i * (L_probe - 1.0) / (k - 1), 0.5 + j * (L_probe - 1.0) / (k - 1), 0.0});
            (void)sp;
            auto s = insert_extra_square(grid_k, L_probe);
            // Tilt a 4x4 patch in corner or center or middle
            double px = (mode % 3 == 0) ? 3.5 : (mode % 3 == 1) ? half : L_probe - 3.5;
            double py = (mode % 2 == 0) ? 3.5 : half;
            double ang = (mode < 3) ? 0.32 : 0.52;
            for (auto& q : s) if (std::abs(q.x - px) < 2.1 && std::abs(q.y - py) < 2.1) q.th = ang;
            sa_seeds.push_back(std::move(s));
        }

        // Seed 14..19: Top 9B def=1 (if size <= n) + inserted square!
        if (!best_9b_full.empty() && (int)best_9b_full.size() == n - 1) {
            auto s9 = insert_extra_square(best_9b_full, L_probe);
            for (int r = 0; r < 6; ++r) sa_seeds.push_back(s9);
        } else {
            for (int r = 0; r < 6; ++r) sa_seeds.push_back(base_inc);
        }

        // Run 42 SA+Adam tasks in parallel (2 repeats per seed across L_targets)
        int n_sa = 42;
        std::vector<double> sa_best_L(n_sa, 1e18);
        std::vector<double> sa_best_E(n_sa, 1e18);
        std::vector<std::vector<Square>> sa_best_sq(n_sa);

        #pragma omp parallel for schedule(dynamic, 1)
        for (int wi = 0; wi < n_sa; ++wi) {
            double L_tgt = (wi % 3 == 0) ? L_front : (wi % 3 == 1) ? (L_ours - 0.002) : (L_ours - 0.004);
            auto init_s = sa_seeds[wi % sa_seeds.size()];
            // Fit inside [0, L_tgt] cleanly
            double maxc = 0.0;
            for (const auto& q : init_s) {
                double hx = 0.5 * (std::abs(std::cos(q.th)) + std::abs(std::sin(q.th)));
                maxc = std::max({maxc, q.x + hx, q.y + hx});
            }
            if (maxc > L_tgt) {
                double f = L_tgt / maxc;
                for (auto& q : init_s) { q.x *= f; q.y *= f; }
            }
            SpatialSA sa;
            sa.init(init_s, L_tgt);
            auto [be, bsq] = sa.run(350000, 7777ULL * (n + 1) + wi);
            sa_best_E[wi] = be;
            if (be < 0.02) {
                double pL = polish_and_shrink(bsq, L_tgt + 0.001, 500);
                if (pL < 1e17) {
                    sa_best_L[wi] = pL;
                    sa_best_sq[wi] = std::move(bsq);
                } else {
                    double pL2 = polish_and_shrink(bsq, L_ours - 0.0002, 500);
                    sa_best_L[wi] = pL2;
                    sa_best_sq[wi] = std::move(bsq);
                }
            }
        }

        double min_sa_E = 1e18;
        for (int wi = 0; wi < n_sa; ++wi) {
            min_sa_E = std::min(min_sa_E, sa_best_E[wi]);
            if (sa_best_L[wi] < best_win_L - 1e-7) {
                best_win_L = sa_best_L[wi];
                best_win_src = "9C_SA_w" + std::to_string(wi);
                save_csv("poc/swarm_angles/data/cand_n" + std::to_string(n) + ".csv", n, sa_best_sq[wi]);
            }
        }

        double dt = omp_get_wtime() - t0;
        std::cout << "n=" << n << " (L_ours=" << std::setprecision(6) << L_ours << ", L_front=" << L_front << "): "
                  << "9B best_cap=" << best_9b_cap << " [" << best_9b_tag << "] | "
                  << "9C min_sa_E=" << std::scientific << std::setprecision(2) << min_sa_E
                  << std::fixed << std::setprecision(6)
                  << " best_win_L=" << best_win_L << " (" << best_win_src << ") in " << std::setprecision(2) << dt << "s\n" << std::flush;

        jout << "  {\"n\": " << n << ", \"L_ours\": " << std::setprecision(12) << L_ours
             << ", \"L_front\": " << L_front << ", \"best_9b_cap\": " << best_9b_cap
             << ", \"best_9b_tag\": \"" << best_9b_tag << "\", \"min_sa_E\": " << min_sa_E
             << ", \"best_win_L\": " << best_win_L << ", \"win_src\": \"" << best_win_src
             << "\", \"seconds\": " << std::setprecision(2) << dt << "}"
             << (ti + 1 < targets.size() ? "," : "") << "\n";
    }
    jout << "]\n";
    return 0;
}
