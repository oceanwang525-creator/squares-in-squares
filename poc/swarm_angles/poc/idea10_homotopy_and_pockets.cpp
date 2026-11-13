// Idea 10: a-Homotopy liquid-to-jamming inflation (n=401,626,842,442,530),
// n=971 oblique-cut reflected wall pockets, bent sheared bands (Idea 7B + 9A Tabu MIS),
// and Grid9 ({0, f/2, f}^2) + sliding border polish for D=1 roots.
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

// Fast O(1) spatial-hash state for SA with adjustable side length `side`
struct SpatialSA {
    int n;
    double L, side, half;
    double cell_sz;
    int grid_n;
    vector<Sq> sq;
    vector<double> c, s, hx;
    vector<int> cell_id;
    vector<vector<int>> cells;
    double total_E = 0.0;

    void init(const vector<Sq>& init_sq, double box_L, double sq_side) {
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
        for (int i = 0; i < n; ++i) {
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

    inline double wall_E(int i) const {
        double e = 0.0;
        double v;
        if ((v = hx[i] - sq[i].x) > 0.0) e += v * v;
        if ((v = sq[i].x + hx[i] - L) > 0.0) e += v * v;
        if ((v = hx[i] - sq[i].y) > 0.0) e += v * v;
        if ((v = sq[i].y + hx[i] - L) > 0.0) e += v * v;
        return e;
    }

    double local_E(int i, double xi, double yi, double ci, double si, double hxi) const {
        double e = 0.0, v;
        if ((v = hxi - xi) > 0.0) e += v * v;
        if ((v = xi + hxi - L) > 0.0) e += v * v;
        if ((v = hxi - yi) > 0.0) e += v * v;
        if ((v = yi + hxi - L) > 0.0) e += v * v;

        int gx = max(0, min(grid_n - 1, (int)(xi / cell_sz) + 1));
        int gy = max(0, min(grid_n - 1, (int)(yi / cell_sz) + 1));
        for (int dy = -1; dy <= 1; ++dy) {
            int yy = gy + dy;
            if (yy < 0 || yy >= grid_n) continue;
            for (int dx = -1; dx <= 1; ++dx) {
                int xx = gx + dx;
                if (xx < 0 || xx >= grid_n) continue;
                for (int j : cells[yy * grid_n + xx]) {
                    if (j == i) continue;
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
            e += wall_E(i);
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

    void run_sa(int iters, double T0, double T1, double step_xy, double step_th, RNG& rng,
                bool allow_align = true) {
        double invN = 1.0 / max(1, iters);
        double log_ratio = log(max(1e-12, T1) / max(1e-12, T0));
        double best_E = total_E;
        vector<Sq> best_sq = sq;

        for (int it = 0; it < iters; ++it) {
            double frac = it * invN;
            double T = T0 * exp(frac * log_ratio);
            double sc = 1.0 - 0.85 * frac;

            int i = rng.irand(n);
            double old_e = local_E(i, sq[i].x, sq[i].y, c[i], s[i], hx[i]);
            // Bias slightly toward overlapping squares
            if (old_e == 0.0 && rng.u01() < 0.55) {
                int j = rng.irand(n);
                double ej = local_E(j, sq[j].x, sq[j].y, c[j], s[j], hx[j]);
                if (ej > 0.0) { i = j; old_e = ej; }
            }

            double nx = sq[i].x + rng.sym() * step_xy * sc;
            double ny = sq[i].y + rng.sym() * step_xy * sc;
            double nth = sq[i].th;
            double mv = rng.u01();
            if (allow_align && mv < 0.14) {
                // Snap orientation to a nearby square's angle or 0
                int k = rng.irand(n);
                nth = (rng.u01() < 0.25) ? 0.0 : sq[k].th + rng.sym() * 0.02;
            } else if (mv < 0.70) {
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
            init(best_sq, L, side);
        }
    }
};

// Adam continuous gradient optimizer with spatial hash
double adam_relax(vector<Sq>& sq, double L, double side, int max_steps,
                  double lr0 = 0.02, bool freeze_angles = false, double eps_stop = 1e-13) {
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
        vector<double> c(n), s(n), hx(n), dc(n), ds(n), dhx(n);
        vector<double> gx(n, 0.0), gy(n, 0.0), gth(n, 0.0);
        for (int i = 0; i < n; ++i) {
            c[i] = cos(sq[i].th);
            s[i] = sin(sq[i].th);
            double sc_c = (c[i] >= 0 ? 1.0 : -1.0);
            double sc_s = (s[i] >= 0 ? 1.0 : -1.0);
            hx[i] = half * (fabs(c[i]) + fabs(s[i]));
            dc[i] = -s[i];
            ds[i] = c[i];
            dhx[i] = half * (sc_c * (-s[i]) + sc_s * c[i]);
        }
        double E = 0.0;
        for (int i = 0; i < n; ++i) {
            double v1;
            if ((v1 = hx[i] - sq[i].x) > 0.0) {
                E += v1 * v1; gx[i] -= 2 * v1; gth[i] += 2 * v1 * dhx[i];
            }
            if ((v1 = sq[i].x + hx[i] - L) > 0.0) {
                E += v1 * v1; gx[i] += 2 * v1; gth[i] += 2 * v1 * dhx[i];
            }
            if ((v1 = hx[i] - sq[i].y) > 0.0) {
                E += v1 * v1; gy[i] -= 2 * v1; gth[i] += 2 * v1 * dhx[i];
            }
            if ((v1 = sq[i].y + hx[i] - L) > 0.0) {
                E += v1 * v1; gy[i] += 2 * v1; gth[i] += 2 * v1 * dhx[i];
            }

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
                        double d_pen_d_rad = 1.0;
                        // da/dthi = -s[i]*c[j] + c[i]*s[j] = b, db/dthi = -a
                        // da/dthj = -b, db/dthj = a
                        double drad_dthi = half * (sa * b - sb * a);
                        double drad_dthj = half * (-sa * b + sb * a);

                        double sgn = (p_axes[kmin] >= 0 ? 1.0 : -1.0);
                        double dpx = 0, dpy = 0, dthi_ax = 0, dthj_ax = 0;
                        if (kmin == 0) {
                            dpx = c[i]; dpy = s[i];
                            dthi_ax = -dx * s[i] + dy * c[i];
                        } else if (kmin == 1) {
                            dpx = -s[i]; dpy = c[i];
                            dthi_ax = -dx * c[i] - dy * s[i];
                        } else if (kmin == 2) {
                            dpx = c[j]; dpy = s[j];
                            dthj_ax = -dx * s[j] + dy * c[j];
                        } else {
                            dpx = -s[j]; dpy = c[j];
                            dthj_ax = -dx * c[j] - dy * s[j];
                        }
                        double w2 = 2.0 * pen;
                        gx[i] += w2 * sgn * dpx;
                        gy[i] += w2 * sgn * dpy;
                        gx[j] -= w2 * sgn * dpx;
                        gy[j] -= w2 * sgn * dpy;
                        gth[i] += w2 * (d_pen_d_rad * drad_dthi - sgn * dthi_ax);
                        gth[j] += w2 * (d_pen_d_rad * drad_dthj - sgn * dthj_ax);
                    }
                }
            }
        }
        if (E < best_E) {
            best_E = E;
            best_sq = sq;
        }
        if (E < eps_stop) break;

        b1t *= b1;
        b2t *= b2;
        double lr = lr0 * (0.1 + 0.9 * 0.5 * (1.0 + cos(PI * step / max(1, max_steps))));
        for (int i = 0; i < n; ++i) {
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

// Polish and shrink L while preserving exact_max_pen == 0
double polish_and_shrink(vector<Sq>& sq, double L_start) {
    double cur_L = L_start;
    auto [p0, bbox0] = exact_max_pen_and_L(sq, 1.0);
    if (p0 > 1e-11) {
        adam_relax(sq, cur_L, 1.0, 900, 0.015, false, 1e-18);
        auto [p1, b1] = exact_max_pen_and_L(sq, 1.0);
        if (p1 > 1e-12) return 1e18;
        cur_L = min(cur_L, b1);
    } else {
        cur_L = bbox0;
    }
    vector<Sq> best_sq = sq;
    double best_L = cur_L;

    // Normalize origin
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
        // Scale centers linearly toward box center first
        double halfL = 0.5 * best_L;
        double sc = (target_L - 1.0) / max(1.0, best_L - 1.0);
        for (auto& q : trial) {
            q.x = 0.5 * target_L + (q.x - halfL) * sc;
            q.y = 0.5 * target_L + (q.y - halfL) * sc;
        }
        adam_relax(trial, target_L, 1.0, 700, 0.01, false, 1e-22);
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
        if (stoi(cols[1]) == target_n) {
            res.push_back({stod(cols[3]), stod(cols[4]), stod(cols[5])});
        }
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

// L-border propagation: add +1 row at top and +1 col at right (+2k+1 squares, +1.0 to L)
vector<Sq> lborder_propagate(const vector<Sq>& base, int k_from, int k_to) {
    vector<Sq> sq = base;
    auto [p, Lcur] = exact_max_pen_and_L(sq, 1.0);
    for (int k = k_from; k < k_to; ++k) {
        // Add k squares along right col x = Lcur + 0.5, y = 0.5 .. k - 0.5
        for (int j = 0; j < k; ++j) sq.push_back({Lcur + 0.5, j + 0.5, 0.0});
        // Add k+1 squares along top row x = 0.5 .. k + 0.5, y = Lcur + 0.5 (check fit!)
        for (int i = 0; i <= k; ++i) sq.push_back({i + 0.5, Lcur + 0.5, 0.0});
        Lcur += 1.0;
    }
    return sq;
}

void save_csv(const string& path, int n, const vector<Sq>& sq) {
    system("mkdir -p poc/swarm_angles/data");
    ofstream out(path);
    out << setprecision(17);
    out << "id,n,square_id,x,y,angle\n";
    for (int i = 0; i < n; ++i) {
        out << i << "," << n << "," << i << "," << sq[i].x << "," << sq[i].y << "," << sq[i].th << "\n";
    }
}

// =========================================================================
// PART 10A: a-Homotopy Liquid-to-Jamming Inflation
// =========================================================================
pair<double, vector<Sq>> run_10a_homotopy(int n, double L_tgt, const vector<Sq>& base_seed, int workers = 28) {
    int k = (int)floor(sqrt((double)(n - 1)) + 0.5);
    double f = L_tgt - k;
    double best_E_global = 1e18;
    vector<Sq> best_sq_global;
    double best_win_L = 1e18;
    vector<Sq> win_sq_global;

    #pragma omp parallel for schedule(dynamic, 1)
    for (int w = 0; w < workers; ++w) {
        RNG rng(100007ULL + (uint64_t)n * 997ULL + (uint64_t)w * 7919ULL);
        vector<Sq> sq;
        int mode = w % 7;

        if (mode == 0 && !base_seed.empty()) {
            // Perturb existing 45-deg solution (`n=401` two 5x5 or `n=626/842` 17x17):
            // Rotate tilted block away from 45 deg + kick all angles by 1..12 deg
            sq = base_seed;
            double dth = (w % 14 - 6) * (PI / 180.0) * 1.8;
            for (auto& q : sq) {
                double deg = fmod(fabs(q.th) * 180.0 / PI, 90.0);
                if (deg > 10.0 && deg < 80.0) {
                    q.th += dth + rng.sym() * 0.06;
                } else {
                    q.th += rng.sym() * 0.04;
                }
                q.x += rng.sym() * 0.08;
                q.y += rng.sym() * 0.08;
            }
        } else if (mode == 1) {
            // Perturbed k x k + 1 with chiral radial or diagonal tilted region
            double th_dom = (18.0 + (w % 9) * 3.0) * PI / 180.0;
            for (int j = 0; j < k; ++j) {
                for (int i = 0; i < k; ++i) {
                    double px = (i + 0.5) * (L_tgt / k);
                    double py = (j + 0.5) * (L_tgt / k);
                    double d_diag = fabs(px - py) / sqrt(2.0);
                    double th = (d_diag < 2.2) ? th_dom : 0.0;
                    sq.push_back({px + rng.sym() * 0.05, py + rng.sym() * 0.05, th + rng.sym() * 0.05});
                }
            }
            sq.push_back({0.5 * L_tgt + rng.sym() * 0.5, 0.5 * L_tgt + rng.sym() * 0.5, th_dom});
        } else if (mode == 2) {
            // Asymmetric corner pocket seed (n=971 style bottom + left tilted strips)
            double th_dom = (19.0 + (w % 7) * 3.2) * PI / 180.0;
            double wp = 2.0 + (w % 4) * 1.2;
            for (int j = 0; j < k; ++j) {
                for (int i = 0; i < k; ++i) {
                    double px = (i + 0.5) * (L_tgt / k);
                    double py = (j + 0.5) * (L_tgt / k);
                    double th = 0.0;
                    if (py < wp && px > 0.28 * L_tgt) th = th_dom;
                    else if (px < wp && py > 0.28 * L_tgt) th = 0.5 * PI - th_dom;
                    sq.push_back({px, py, th + rng.sym() * 0.04});
                }
            }
            sq.push_back({0.5 * L_tgt, 0.5 * L_tgt, rng.u01() * 0.5 * PI});
        } else if (mode == 3) {
            // 4-pocket chiral pinwheel seed around central core
            double th_dom = (16.0 + (w % 8) * 3.5) * PI / 180.0;
            double wp = 1.8 + (w % 4) * 1.0;
            for (int j = 0; j < k; ++j) {
                for (int i = 0; i < k; ++i) {
                    double px = (i + 0.5) * (L_tgt / k);
                    double py = (j + 0.5) * (L_tgt / k);
                    double th = 0.0;
                    if (py < wp && px > wp) th = th_dom;
                    else if (px > L_tgt - wp && py > wp) th = th_dom + 0.5 * PI;
                    else if (py > L_tgt - wp && px < L_tgt - wp) th = th_dom;
                    else if (px < wp && py < L_tgt - wp) th = th_dom + 0.5 * PI;
                    sq.push_back({px, py, th + rng.sym() * 0.05});
                }
            }
            sq.push_back({0.5 * L_tgt, 0.5 * L_tgt, 0.0});
        } else if (mode == 4) {
            // Central rotated block seed (m x m at th in 25..44 deg inside shifted k x k)
            double th_dom = (24.0 + (w % 7) * 3.2) * PI / 180.0;
            double R = 0.35 * L_tgt;
            double cx = 0.5 * L_tgt, cy = 0.5 * L_tgt;
            for (int j = 0; j < k; ++j) {
                for (int i = 0; i < k; ++i) {
                    double px = (i + 0.5) * (L_tgt / k);
                    double py = (j + 0.5) * (L_tgt / k);
                    double u = (px - cx) * cos(th_dom) + (py - cy) * sin(th_dom);
                    double v = -(px - cx) * sin(th_dom) + (py - cy) * cos(th_dom);
                    double th = (fabs(u) < R && fabs(v) < R) ? th_dom : 0.0;
                    sq.push_back({px, py, th + rng.sym() * 0.05});
                }
            }
            sq.push_back({cx + rng.sym(), cy + rng.sym(), th_dom});
        } else if (mode == 5) {
            // Multi-angle fluid grid (a=0.90) with small smooth vortex angle field
            for (int j = 0; j < k; ++j) {
                for (int i = 0; i < k; ++i) {
                    double px = (i + 0.5) * (L_tgt / k);
                    double py = (j + 0.5) * (L_tgt / k);
                    double th = 0.35 * PI * rng.u01();
                    sq.push_back({px + rng.sym() * 0.1, py + rng.sym() * 0.1, th});
                }
            }
            sq.push_back({0.5 * L_tgt, 0.5 * L_tgt, 0.25 * PI});
        } else {
            // Split shifted k x k (SW, NW, SE, NE) + single wall slab
            double th_dom = (20.0 + (w % 7) * 3.5) * PI / 180.0;
            for (int j = 0; j < k; ++j) {
                for (int i = 0; i < k; ++i) {
                    double px = i + 0.5 + (i > k / 2 ? f : 0.0);
                    double py = j + 0.5 + (j > k / 2 ? f : 0.0);
                    double th = (i >= k / 2 - 2 && i <= k / 2 + 2) ? th_dom : 0.0;
                    sq.push_back({min(L_tgt - 0.5, px), min(L_tgt - 0.5, py), th});
                }
            }
            sq.push_back({0.5 * L_tgt, 0.5 * L_tgt, th_dom});
        }

        // a-Homotopy schedule: 0.92 -> 1.00
        double a_sched[8] = {0.92, 0.95, 0.972, 0.986, 0.994, 0.998, 0.9995, 1.0};
        SpatialSA sa;
        for (int st = 0; st < 8; ++st) {
            double a = a_sched[st];
            sa.init(sq, L_tgt, a);
            int sa_steps = (st < 4) ? 75000 : 95000;
            double T0 = (st == 0) ? 0.04 : 0.008 * (1.01 - a);
            double T1 = max(1e-7, 1e-4 * (1.01 - a));
            sa.run_sa(sa_steps, T0, T1, 0.35 * (1.05 - a) + 0.02, 0.45 * (1.05 - a) + 0.03, rng, true);
            sq = sa.sq;
            adam_relax(sq, L_tgt, a, (st < 6 ? 120 : 400), 0.015 * (1.05 - a) + 0.004, false, 1e-13);
        }
        double E_fin = adam_relax(sq, L_tgt, 1.0, 900, 0.01, false, 1e-18);
        auto [mp, mL] = exact_max_pen_and_L(sq, 1.0);

        #pragma omp critical
        {
            if (E_fin < best_E_global) {
                best_E_global = E_fin;
                best_sq_global = sq;
            }
            if (mp <= 1e-11 || E_fin < 1e-9) {
                vector<Sq> pol = sq;
                double wL = polish_and_shrink(pol, L_tgt);
                if (wL < best_win_L) {
                    best_win_L = wL;
                    win_sq_global = pol;
                }
            }
        }
    }
    if (best_win_L < 1e17) return {best_win_L, win_sq_global};
    return {-best_E_global, best_sq_global};
}

// =========================================================================
// PART 10B/C/D: Exact Grid9 ({0, f/2, f}^2) Pinwheel+Tabu MIS + Sliding Fill
// =========================================================================
struct Grid9 {
    double L;
    int k;
    vector<double> gx, gy;
    vector<int> mode;  // 0..8 for (ox_id * 3 + oy_id)
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

    int solve_mis_tabu(const vector<Sq>& tilted, vector<Sq>& out_up, double min_gap = 0.0) const {
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
                double dx = ux - tilted[t].x;
                double rad = 0.5 + thx[t] + min_gap;
                if (fabs(dx) >= rad) continue;
                double dy = uy - tilted[t].y;
                if (fabs(dy) >= rad) continue;
                if (fabs(dx * tc[t] + dy * ts[t]) < rad && fabs(-dx * ts[t] + dy * tc[t]) < rad) {
                    ok = false;
                    break;
                }
            }
            if (ok) cand.push_back(u);
        }

        int nc = (int)cand.size();
        if (nc == 0) { out_up.clear(); return 0; }
        sort(cand.begin(), cand.end(), [&](int a, int b) { return gx[a] < gx[b]; });

        double thr = 1.0 + min_gap - 1e-12;
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

        // Orderings covering classic 4-corner + CW/CCW pinwheels + wall strips
        auto greedy_order = [&](const vector<int>& order) -> vector<int> {
            vector<int> in_s(nc, 0), blk(nc, 0), sel;
            for (int u : order) {
                if (blk[u] == 0) {
                    in_s[u] = 1;
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
                // Penalize half-grid slightly so corners stay cohesive unless needed
                double mid_pen = (ox == 1 || oy == 1) ? 20.0 : 0.0;

                if (om == 0) sc[a] = d + mid_pen;
                else if (om == 1) {
                    // CW pinwheel bias: SW(0,0)->SE(2,0)->NE(2,2)->NW(0,2)
                    double px = gx[u], py = gy[u];
                    int q = (ox == 0 && oy == 0) ? (int)(px) :
                            (ox == 2 && oy == 0) ? (int)(py) :
                            (ox == 2 && oy == 2) ? (int)(L - px) : (int)(L - py);
                    sc[a] = q + 0.01 * d + mid_pen;
                } else if (om == 2) {
                    // CCW pinwheel
                    double px = gx[u], py = gy[u];
                    int q = (ox == 0 && oy == 0) ? (int)(py) :
                            (ox == 0 && oy == 2) ? (int)(px) :
                            (ox == 2 && oy == 2) ? (int)(L - py) : (int)(L - px);
                    sc[a] = q + 0.01 * d + mid_pen;
                } else if (om == 3) {
                    sc[a] = ((md == 0 || md == 8) ? 0.0 : 40.0) + d + mid_pen;
                } else if (om == 4) {
                    sc[a] = ((md == 2 || md == 6) ? 0.0 : 40.0) + d + mid_pen;
                } else if (om == 5) {
                    sc[a] = -min({gx[u], L - gx[u], gy[u], L - gy[u]}) * -1.0 + mid_pen;
                    sc[a] = min({gx[u], L - gx[u], gy[u], L - gy[u]}) + mid_pen;
                } else if (om == 6) {
                    sc[a] = min(gx[u], gy[u]) + mid_pen;
                } else if (om == 7) {
                    sc[a] = min(L - gx[u], L - gy[u]) + mid_pen;
                } else if (om == 8) {
                    sc[a] = (double)adj[a].size() + 0.01 * d + mid_pen;
                } else {
                    sc[a] = d;  // half-grid allowed freely
                }
            }
            sort(ord.begin(), ord.end(), [&](int i, int j) { return sc[i] < sc[j]; });
            auto sel = greedy_order(ord);
            if (sel.size() > best_sel.size()) best_sel = move(sel);
        }

        // Fast (1, 2)-swap + plateau local search
        vector<int> in_s(nc, 0), tight_cnt(nc, 0);
        for (int u : best_sel) in_s[u] = 1;
        for (int u = 0; u < nc; ++u) {
            if (!in_s[u]) continue;
            for (int v : adj[u]) tight_cnt[v]++;
        }
        int cur_sz = (int)best_sel.size();
        // Repeatedly search for any u in S whose removal frees >= 2 mutually non-adjacent neighbors
        bool improved = true;
        int rounds = 0;
        while (improved && rounds < 12) {
            improved = false;
            rounds++;
            for (int u = 0; u < nc; ++u) {
                if (!in_s[u]) continue;
                vector<int> freed;
                for (int v : adj[u]) {
                    if (!in_s[v] && tight_cnt[v] == 1) freed.push_back(v);
                }
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
                    auto add_node = [&](int x) {
                        in_s[x] = 1;
                        for (int w : adj[x]) tight_cnt[w]++;
                    };
                    add_node(add1);
                    add_node(add2);
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

// Sliding-strip greedy upright pack along horizontal rows (or vertical cols) around tilted core:
// For each row j (y = j+0.5 or L-j-0.5), greedily scans [0.5, L-0.5] and places upright squares
// flush against left wall or tilted block edge (catching non-Grid9 border squares like n=364/404/971!).
vector<Sq> merge_sliding_fill(double L, const vector<Sq>& tilted, const vector<Sq>& base_up) {
    vector<Sq> sq = tilted;
    sq.insert(sq.end(), base_up.begin(), base_up.end());

    auto can_place = [&](double px, double py) -> bool {
        if (px < 0.5 - 1e-9 || px > L - 0.5 + 1e-9 || py < 0.5 - 1e-9 || py > L - 0.5 + 1e-9) return false;
        for (const auto& q : sq) {
            double c = cos(q.th), s = sin(q.th), hx = 0.5 * (fabs(c) + fabs(s));
            if (sat_pen_pair(px, py, 1.0, 0.0, 0.5, q.x, q.y, c, s, hx, 0.5, -1e-9) > 0.0) return false;
        }
        return true;
    };

    int k = (int)floor(L);
    double f = L - k;
    vector<double> y_cands, x_cands;
    for (int i = 0; i < k; ++i) {
        y_cands.push_back(i + 0.5);
        y_cands.push_back(i + f + 0.5);
        x_cands.push_back(i + 0.5);
        x_cands.push_back(i + f + 0.5);
    }
    y_cands.push_back(0.5 * L);
    x_cands.push_back(0.5 * L);

    // Fine scan on remaining wall/interface gaps
    for (double py : y_cands) {
        for (int step = 0; step <= (int)((L - 1.0) * 10); ++step) {
            double px = 0.5 + step * 0.1;
            if (can_place(px, py)) sq.push_back({px, py, 0.0});
        }
    }
    for (double px : x_cands) {
        for (int step = 0; step <= (int)((L - 1.0) * 10); ++step) {
            double py = 0.5 + step * 0.1;
            if (can_place(px, py)) sq.push_back({px, py, 0.0});
        }
    }
    return sq;
}

// =========================================================================
// Builders for 10B (Bent/Jogging Bands) & 10C (n=971 Oblique Reflected Pockets)
// =========================================================================
vector<Sq> build_971_oblique_pockets(double L, double th, double W_p, double u_cut, int mode) {
    // Bottom pocket: y in [0, W_p], rotated coords u = x*c + y*s, v = -x*s + y*c
    // Cut inner left edge along u >= u_cut, right edge near x <= L - x_right_margin
    double c = cos(th), s = sin(th), hx = 0.5 * (fabs(c) + fabs(s));
    vector<Sq> p1;
    int max_i = (int)ceil(L * 1.5);
    for (int iu = 0; iu <= max_i; ++iu) {
        for (int iv = -max_i; iv <= max_i; ++iv) {
            double u = u_cut + iu;
            double v = iv;
            double x = u * c - v * s;
            double y = u * s + v * c;
            if (x - hx >= -1e-7 && x + hx <= L + 1e-7 && y - hx >= -1e-7 && y + hx <= W_p + 1e-7) {
                p1.push_back({x, y, th});
            }
        }
    }
    if (p1.empty()) return {};
    vector<Sq> out = p1;
    if (mode == 0) {
        // Adjacent reflection across x = y (n=971 topology: bottom @ th + left @ 90-th)
        for (const auto& q : p1) out.push_back({q.y, q.x, 0.5 * PI - th});
    } else if (mode == 1) {
        // Anti-diagonal reflection across x+y=L (n=908 topology: bottom @ th + right? -> (L-y, L-x, 90-th))
        for (const auto& q : p1) out.push_back({L - q.y, L - q.x, 0.5 * PI - th});
    } else if (mode == 2) {
        // C2 opposite reflection (L-x, L-y, th)
        for (const auto& q : p1) out.push_back({L - q.x, L - q.y, th});
    } else if (mode == 3) {
        // C4 4-pocket chiral frame: rotate 90 deg around box center
        for (const auto& q : p1) {
            out.push_back({L - q.y, q.x, th + 0.5 * PI});
            out.push_back({L - q.x, L - q.y, th});
            out.push_back({q.y, L - q.x, th + 0.5 * PI});
        }
    }
    // Verify tilted set is non-overlapping internally
    auto [pen, bL] = exact_max_pen_and_L(out, 1.0);
    if (pen > 1e-6 || bL > L + 1e-5) return {};
    return out;
}

vector<Sq> build_bent_band(double L, int a1, int a2, int w, double th_mid, double bend_end, double shear, int n_cols_limit) {
    // Diagonal band along SW-NE separating NW/SE (or rotated separating SW/NE)
    // Start around center of diagonal (u45 along x+y, v45 along y-x or x-y)
    double s2 = sqrt(0.5);
    double u_lo = (a1 + 0.5) * sqrt(2.0);
    double u_hi = 2.0 * L * s2 - (a2 + 0.5) * sqrt(2.0);
    double th_band = 0.25 * PI;
    // Actually use rotated coordinates in (th_mid): place w rows along e2, step along e1
    double c = cos(th_mid), s = sin(th_mid), hx = 0.5 * (fabs(c) + fabs(s));
    vector<Sq> out;
    double step_u = 1.0, step_v = 1.0;
    // Place centered along main anti-diagonal NW->SE
    double d_diag = (a1 - a2) * 0.5;
    double cx0 = 0.5 * L + d_diag, cy0 = 0.5 * L + d_diag;
    for (int col = -n_cols_limit; col <= n_cols_limit; ++col) {
        double col_th = th_mid;
        if (abs(col) >= max(2, n_cols_limit - 3)) {
            col_th += (col > 0 ? bend_end : -bend_end);
        }
        double cc = cos(col_th), ss = sin(col_th), chx = 0.5 * (fabs(cc) + fabs(ss));
        for (int row = 0; row < w; ++row) {
            double rv = row - 0.5 * (w - 1);
            double ru = col * step_u + rv * shear;
            double px = cx0 + ru * cos(th_mid) - rv * sin(th_mid);
            double py = cy0 + ru * sin(th_mid) + rv * cos(th_mid);
            if (px - chx < -1e-7 || px + chx > L + 1e-7 || py - chx < -1e-7 || py + chx > L + 1e-7) continue;
            // Verify vs already placed tilted
            bool ok = true;
            for (const auto& q : out) {
                double qc = cos(q.th), qs = sin(q.th), qhx = 0.5 * (fabs(qc) + fabs(qs));
                if (sat_pen_pair(px, py, cc, ss, chx, q.x, q.y, qc, qs, qhx, 0.5, -1e-7) > 0.0) {
                    ok = false; break;
                }
            }
            if (ok) out.push_back({px, py, col_th});
        }
    }
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

int main() {
    omp_set_num_threads(7);
    auto t0 = chrono::steady_clock::now();

    struct Target {
        int n;
        double L_base;   // min known (e.g. per_n_gap L)
        double L_front;
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
    auto s842 = lborder_propagate(s626, 25, 29);
    auto [p842, L842] = exact_max_pen_and_L(s842, 1.0);
    cout << fixed << setprecision(9)
         << "L-prop check s842: size=" << s842.size() << " pen=" << scientific << p842
         << fixed << " L=" << L842 << "\n" << flush;

    map<int, vector<Sq>> seeds;
    seeds[401] = s401;
    seeds[442] = lborder_propagate(s401, 20, 21);
    seeds[530] = lborder_propagate(s401, 20, 23);
    seeds[626] = s626;
    seeds[842] = s842;

    // Quick verify: calibrate Grid9 on n=626 tilted (289) -> should get >= 337 (total >= 626)!
    {
        vector<Sq> t626;
        for (const auto& q : s626) {
            double deg = fmod(fabs(q.th) * 180.0 / PI, 90.0);
            if (deg > 10.0 && deg < 80.0) t626.push_back(q);
        }
        Grid9 g9;
        g9.init(25.520815280171);
        vector<Sq> up9;
        int nup = g9.solve_mis_tabu(t626, up9, -1e-7);
        cout << "Calib Grid9 n=626: tilt=" << t626.size() << " up=" << nup << " tot=" << (t626.size() + nup) << "\n" << flush;
    }

    ofstream jout("poc/swarm_angles/docs/idea10_results.json");
    jout << "[\n";

    for (size_t ti = 0; ti < targets.size(); ++ti) {
        const auto& T = targets[ti];
        int n = T.n;
        int k = (int)floor(sqrt((double)(n - 1)) + 0.5);
        double L_probe = T.L_base - 0.001;
        double L_fprobe = T.L_front + 0.002;
        auto tn = chrono::steady_clock::now();

        // 1) Run 10A a-Homotopy at BOTH L_probe and L_fprobe
        auto [res1_val, res1_sq] = run_10a_homotopy(n, L_probe, seeds[n], 21);
        double best_win_L = T.L_base;
        vector<Sq> best_win_sq;
        string best_src = "none";
        double min_E_10a = (res1_val < 0 ? -res1_val : 0.0);
        if (res1_val > 0 && res1_val < best_win_L - 1e-11) {
            best_win_L = res1_val;
            best_win_sq = res1_sq;
            best_src = "10A_Lprobe";
        }
        auto [res2_val, res2_sq] = run_10a_homotopy(n, L_fprobe, seeds[n], 14);
        if (res2_val < 0) min_E_10a = min(min_E_10a, -res2_val);
        if (res2_val > 0 && res2_val < best_win_L - 1e-11) {
            best_win_L = res2_val;
            best_win_sq = res2_sq;
            best_src = "10A_Lfprobe";
        }

        // 2) Run 10B/10C/10D Grid9 + Tabu MIS + sliding refill
        struct Cand {
            vector<Sq> tilted;
            string tag;
        };
        vector<Cand> cands;

        // 10C: n=971 oblique-cut reflected pockets (adjacent, anti-diag, C2, C4)
        for (double deg : {18.0, 21.0, 24.5, 27.0, 30.0, 33.88, 36.87, 40.0}) {
            double th = deg * PI / 180.0;
            for (double wp : {2.2, 3.2, 4.2, 5.2, 6.5}) {
                for (double u_cut : {4.0, 7.0, 10.0, 14.0}) {
                    if (u_cut >= L_probe - 2.0) continue;
                    for (int md : {0, 1, 3}) {
                        auto tq = build_971_oblique_pockets(L_probe, th, wp, u_cut, md);
                        if (!tq.empty() && (int)tq.size() < n) {
                            cands.push_back({move(tq), "10C_971 deg=" + to_string((int)deg) + " w=" + to_string((int)(wp*10)) + " m=" + to_string(md)});
                        }
                    }
                }
            }
        }

        // 10B: bent & sheared bands (SW+NE / NW+SE)
        for (double deg : {22.5, 26.56, 30.0, 33.88, 36.87, 41.0, 45.0}) {
            double th = deg * PI / 180.0;
            for (int w : {1, 2, 3, 4, 5}) {
                for (double bend : {0.0, 0.05, -0.05}) {
                    for (int a_diff : {0, 1, 2}) {
                        int a1 = k - w - 1, a2 = a1 - a_diff;
                        if (a2 < 4) continue;
                        auto tq = build_bent_band(L_probe, a1, a2, w, th, bend, -0.04, k + 2);
                        if (!tq.empty() && (int)tq.size() < n) {
                            cands.push_back({move(tq), "10B_band deg=" + to_string((int)deg) + " w=" + to_string(w)});
                        }
                    }
                }
            }
        }

        // 10D: larger tilted / chamfered U x V cores (U,V up to 21) at non-45 & 45 deg
        for (double deg : {30.0, 34.0, 37.0, 40.0, 42.5, 44.0, 45.0}) {
            double th = deg * PI / 180.0;
            for (int U = 10; U <= min(21, k - 2); ++U) {
                for (int V : {U, U - 1}) {
                    for (int cut : {0, 1, 2, 3}) {
                        for (double ox : {0.0, 0.25, 0.5}) {
                            auto tq = build_chamfer_core(L_probe, U, V, cut, th, 0.5 * L_probe - ox, 0.5 * L_probe);
                            if (!tq.empty() && (int)tq.size() < n) {
                                cands.push_back({move(tq), "10D_core " + to_string(U) + "x" + to_string(V) + " c=" + to_string(cut) + " d=" + to_string((int)deg)});
                            }
                        }
                    }
                }
            }
        }

        Grid9 g9;
        g9.init(L_probe);
        int best_cap = 0;
        string best_tag = "none";
        vector<Sq> best_mis_sq;

        // Pre-filter by area bound before full Grid9 MIS
        vector<int> active;
        for (int ci = 0; ci < (int)cands.size(); ++ci) {
            double area_rem = L_probe * L_probe - (double)cands[ci].tilted.size();
            if ((int)cands[ci].tilted.size() + (int)area_rem >= n - 1) active.push_back(ci);
        }

        #pragma omp parallel for schedule(dynamic, 4)
        for (int ai = 0; ai < (int)active.size(); ++ai) {
            int ci = active[ai];
            vector<Sq> up;
            int nup = g9.solve_mis_tabu(cands[ci].tilted, up, 0.0);
            int cap = (int)cands[ci].tilted.size() + nup;
            if (cap >= n - 2) {
                auto merged = merge_sliding_fill(L_probe, cands[ci].tilted, up);
                if ((int)merged.size() > cap) cap = (int)merged.size();
                if (cap >= n) {
                    vector<Sq> pol(merged.begin(), merged.begin() + n);
                    double wL = polish_and_shrink(pol, L_probe);
                    #pragma omp critical
                    {
                        if (wL < best_win_L - 1e-11) {
                            best_win_L = wL;
                            best_win_sq = pol;
                            best_src = cands[ci].tag;
                        }
                    }
                }
                #pragma omp critical
                {
                    if (cap > best_cap) {
                        best_cap = cap;
                        best_tag = cands[ci].tag + " nt=" + to_string(cands[ci].tilted.size()) + " up=" + to_string(nup);
                        best_mis_sq = merged;
                    }
                }
            } else {
                #pragma omp critical
                {
                    if (cap > best_cap) {
                        best_cap = cap;
                        best_tag = cands[ci].tag + " nt=" + to_string(cands[ci].tilted.size()) + " up=" + to_string(nup);
                    }
                }
            }
        }

        // If best_mis_sq has >= n-1, try inserting 1 missing square in largest empty void and a-homotopy polish
        if (!best_mis_sq.empty() && best_win_L >= T.L_base - 1e-11) {
            while ((int)best_mis_sq.size() < n) {
                best_mis_sq.push_back({0.5 * L_probe, 0.5 * L_probe, 0.25 * PI});
            }
            best_mis_sq.resize(n);
            auto [hL, hsq] = run_10a_homotopy(n, L_probe, best_mis_sq, 7);
            if (hL > 0 && hL < best_win_L - 1e-11) {
                best_win_L = hL;
                best_win_sq = hsq;
                best_src = "10_MIS+Homotopy(" + best_tag + ")";
            } else if (hL < 0) {
                min_E_10a = min(min_E_10a, -hL);
            }
        }

        if (best_win_L < T.L_base - 1e-11) {
            string path = "poc/swarm_angles/data/cand10_n" + to_string(n) + ".csv";
            save_csv(path, n, best_win_sq);
        }

        double secs = chrono::duration<double>(chrono::steady_clock::now() - tn).count();
        cout << setprecision(6)
             << "n=" << n << " (L_base=" << T.L_base << ", L_front=" << T.L_front << "): "
             << "min_10A_E=" << scientific << setprecision(2) << min_E_10a << fixed << setprecision(6)
             << " | active=" << active.size() << " MIS_cap=" << best_cap << " [" << best_tag << "]"
             << " | best_win_L=" << setprecision(9) << best_win_L << " (" << best_src << ") in "
             << setprecision(2) << secs << "s\n" << flush;

        jout << "  {\"n\": " << n << ", \"L_base\": " << setprecision(12) << T.L_base
             << ", \"L_front\": " << T.L_front
             << ", \"min_10a_E\": " << scientific << min_E_10a << fixed
             << ", \"best_cap\": " << best_cap << ", \"best_tag\": \"" << best_tag
             << "\", \"best_win_L\": " << setprecision(12) << best_win_L
             << ", \"best_src\": \"" << best_src << "\", \"seconds\": " << setprecision(2) << secs << "}"
             << (ti + 1 < targets.size() ? "," : "") << "\n";
    }
    jout << "]\n";
    jout.close();
    double tot = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
    cout << "Idea 10 finished in " << fixed << setprecision(1) << tot << "s\n";
    return 0;
}
