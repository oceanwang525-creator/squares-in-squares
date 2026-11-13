// Pure C++/OpenMP + CUDA (sat_lmin_cuda) end-to-end inflation pipeline.
// Runs unconstrained in Standard Sandbox Mode without venv/NumPy/SciPy:
// 1. Exact-capacity 7-family geometric seeds (ported 1:1 from seeds_v2.py).
// 2. Shared-memory sm_89 CUDA inflation + void/cluster moves + fp64 polish.
// 3. OpenMP float64 Augmented-Lagrangian + L-BFGS (fixed-angle then free-angle)
//    L-squeeze + exact minimal feasible dilation + rounded-CSV self-certification.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <omp.h>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

static constexpr double PI = 3.14159265358979323846;
static constexpr double INV_SQRT2 = 0.70710678118654752440;

struct Sq {
    double x, y, th;
};

struct TargetInfo {
    int n;
    double L_front;
    double inc_L;
};

static inline double sq_extent(double th) {
    return 0.5 * (std::abs(std::cos(th)) + std::abs(std::sin(th)));
}

static inline double exact_bbox_side(const std::vector<Sq>& pts) {
    double xmin = 1e18, xmax = -1e18, ymin = 1e18, ymax = -1e18;
    for (const auto& s : pts) {
        double e = sq_extent(s.th);
        xmin = std::min(xmin, s.x - e);
        xmax = std::max(xmax, s.x + e);
        ymin = std::min(ymin, s.y - e);
        ymax = std::max(ymax, s.y + e);
    }
    return std::max(xmax - xmin, ymax - ymin);
}

static void shift_positive(std::vector<Sq>& pts, double margin = 1e-11) {
    double xmin = 1e18, ymin = 1e18;
    for (const auto& s : pts) {
        double e = sq_extent(s.th);
        xmin = std::min(xmin, s.x - e);
        ymin = std::min(ymin, s.y - e);
    }
    for (auto& s : pts) {
        s.x -= (xmin - margin);
        s.y -= (ymin - margin);
    }
}

static double sat_max_penetration(const std::vector<Sq>& pts) {
    int n = (int)pts.size();
    std::vector<double> c(n), s(n);
    for (int i = 0; i < n; ++i) {
        c[i] = std::cos(pts[i].th);
        s[i] = std::sin(pts[i].th);
    }
    // Cell grid for O(N) exact SAT check
    double L = exact_bbox_side(pts);
    double xmin = 1e18, ymin = 1e18;
    for (int i = 0; i < n; ++i) {
        xmin = std::min(xmin, pts[i].x);
        ymin = std::min(ymin, pts[i].y);
    }
    double cw = 1.5;
    int G = std::max(4, (int)std::ceil((L + 3.0) / cw));
    std::vector<std::vector<int>> grid(G * G);
    for (int i = 0; i < n; ++i) {
        int cx = std::max(0, std::min(G - 1, (int)std::floor((pts[i].x - xmin + 1.0) / cw)));
        int cy = std::max(0, std::min(G - 1, (int)std::floor((pts[i].y - ymin + 1.0) / cw)));
        grid[cy * G + cx].push_back(i);
    }
    double max_pen = 0.0;
    for (int i = 0; i < n; ++i) {
        int cx = std::max(0, std::min(G - 1, (int)std::floor((pts[i].x - xmin + 1.0) / cw)));
        int cy = std::max(0, std::min(G - 1, (int)std::floor((pts[i].y - ymin + 1.0) / cw)));
        double xi = pts[i].x, yi = pts[i].y, ci = c[i], si = s[i];
        for (int dy = -1; dy <= 1; ++dy) {
            int ny = cy + dy;
            if (ny < 0 || ny >= G) continue;
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = cx + dx;
                if (nx < 0 || nx >= G) continue;
                for (int j : grid[ny * G + nx]) {
                    if (j <= i) continue;
                    double ddx = pts[j].x - xi, ddy = pts[j].y - yi;
                    if (ddx * ddx + ddy * ddy >= 2.08) continue;
                    double cj = c[j], sj = s[j];
                    double rad = 0.5 * (1.0 + std::abs(ci * cj + si * sj) + std::abs(si * cj - ci * sj));
                    double p0 = std::abs(ddx * ci + ddy * si);
                    double p1 = std::abs(-ddx * si + ddy * ci);
                    double p2 = std::abs(ddx * cj + ddy * sj);
                    double p3 = std::abs(-ddx * sj + ddy * cj);
                    double pen = rad - std::max(std::max(p0, p1), std::max(p2, p3));
                    if (pen > max_pen) max_pen = pen;
                }
            }
        }
    }
    return max_pen;
}

static double minimal_feasible_dilation(std::vector<Sq>& pts, double tau = 1.5e-10) {
    shift_positive(pts, 1e-11);
    int n = (int)pts.size();
    std::vector<double> c(n), s(n);
    for (int i = 0; i < n; ++i) {
        c[i] = std::cos(pts[i].th);
        s[i] = std::sin(pts[i].th);
    }
    double L = exact_bbox_side(pts);
    double cw = 1.55;
    int G = std::max(4, (int)std::ceil((L + 3.0) / cw));
    std::vector<std::vector<int>> grid(G * G);
    for (int i = 0; i < n; ++i) {
        int cx = std::max(0, std::min(G - 1, (int)std::floor((pts[i].x + 1.0) / cw)));
        int cy = std::max(0, std::min(G - 1, (int)std::floor((pts[i].y + 1.0) / cw)));
        grid[cy * G + cx].push_back(i);
    }
    double lam = 1.0;
    for (int i = 0; i < n; ++i) {
        int cx = std::max(0, std::min(G - 1, (int)std::floor((pts[i].x + 1.0) / cw)));
        int cy = std::max(0, std::min(G - 1, (int)std::floor((pts[i].y + 1.0) / cw)));
        double xi = pts[i].x, yi = pts[i].y, ci = c[i], si = s[i];
        for (int dy = -1; dy <= 1; ++dy) {
            int ny = cy + dy;
            if (ny < 0 || ny >= G) continue;
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = cx + dx;
                if (nx < 0 || nx >= G) continue;
                for (int j : grid[ny * G + nx]) {
                    if (j <= i) continue;
                    double ddx = xi - pts[j].x, ddy = yi - pts[j].y;
                    if (ddx * ddx + ddy * ddy >= 2.08) continue;
                    double cj = c[j], sj = s[j];
                    double rad = 0.5 * (1.0 + std::abs(ci * cj + si * sj) + std::abs(si * cj - ci * sj));
                    double p0 = std::abs(ddx * ci + ddy * si);
                    double p1 = std::abs(-ddx * si + ddy * ci);
                    double p2 = std::abs(ddx * cj + ddy * sj);
                    double p3 = std::abs(-ddx * sj + ddy * cj);
                    double vmax = std::max(std::max(p0, p1), std::max(p2, p3));
                    if (vmax <= 1e-12) return 1e9;
                    double req = (rad + tau) / vmax;
                    if (req > lam) lam = req;
                }
            }
        }
    }
    if (lam > 1.0) {
        for (auto& p : pts) {
            p.x *= lam;
            p.y *= lam;
        }
    }
    shift_positive(pts, 1e-11);
    return lam;
}

// =========================================================================
// Seed Generators (exact-capacity screened + SAT-clipped, matching seeds_v2)
// =========================================================================

static bool solve_shear_angle(int w, double W, double& out_th) {
    if (w <= 0 || W <= 0.0) return false;
    double v_peak = std::sqrt((double)(w * w) + 1.0);
    double v_45 = ((double)w + 1.0) * INV_SQRT2;
    if (W > v_peak || W < v_45) return false;
    double phi = std::atan2(1.0, (double)w);
    double ratio = std::max(-1.0, std::min(1.0, W / v_peak));
    double th = phi + std::acos(ratio);
    if (th < 0.0 || th > PI * 0.25 + 1e-9) return false;
    out_th = std::min(th, PI * 0.25);
    return true;
}

static std::vector<Sq> sat_filter_against(
    const std::vector<Sq>& cands,
    const std::vector<Sq>& existing,
    double scale_a = 0.86,
    double box_L = -1.0
) {
    std::vector<Sq> pts;
    pts.reserve(cands.size());
    for (const auto& s : cands) {
        if (box_L > 0.0) {
            double ext = 0.5 * scale_a * (std::abs(std::cos(s.th)) + std::abs(std::sin(s.th)));
            if (s.x - ext < -1e-4 || s.x + ext > box_L + 1e-4 || s.y - ext < -1e-4 || s.y + ext > box_L + 1e-4)
                continue;
        }
        pts.push_back(s);
    }
    if (pts.empty() || existing.empty()) return pts;

    double half_a = 0.5 * scale_a;
    double cut2 = (1.45 * scale_a) * (1.45 * scale_a);
    int m = (int)existing.size();
    std::vector<double> cj(m), sj(m);
    for (int j = 0; j < m; ++j) {
        cj[j] = std::cos(existing[j].th);
        sj[j] = std::sin(existing[j].th);
    }
    std::vector<Sq> out;
    out.reserve(pts.size());
    for (const auto& s : pts) {
        double ci = std::cos(s.th), si = std::sin(s.th);
        bool hit = false;
        for (int j = 0; j < m; ++j) {
            double dx = existing[j].x - s.x;
            double dy = existing[j].y - s.y;
            if (dx * dx + dy * dy >= cut2) continue;
            double rad = half_a * (1.0 + std::abs(ci * cj[j] + si * sj[j]) + std::abs(si * cj[j] - ci * sj[j]));
            double p0 = std::abs(dx * ci + dy * si);
            double p1 = std::abs(-dx * si + dy * ci);
            double p2 = std::abs(dx * cj[j] + dy * sj[j]);
            double p3 = std::abs(-dx * sj[j] + dy * cj[j]);
            double vmax = std::max(std::max(p0, p1), std::max(p2, p3));
            if (vmax < rad - 1e-6) {
                hit = true;
                break;
            }
        }
        if (!hit) out.push_back(s);
    }
    return out;
}

static std::vector<Sq> make_sheared_strip_exact(
    double x_min, double x_max, double y_min, double y_max,
    int w, double th, int sign, bool vertical, double y_phase, double scale_margin = 0.89
) {
    double W = vertical ? (x_max - x_min) : (y_max - y_min);
    double H_min = vertical ? y_min : x_min;
    double H_max = vertical ? y_max : x_max;
    std::vector<Sq> pts;
    if (W <= 0.5 || H_max - H_min <= 0.5) return pts;

    double c = std::cos(th), s = std::sin(th);
    double hx = 0.5 * (c + s);
    double hx_m = hx * scale_margin;
    double dy_row = 1.0 / std::max(c, 1e-6);
    double row_span = w * c + s;
    double x_pad = 0.5 * std::max(0.0, W - row_span);

    int r_min = -(int)std::ceil(w * s / dy_row) - 2;
    int r_max = (int)std::ceil((H_max - H_min + w * s) / dy_row) + 2;

    for (int r = r_min; r <= r_max; ++r) {
        double y0 = H_min + hx_m + (r + y_phase) * dy_row;
        for (int k = 0; k < w; ++k) {
            double uk = x_pad + hx + k * c;
            double vk = y0 + sign * k * s;
            if (uk - hx_m >= -1e-4 && uk + hx_m <= W + 1e-4 && vk - hx_m >= H_min - 1e-4 && vk + hx_m <= H_max + 1e-4) {
                double ang = sign * th;
                if (vertical)
                    pts.push_back({x_min + uk, vk, ang});
                else
                    pts.push_back({vk, y_min + uk, -ang});
            }
        }
    }
    return pts;
}

static std::vector<Sq> fill_wedges_and_complete(
    std::vector<Sq> pts, int n, double L, std::mt19937_64& rng, double wedge_scale = 0.86
) {
    for (auto& p : pts) {
        double ext = 0.5 * (std::abs(std::cos(p.th)) + std::abs(std::sin(p.th)));
        p.x = std::max(ext * 0.90, std::min(L - ext * 0.90, p.x));
        p.y = std::max(ext * 0.90, std::min(L - ext * 0.90, p.y));
    }
    if ((int)pts.size() > n) {
        while ((int)pts.size() > n) {
            int m = (int)pts.size();
            std::vector<double> c(m), s(m);
            for (int i = 0; i < m; ++i) { c[i] = std::cos(pts[i].th); s[i] = std::sin(pts[i].th); }
            int best_drop = 0;
            double best_score = -1.0;
            double min_pair_d2 = 1e9;
            std::vector<double> sq_min_d2(m, 1e9);
            for (int i = 0; i < m; ++i) {
                double ext = 0.48 * (std::abs(c[i]) + std::abs(s[i]));
                double vL = std::max(0.0, ext - pts[i].x);
                double vR = std::max(0.0, pts[i].x + ext - L);
                double vB = std::max(0.0, ext - pts[i].y);
                double vT = std::max(0.0, pts[i].y + ext - L);
                double sc_i = 5.0 * (vL * vL + vR * vR + vB * vB + vT * vT);
                for (int j = 0; j < m; ++j) {
                    if (i == j) continue;
                    double dx = pts[j].x - pts[i].x, dy = pts[j].y - pts[i].y;
                    double d2 = dx * dx + dy * dy;
                    if (d2 < sq_min_d2[i]) sq_min_d2[i] = d2;
                    if (d2 >= 2.0) continue;
                    double rad = 0.48 * (1.0 + std::abs(c[i] * c[j] + s[i] * s[j]) + std::abs(s[i] * c[j] - c[i] * s[j]));
                    double p0 = std::abs(dx * c[i] + dy * s[i]), p1 = std::abs(-dx * s[i] + dy * c[i]);
                    double p2 = std::abs(dx * c[j] + dy * s[j]), p3 = std::abs(-dx * s[j] + dy * c[j]);
                    double pen = std::max(0.0, rad - std::max(std::max(p0, p1), std::max(p2, p3)));
                    sc_i += pen * pen;
                }
                if (sq_min_d2[i] < min_pair_d2) min_pair_d2 = sq_min_d2[i];
                if (sc_i > best_score) { best_score = sc_i; best_drop = i; }
            }
            if (best_score <= 1e-8) {
                std::vector<int> tight;
                for (int i = 0; i < m; ++i) if (sq_min_d2[i] <= min_pair_d2 + 0.015) tight.push_back(i);
                best_drop = tight[rng() % tight.size()];
            }
            pts.erase(pts.begin() + best_drop);
        }
        return pts;
    }

    int need = n - (int)pts.size();
    if (need == 0) return pts;
    int gx_n = std::max(8, (int)std::round(L * 1.35));
    double step = (L - 0.92) / std::max(1, gx_n - 1);
    std::vector<Sq> probes;
    probes.reserve(gx_n * gx_n);
    for (int ix = 0; ix < gx_n; ++ix) {
        double px = 0.46 + ix * step;
        for (int iy = 0; iy < gx_n; ++iy) {
            probes.push_back({px, 0.46 + iy * step, 0.0});
        }
    }
    int P = (int)probes.size();
    std::vector<double> min_d2(P, 1e9);
    std::vector<double> near_ang(P, 0.0);
    for (const auto& s : pts) {
        for (int k = 0; k < P; ++k) {
            double dx = probes[k].x - s.x, dy = probes[k].y - s.y;
            double d2 = dx * dx + dy * dy;
            if (d2 < min_d2[k]) { min_d2[k] = d2; near_ang[k] = s.th; }
        }
    }
    double half_a = 0.5 * wedge_scale;
    while ((int)pts.size() < n) {
        std::vector<int> order(P);
        std::iota(order.begin(), order.end(), 0);
        int top_m = std::min(P, 28);
        std::partial_sort(order.begin(), order.begin() + top_m, order.end(),
                          [&](int a, int b) { return min_d2[a] > min_d2[b]; });
        double best_sc = 1e18;
        Sq best_cand{0.5 * L, 0.5 * L, 0.0};
        int best_idx = order[0];
        for (int t = 0; t < top_m; ++t) {
            int oi = order[t];
            double px = probes[oi].x, py = probes[oi].y;
            double angs[3] = {0.0, PI * 0.25, near_ang[oi]};
            for (int ak = 0; ak < 3; ++ak) {
                double ang = angs[ak];
                double ca = std::cos(ang), sa = std::sin(ang);
                double ext = half_a * (std::abs(ca) + std::abs(sa));
                double vL = std::max(0.0, ext - px), vR = std::max(0.0, px + ext - L);
                double vB = std::max(0.0, ext - py), vT = std::max(0.0, py + ext - L);
                double tot = 5.0 * (vL * vL + vR * vR + vB * vB + vT * vT);
                for (const auto& s : pts) {
                    double dx = s.x - px, dy = s.y - py;
                    if (dx * dx + dy * dy >= 2.0) continue;
                    double cj = std::cos(s.th), sj = std::sin(s.th);
                    double rad = half_a * (1.0 + std::abs(ca * cj + sa * sj) + std::abs(sa * cj - ca * sj));
                    double p0 = std::abs(dx * ca + dy * sa), p1 = std::abs(-dx * sa + dy * ca);
                    double p2 = std::abs(dx * cj + dy * sj), p3 = std::abs(-dx * sj + dy * cj);
                    double pen = std::max(0.0, rad - std::max(std::max(p0, p1), std::max(p2, p3)));
                    tot += pen * pen;
                }
                if (tot < best_sc) {
                    best_sc = tot;
                    best_cand = {px, py, ang};
                    best_idx = oi;
                }
                if (best_sc == 0.0) break;
            }
            if (best_sc == 0.0) break;
        }
        pts.push_back(best_cand);
        for (int k = 0; k < P; ++k) {
            double dx = probes[k].x - best_cand.x, dy = probes[k].y - best_cand.y;
            double d2 = dx * dx + dy * dy;
            if (d2 < min_d2[k]) { min_d2[k] = d2; near_ang[k] = best_cand.th; }
        }
        min_d2[best_idx] = -1.0;
    }
    return pts;
}

static void add_jitter(std::vector<Sq>& pts, std::mt19937_64& rng, double amp = 0.008) {
    std::normal_distribution<double> nd(0.0, amp);
    for (auto& p : pts) {
        p.x += nd(rng);
        p.y += nd(rng);
        p.th += nd(rng);
    }
}

static std::vector<Sq> seed_pinwheel_4pocket(int n, double L, std::mt19937_64& rng, std::string& desc);
static std::vector<Sq> seed_45_upright_hybrids(int n, double L, std::mt19937_64& rng, std::string& desc);

static std::vector<Sq> seed_tapered_sheared_l(int n, double L, std::mt19937_64& rng, std::string& desc) {
    int k = (int)std::floor(std::sqrt((double)n));
    struct Cand { int Kx, Ky, w1, w2, mode; double th1, th2, score; };
    std::vector<Cand> coarse;
    for (int Kx = std::max(4, k - 13); Kx < std::max(5, k - 1); ++Kx) {
        double W1 = L - Kx;
        if (W1 < 1.6) continue;
        std::vector<std::pair<int, double>> w1_opts;
        for (int w1 : {(int)std::floor(W1) + 1, (int)std::floor(W1) + 2}) {
            double th1;
            if (solve_shear_angle(w1, W1 - 1e-4, th1) && th1 <= 0.72) w1_opts.push_back({w1, th1});
        }
        if (w1_opts.empty()) continue;
        for (int Ky = std::max(4, k - 13); Ky < std::max(5, k - 1); ++Ky) {
            double W2 = L - Ky;
            if (W2 < 1.6) continue;
            for (auto [w1, th1] : w1_opts) {
                for (int w2 : {(int)std::floor(W2) + 1, (int)std::floor(W2) + 2}) {
                    double th2;
                    if (!solve_shear_angle(w2, W2 - 1e-4, th2) || th2 > 0.72) continue;
                    double rho_w = w1 * std::cos(th1) / W1;
                    double rho_s = w2 * std::cos(th2) / W2;
                    for (int mode = 0; mode < 3; ++mode) {
                        double cap = 0.0;
                        if (mode == 0)
                            cap = Kx * Ky + W1 * L * rho_w + Kx * W2 * rho_s - 0.8 * (w1 * std::sin(th1) + w2 * std::sin(th2));
                        else if (mode == 1)
                            cap = Kx * Ky + W2 * L * rho_s + Ky * W1 * rho_w - 0.8 * (w1 * std::sin(th1) + w2 * std::sin(th2));
                        else
                            cap = Kx * Ky + W1 * W2 * 0.95 + Ky * W1 * rho_w + Kx * W2 * rho_s - 0.8 * (w1 * std::sin(th1) + w2 * std::sin(th2));
                        double sc = std::abs(cap - n) - 0.015 * (Kx + Ky);
                        coarse.push_back({Kx, Ky, w1, w2, mode, th1, th2, sc});
                    }
                }
            }
        }
    }
    if (coarse.empty()) return seed_pinwheel_4pocket(n, L, rng, desc);
    std::sort(coarse.begin(), coarse.end(), [](const Cand& a, const Cand& b) { return a.score < b.score; });

    int pool = std::min(48, (int)coarse.size());
    std::vector<int> idx(pool);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), rng);
    int n_eval = std::min(8, pool);
    double pitches[3] = {0.985, 0.994, 1.0};
    double pitch = pitches[rng() % 3];
    std::uniform_real_distribution<double> u09(0.0, 0.9);

    std::vector<Sq> best_pts;
    double best_pen = 1e18;
    for (int t = 0; t < n_eval; ++t) {
        const auto& c = coarse[idx[t]];
        double W1 = L - c.Kx * pitch, W2 = L - c.Ky * pitch;
        std::vector<Sq> pts;
        pts.reserve(n + 20);
        for (int ix = 0; ix < c.Kx; ++ix)
            for (int iy = 0; iy < c.Ky; ++iy)
                pts.push_back({L - c.Kx * pitch + (ix + 0.5) * pitch, L - c.Ky * pitch + (iy + 0.5) * pitch, 0.0});
        int s1 = (rng() % 2) ? 1 : -1;
        int s2 = (rng() % 2) ? 1 : -1;
        std::vector<Sq> west, south, sw;
        const char* mtag = "SWmix";
        if (c.mode == 0) {
            west = make_sheared_strip_exact(0.0, W1, 0.0, L, c.w1, c.th1, s1, true, u09(rng), 0.89);
            south = make_sheared_strip_exact(W1, L, 0.0, W2, c.w2, c.th2, s2, false, u09(rng), 0.89);
            mtag = "Wfull";
        } else if (c.mode == 1) {
            south = make_sheared_strip_exact(0.0, L, 0.0, W2, c.w2, c.th2, s2, false, u09(rng), 0.89);
            west = make_sheared_strip_exact(0.0, W1, W2, L, c.w1, c.th1, s1, true, u09(rng), 0.89);
            mtag = "Sfull";
        } else {
            west = make_sheared_strip_exact(0.0, W1, W2, L, c.w1, c.th1, s1, true, u09(rng), 0.89);
            south = make_sheared_strip_exact(W1, L, 0.0, W2, c.w2, c.th2, s2, false, u09(rng), 0.89);
            if (W1 >= 1.0 && W2 >= 1.0) {
                int p1 = (int)std::floor(W1), p2 = (int)std::floor(W2);
                if (p1 > 0 && p2 > 0) {
                    double dx = std::min(0.04, 0.5 * (W1 - p1) / std::max(1, p1));
                    double dy = std::min(0.04, 0.5 * (W2 - p2) / std::max(1, p2));
                    for (int ix = 0; ix < p1; ++ix)
                        for (int iy = 0; iy < p2; ++iy)
                            sw.push_back({(ix + 0.5) * (1.0 + dx), (iy + 0.5) * (1.0 + dy), 0.0});
                }
                for (int vert = 0; vert < 2; ++vert) {
                    double span_w = vert ? W1 : W2;
                    int ws = (int)std::floor(span_w) + 1;
                    double ths;
                    if (solve_shear_angle(ws, span_w - 1e-4, ths) && ths <= 0.72) {
                        auto cand = make_sheared_strip_exact(0.0, W1, 0.0, W2, ws, ths, (rng() % 2) ? 1 : -1, vert == 1, 0.0, 0.90);
                        if (cand.size() > sw.size()) sw = std::move(cand);
                    }
                }
            }
        }
        for (const auto* part : {&west, &south, &sw}) {
            if (!part->empty()) {
                auto ok = sat_filter_against(*part, pts, 0.86, L);
                pts.insert(pts.end(), ok.begin(), ok.end());
            }
        }
        int diff = (int)pts.size() - n;
        double pen = std::abs(diff) + (diff < -8 ? 4.0 : 0.0);
        if (pen < best_pen) {
            best_pen = pen;
            best_pts = std::move(pts);
            std::ostringstream os;
            os << "tapered_L(K=" << c.Kx << "x" << c.Ky << ",w=" << c.w1 << "/" << c.w2 << "," << mtag << ")";
            desc = os.str();
        }
    }
    best_pts = fill_wedges_and_complete(std::move(best_pts), n, L, rng, 0.86);
    add_jitter(best_pts, rng, 0.008);
    return best_pts;
}

static std::vector<Sq> seed_pinwheel_4pocket(int n, double L, std::mt19937_64& rng, std::string& desc) {
    int k = (int)std::floor(std::sqrt((double)n));
    int handed = (rng() % 2) ? 1 : -1;
    std::uniform_real_distribution<double> u06(-0.06, 0.06), u09(0.0, 0.9);
    std::vector<Sq> best_pts;
    double best_pen = 1e18;

    int lo_k = std::max(4, k - 12), hi_k = std::max(5, k - 2);
    for (int att = 0; att < 18; ++att) {
        int Kx = lo_k + (int)(rng() % (hi_k - lo_k));
        int Ky = lo_k + (int)(rng() % (hi_k - lo_k));
        double rx = L - Kx, ry = L - Ky;
        int Px = (int)std::floor(rx), Py = (int)std::floor(ry);
        if (Px < 2 || Py < 2) continue;
        double fx = rx - Px, fy = ry - Py;
        int pw = (Px == 2) ? 1 : (1 + (int)(rng() % (Px - 2)));
        int ps = (Py == 2) ? 1 : (1 + (int)(rng() % (Py - 2)));
        double W_W = pw + 0.5 * (fx + 1.0) + u06(rng);
        double W_E = rx - W_W;
        double W_S = ps + 0.5 * (fy + 1.0) + u06(rng);
        double W_N = ry - W_S;
        if (std::min({W_W, W_E, W_S, W_N}) < 1.1) continue;
        int ww = (int)std::floor(W_W) + 1, we = (int)std::floor(W_E) + 1;
        int ws = (int)std::floor(W_S) + 1, wn = (int)std::floor(W_N) + 1;
        double th_w, th_e, th_s, th_n;
        if (!solve_shear_angle(ww, W_W - 1e-4, th_w) || !solve_shear_angle(we, W_E - 1e-4, th_e) ||
            !solve_shear_angle(ws, W_S - 1e-4, th_s) || !solve_shear_angle(wn, W_N - 1e-4, th_n)) continue;
        if (std::max({th_w, th_e, th_s, th_n}) > 0.68) continue;

        std::vector<Sq> pts;
        for (int ix = 0; ix < Kx; ++ix)
            for (int iy = 0; iy < Ky; ++iy)
                pts.push_back({W_W + ix + 0.5, W_S + iy + 0.5, 0.0});
        struct Rect { double x0, x1, y0, y1; int w; double th; bool vert; int sgn; };
        Rect rects[4] = {
            {0.0, W_W, 0.0, L - W_N, ww, th_w, true, handed},
            {0.0, L - W_E, L - W_N, L, wn, th_n, false, -handed},
            {L - W_E, L, W_S, L, we, th_e, true, handed},
            {W_W, L, 0.0, W_S, ws, th_s, false, -handed},
        };
        for (const auto& rc : rects) {
            auto st = make_sheared_strip_exact(rc.x0, rc.x1, rc.y0, rc.y1, rc.w, rc.th, rc.sgn, rc.vert, u09(rng), 0.89);
            auto ok = sat_filter_against(st, pts, 0.86, L);
            pts.insert(pts.end(), ok.begin(), ok.end());
        }
        int diff = (int)pts.size() - n;
        double pen = std::abs(diff) + (diff < -8 ? 4.0 : 0.0);
        if (pen < best_pen) {
            best_pen = pen;
            best_pts = std::move(pts);
            std::ostringstream os;
            os << "pinwheel4(K=" << Kx << "x" << Ky << ",w=" << ww << "/" << we << "/" << ws << "/" << wn << ")";
            desc = os.str();
            if (std::abs(diff) <= 3) break;
        }
    }
    if (best_pts.empty()) return seed_45_upright_hybrids(n, L, rng, desc);
    best_pts = fill_wedges_and_complete(std::move(best_pts), n, L, rng, 0.86);
    add_jitter(best_pts, rng, 0.008);
    return best_pts;
}

static std::vector<Sq> seed_45_upright_hybrids(int n, double L, std::mt19937_64& rng, std::string& desc) {
    int var = (int)(rng() % 4);
    std::vector<Sq> best_pts;
    double best_pen = 1e18;
    for (int att = 0; att < 10; ++att) {
        std::vector<Sq> pts;
        std::string cand_desc;
        double ext45 = INV_SQRT2 * 0.90;
        if (var == 0) {
            int max_wb = std::max(4, (int)(L * 0.5));
            int wb = 2 + (int)(rng() % std::max(1, max_wb - 2));
            double u_mid = std::uniform_real_distribution<double>(0.35, 0.65)(rng) * (L * std::sqrt(2.0));
            double u0 = u_mid - 0.5 * (wb - 1.0);
            int m = (int)std::ceil(L * 1.5);
            double vp = std::uniform_real_distribution<double>(-0.4, 0.4)(rng);
            for (int iu = 0; iu < wb; ++iu) {
                double u = u0 + iu;
                for (int iv = -m; iv <= m; ++iv) {
                    double v = iv + vp;
                    double px = (u - v) * INV_SQRT2, py = (u + v) * INV_SQRT2;
                    if (px >= ext45 && px <= L - ext45 && py >= ext45 && py <= L - ext45)
                        pts.push_back({px, py, PI * 0.25});
                }
            }
            double phx = std::uniform_real_distribution<double>(0.0, 0.8)(rng);
            double phy = std::uniform_real_distribution<double>(0.0, 0.8)(rng);
            std::vector<Sq> up;
            for (double gx = 0.47 + phx * 0.15; gx < L - 0.46; gx += 0.985)
                for (double gy = 0.47 + phy * 0.15; gy < L - 0.46; gy += 0.985)
                    up.push_back({gx, gy, 0.0});
            auto ok = sat_filter_against(up, pts, 0.86, L);
            pts.insert(pts.end(), ok.begin(), ok.end());
            cand_desc = "hybrid45_band(wb=" + std::to_string(wb) + ")";
        } else if (var == 1) {
            double sc = std::uniform_real_distribution<double>(0.30, 0.65)(rng);
            int M = (int)std::round(sc * L), mh = M / 2;
            double ox = 0.5 * L + std::uniform_real_distribution<double>(-0.35, 0.35)(rng);
            double oy = 0.5 * L + std::uniform_real_distribution<double>(-0.35, 0.35)(rng);
            for (int iu = -mh; iu < M - mh; ++iu) {
                for (int iv = -mh; iv < M - mh; ++iv) {
                    double gu = iu * 0.99, gv = iv * 0.99;
                    double px = ox + (gu - gv) * INV_SQRT2, py = oy + (gu + gv) * INV_SQRT2;
                    if (px >= ext45 && px <= L - ext45 && py >= ext45 && py <= L - ext45)
                        pts.push_back({px, py, PI * 0.25});
                }
            }
            std::vector<Sq> up;
            for (double gx = 0.48; gx < L - 0.47; gx += 0.99)
                for (double gy = 0.48; gy < L - 0.47; gy += 0.99)
                    up.push_back({gx, gy, 0.0});
            auto ok = sat_filter_against(up, pts, 0.86, L);
            pts.insert(pts.end(), ok.begin(), ok.end());
            cand_desc = "hybrid45_core(M=" + std::to_string(M) + ")";
        } else if (var == 2) {
            double c_leg = std::uniform_real_distribution<double>(0.65, 1.35)(rng) * L;
            int m = (int)std::ceil(L * 1.5);
            double vp = std::uniform_real_distribution<double>(-0.4, 0.4)(rng);
            for (double u = INV_SQRT2; u < L * std::sqrt(2.0) + INV_SQRT2; u += 0.99) {
                for (int iv = -m; iv <= m; ++iv) {
                    double v = iv * 0.99 + vp;
                    double px = (u - v) * INV_SQRT2, py = (u + v) * INV_SQRT2;
                    if (px >= ext45 && px <= L - ext45 && py >= ext45 && py <= L - ext45 && (px + py) <= c_leg)
                        pts.push_back({px, py, PI * 0.25});
                }
            }
            int kx = (int)std::floor(L);
            std::vector<Sq> up;
            for (int ix = 0; ix < kx; ++ix)
                for (int iy = 0; iy < kx; ++iy)
                    up.push_back({L - 0.49 - ix * 0.995, L - 0.49 - iy * 0.995, 0.0});
            auto ok = sat_filter_against(up, pts, 0.86, L);
            pts.insert(pts.end(), ok.begin(), ok.end());
            cand_desc = "hybrid45_split(leg=" + std::to_string((int)std::round(c_leg)) + ")";
        } else {
            double c_leg = std::uniform_real_distribution<double>(0.32, 0.65)(rng) * L;
            int m = (int)std::ceil(L * 1.4);
            for (double u = INV_SQRT2; u < L * std::sqrt(2.0) + INV_SQRT2; u += 0.99) {
                for (int iv = -m; iv <= m; ++iv) {
                    double v = iv * 0.99;
                    double px = (u - v) * INV_SQRT2, py = (u + v) * INV_SQRT2;
                    if (px >= ext45 && px <= L - ext45 && py >= ext45 && py <= L - ext45) {
                        if ((px + py <= c_leg) || ((L - px) + (L - py) <= c_leg))
                            pts.push_back({px, py, PI * 0.25});
                    }
                }
            }
            std::vector<Sq> up;
            for (double gx = 0.49; gx < L - 0.48; gx += 0.995)
                for (double gy = 0.49; gy < L - 0.48; gy += 0.995)
                    up.push_back({gx, gy, 0.0});
            auto ok = sat_filter_against(up, pts, 0.86, L);
            pts.insert(pts.end(), ok.begin(), ok.end());
            cand_desc = "hybrid45_2c(leg=" + std::to_string((int)std::round(c_leg)) + ")";
        }
        int diff = (int)pts.size() - n;
        double pen = std::abs(diff) + (diff < -8 ? 4.0 : 0.0);
        if (pen < best_pen) { best_pen = pen; best_pts = std::move(pts); desc = cand_desc; }
    }
    best_pts = fill_wedges_and_complete(std::move(best_pts), n, L, rng, 0.86);
    add_jitter(best_pts, rng, 0.008);
    return best_pts;
}

static std::vector<Sq> seed_clipped_chiral(int n, double L, std::mt19937_64& rng, std::string& desc) {
    double degs[10] = {16.2602, 18.9246, 22.6199, 28.0725, 36.8699, 14.0, 20.0, 26.0, 32.0, 40.0};
    double phi = (degs[rng() % 10] * PI / 180.0) * ((rng() % 2) ? 1.0 : -1.0);
    double c = std::cos(phi), s = std::sin(phi);
    bool sheared = (rng() % 2) == 1;
    std::vector<Sq> best_pts;
    double best_pen = 1e18;
    for (int ri = 0; ri < 6; ++ri) {
        double rr = 0.20 + 0.044 * ri;
        double ch = rr * L;
        int m = (int)std::ceil(ch * 2.2) + 2;
        std::vector<Sq> pts;
        for (int ix = -m; ix <= m; ++ix) {
            for (int iy = -m; iy <= m; ++iy) {
                double gx = ix * 0.985, gy = iy * 0.985;
                double rx = c * gx - s * gy + 0.5 * L;
                double ry = s * gx + c * gy + 0.5 * L;
                if (std::abs(rx - 0.5 * L) <= ch && std::abs(ry - 0.5 * L) <= ch)
                    pts.push_back({rx, ry, phi});
            }
        }
        double ww = std::max(1.5, 0.5 * L - ch - 0.1);
        for (int rot = 0; rot < 4; ++rot) {
            std::vector<Sq> st;
            if (sheared) {
                int wc = std::max(2, (int)std::floor(ww) + 1);
                double ths = std::abs(phi);
                solve_shear_angle(wc, ww, ths);
                st = make_sheared_strip_exact(0.0, ww, 0.0, L - ww * 0.4, wc, std::min(ths, 0.65), 1, true, 0.0, 0.89);
            } else {
                int nx = std::max(1, (int)std::floor(ww)), ny = std::max(1, (int)std::floor(L - ww * 0.4));
                for (int ix = 0; ix < nx; ++ix)
                    for (int iy = 0; iy < ny; ++iy)
                        st.push_back({ix + 0.5, iy + 0.5, 0.0});
            }
            for (auto& q : st) {
                double ox = q.x, oy = q.y, oa = q.th;
                if (rot == 1) { q.x = L - oy; q.y = ox; q.th = oa + 0.5 * PI; }
                else if (rot == 2) { q.x = L - ox; q.y = L - oy; q.th = oa; }
                else if (rot == 3) { q.x = oy; q.y = L - ox; q.th = oa - 0.5 * PI; }
            }
            auto ok = sat_filter_against(st, pts, 0.86, L);
            pts.insert(pts.end(), ok.begin(), ok.end());
        }
        int diff = (int)pts.size() - n;
        double pen = std::abs(diff) + (diff < -8 ? 4.0 : 0.0);
        if (pen < best_pen) {
            best_pen = pen;
            best_pts = std::move(pts);
            desc = std::string("chiral_clip(") + (sheared ? "sheared)" : "upright)");
        }
    }
    best_pts = fill_wedges_and_complete(std::move(best_pts), n, L, rng, 0.86);
    add_jitter(best_pts, rng, 0.008);
    return best_pts;
}

static std::vector<Sq> seed_2tier_pocket(int n, double L, std::mt19937_64& rng, std::string& desc) {
    int k = (int)std::floor(std::sqrt((double)n));
    int lo_k = std::max(4, k - 12), hi_k = std::max(5, k - 3);
    std::vector<Sq> best_pts;
    double best_pen = 1e18;
    for (int att = 0; att < 10; ++att) {
        int Kx = lo_k + (int)(rng() % (hi_k - lo_k));
        int Ky = lo_k + (int)(rng() % (hi_k - lo_k));
        double W1 = L - Kx, W2 = L - Ky;
        std::vector<Sq> pts;
        for (int ix = 0; ix < Kx; ++ix)
            for (int iy = 0; iy < Ky; ++iy)
                pts.push_back({L - Kx + ix + 0.5, L - Ky + iy + 0.5, 0.0});
        if (W1 >= 3.5) {
            int P1 = (int)std::floor(W1); double f1 = W1 - P1;
            int p1a = std::max(1, P1 / 2);
            double W1a = p1a + 0.5 * (f1 + 1.0), W1b = W1 - W1a;
            int w1a = (int)std::floor(W1a) + 1, w1b = (int)std::floor(W1b) + 1;
            double th1a = 0.34, th1b = 0.34;
            solve_shear_angle(w1a, W1a - 1e-4, th1a);
            solve_shear_angle(w1b, W1b - 1e-4, th1b);
            for (int side = 0; side < 2; ++side) {
                auto st = make_sheared_strip_exact(
                    side == 0 ? 0.0 : W1a, side == 0 ? W1a : W1, 0.0, L,
                    side == 0 ? w1a : w1b, std::min(side == 0 ? th1a : th1b, 0.65),
                    side == 0 ? 1 : -1, true, std::uniform_real_distribution<double>(0.0, 0.8)(rng), 0.89);
                auto ok = sat_filter_against(st, pts, 0.86, L);
                pts.insert(pts.end(), ok.begin(), ok.end());
            }
        }
        if (W2 >= 3.5) {
            int P2 = (int)std::floor(W2); double f2 = W2 - P2;
            int p2a = std::max(1, P2 / 2);
            double W2a = p2a + 0.5 * (f2 + 1.0);
            for (int side = 0; side < 2; ++side) {
                double y0 = side == 0 ? 0.0 : W2a, y1 = side == 0 ? W2a : W2;
                double sp = y1 - y0;
                int ww = std::max(1, (int)std::floor(sp) + 1);
                double th = 0.34; solve_shear_angle(ww, sp - 1e-4, th);
                auto st = make_sheared_strip_exact(W1, L, y0, y1, ww, std::min(th, 0.65), (rng() % 2) ? 1 : -1, false, 0.2, 0.89);
                auto ok = sat_filter_against(st, pts, 0.86, L);
                pts.insert(pts.end(), ok.begin(), ok.end());
            }
        } else if (W2 >= 1.1) {
            int ww = std::max(1, (int)std::floor(W2) + 1);
            double th = 0.38; solve_shear_angle(ww, W2 - 1e-4, th);
            auto st = make_sheared_strip_exact(W1, L, 0.0, W2, ww, std::min(th, 0.68), 1, false, 0.1, 0.89);
            auto ok = sat_filter_against(st, pts, 0.86, L);
            pts.insert(pts.end(), ok.begin(), ok.end());
        }
        int diff = (int)pts.size() - n;
        double pen = std::abs(diff) + (diff < -8 ? 4.0 : 0.0);
        if (pen < best_pen) {
            best_pen = pen; best_pts = std::move(pts);
            desc = "step2tier(K=" + std::to_string(Kx) + "x" + std::to_string(Ky) + ")";
        }
    }
    best_pts = fill_wedges_and_complete(std::move(best_pts), n, L, rng, 0.86);
    add_jitter(best_pts, rng, 0.008);
    return best_pts;
}

static std::vector<Sq> seed_pythagorean_bicryst(int n, double L, std::mt19937_64& rng, std::string& desc) {
    static const double pyth[5] = {
        std::atan2(3.0, 4.0), std::atan2(5.0, 12.0), std::atan2(8.0, 15.0),
        std::atan2(7.0, 24.0), std::atan2(12.0, 35.0)
    };
    static const double bi_deg[5] = {12.0, 16.2602, 18.9246, 22.6199, 28.0725};
    bool use_pyth = (rng() % 100) < 65;
    std::vector<Sq> best_pts;
    double best_pen = 1e18;
    for (int att = 0; att < 8; ++att) {
        std::vector<Sq> pts;
        std::string cand_desc;
        if (use_pyth) {
            double phi = pyth[rng() % 5] * ((rng() % 2) ? 1.0 : -1.0);
            double c = std::cos(phi), s = std::sin(phi);
            double cw = std::uniform_real_distribution<double>(0.32, 0.68)(rng) * L;
            double ch = std::uniform_real_distribution<double>(0.32, 0.68)(rng) * L;
            int m = (int)std::ceil(std::max(cw, ch) * 1.6);
            double ext = 0.45 * (std::abs(c) + std::abs(s));
            for (int ix = -m; ix <= m; ++ix) {
                for (int iy = -m; iy <= m; ++iy) {
                    double gx = ix * 0.99, gy = iy * 0.99;
                    double rx = c * gx - s * gy + 0.5 * cw;
                    double ry = s * gx + c * gy + 0.5 * ch;
                    if (rx >= ext && rx <= cw && ry >= ext && ry <= ch) pts.push_back({rx, ry, phi});
                }
            }
            int k = (int)std::floor(L);
            std::vector<Sq> up;
            for (int ix = 0; ix < k; ++ix)
                for (int iy = 0; iy < k; ++iy)
                    up.push_back({L - 0.49 - ix * 0.995, L - 0.49 - iy * 0.995, 0.0});
            auto ok = sat_filter_against(up, pts, 0.86, L);
            pts.insert(pts.end(), ok.begin(), ok.end());
            cand_desc = "pyth_corner";
        } else {
            double alpha = bi_deg[rng() % 5] * PI / 180.0;
            int border = (int)(rng() % 3);
            if (border > 0) {
                for (double gx = 0.49; gx < L - 0.45; gx += 0.99)
                    for (double gy = 0.49; gy < L - 0.45; gy += 0.99)
                        if (gx < border + 0.1 || gx > L - border - 0.1 || gy < border + 0.1 || gy > L - border - 0.1)
                            pts.push_back({gx, gy, 0.0});
            }
            int m = (int)std::ceil(L * 1.4);
            double lo = border + 0.45, hi = L - border - 0.45;
            for (int side = 0; side < 2; ++side) {
                double ang = (side == 0 ? -1.0 : 1.0) * alpha;
                double ca = std::cos(ang), sa = std::sin(ang);
                double x_lo = (side == 0) ? lo : 0.5 * L;
                double x_hi = (side == 0) ? 0.5 * L : hi;
                std::vector<Sq> cand;
                for (int iu = -m; iu <= m; ++iu) {
                    for (int iv = -m; iv <= m; ++iv) {
                        double u = iu * 0.985, v = iv * 0.985;
                        double rx = ca * u - sa * v + 0.5 * L;
                        double ry = sa * u + ca * v + 0.5 * L;
                        if (rx >= x_lo && rx <= x_hi && ry >= lo && ry <= hi) cand.push_back({rx, ry, ang});
                    }
                }
                auto ok = sat_filter_against(cand, pts, 0.86, L);
                pts.insert(pts.end(), ok.begin(), ok.end());
            }
            cand_desc = "sym_bicryst(b=" + std::to_string(border) + ")";
        }
        int diff = (int)pts.size() - n;
        double pen = std::abs(diff) + (diff < -8 ? 4.0 : 0.0);
        if (pen < best_pen) { best_pen = pen; best_pts = std::move(pts); desc = cand_desc; }
    }
    best_pts = fill_wedges_and_complete(std::move(best_pts), n, L, rng, 0.86);
    add_jitter(best_pts, rng, 0.008);
    return best_pts;
}

// =========================================================================
// Idea 3: Dual-End Smectic-Fanned Strips, Full-Range K, Sandwich & Pinched 4-Frame
// =========================================================================

static inline bool sat_ok_point(double px, double py, double pth, const std::vector<Sq>& pts, double a) {
    double ha = 0.5 * a;
    double ci = std::cos(pth), si = std::sin(pth);
    double cut2 = (1.45 * a) * (1.45 * a);
    for (const auto& q : pts) {
        double dx = q.x - px, dy = q.y - py;
        if (dx * dx + dy * dy >= cut2) continue;
        double cj = std::cos(q.th), sj = std::sin(q.th);
        double rad = ha * (1.0 + std::abs(ci * cj + si * sj) + std::abs(si * cj - ci * sj));
        double p0 = std::abs(dx * ci + dy * si), p1 = std::abs(-dx * si + dy * ci);
        double p2 = std::abs(dx * cj + dy * sj), p3 = std::abs(-dx * sj + dy * cj);
        if (std::max(std::max(p0, p1), std::max(p2, p3)) < rad - 1e-6) return false;
    }
    return true;
}

static std::vector<Sq> build_dual_fanned_strip_box(
    double x0, double x1, double y0, double y1,
    int w, double th, int sign, bool vertical,
    int nf_bot, int nf_top, double a, double y_phase = 0.0
) {
    double W = vertical ? (x1 - x0) : (y1 - y0);
    double H = vertical ? (y1 - y0) : (x1 - x0);
    std::vector<Sq> loc;
    if (W < 0.9 || H < 0.9) return loc;

    auto push_fan_row = [&](int f_i, int nf, bool from_top) {
        double t = ((double)f_i / (double)std::max(1, nf)) * th;
        double c = std::cos(t), s = std::sin(t);
        int wf = std::max(1, (int)std::floor((W - s) / std::max(c, 1e-6)));
        double span = wf * c + s;
        double x_pad = 0.5 * std::max(0.0, W - span);
        double ext = 0.5 * a * (c + s);
        for (int k = 0; k < wf; ++k) {
            double xk = x_pad + 0.5 * (c + s) + k * c;
            if (xk - ext < -1e-4 || xk + ext > W + 1e-4) continue;
            double ang = sign * t;
            if (!from_top) {
                double lo = ext, hi = H - ext;
                if (!sat_ok_point(xk, hi, ang, loc, a)) continue;
                for (int bs = 0; bs < 16; ++bs) {
                    double mid = 0.5 * (lo + hi);
                    if (sat_ok_point(xk, mid, ang, loc, a)) hi = mid;
                    else lo = mid;
                }
                if (hi + ext <= H + 1e-4) loc.push_back({xk, hi, ang});
            } else {
                double lo = ext, hi = H - ext;
                if (!sat_ok_point(xk, lo, ang, loc, a)) continue;
                for (int bs = 0; bs < 16; ++bs) {
                    double mid = 0.5 * (lo + hi);
                    if (sat_ok_point(xk, mid, ang, loc, a)) lo = mid;
                    else hi = mid;
                }
                if (lo - ext >= -1e-4) loc.push_back({xk, lo, ang});
            }
        }
    };

    for (int f_i = 0; f_i < nf_bot; ++f_i) push_fan_row(f_i, nf_bot, false);
    for (int f_i = 0; f_i < nf_top; ++f_i) push_fan_row(f_i, nf_top, true);

    double c = std::cos(th), s = std::sin(th);
    double hx = 0.5 * a * (c + s);
    double dy = 1.0 / std::max(c, 1e-6);
    double x_pad = 0.5 * std::max(0.0, W - (w * c + s));
    int r0 = -(int)std::ceil(w * s / dy) - 2;
    int r1 = (int)std::ceil((H + w * s) / dy) + 2;
    for (int r = r0; r <= r1; ++r) {
        double yb = hx + (r + y_phase) * dy;
        for (int k = 0; k < w; ++k) {
            double uk = x_pad + 0.5 * (c + s) + k * c;
            double vk = yb + sign * k * s;
            if (uk - hx >= -1e-4 && uk + hx <= W + 1e-4 && vk - hx >= -1e-4 && vk + hx <= H + 1e-4) {
                if (sat_ok_point(uk, vk, sign * th, loc, a)) loc.push_back({uk, vk, sign * th});
            }
        }
    }

    std::vector<Sq> out;
    out.reserve(loc.size());
    for (const auto& q : loc) {
        if (vertical) out.push_back({x0 + q.x, y0 + q.y, q.th});
        else out.push_back({x0 + q.y, y0 + q.x, -q.th});
    }
    return out;
}

static std::vector<Sq> seed_smectic_1strip(int n, double L, std::mt19937_64& rng, std::string& desc) {
    int Ky = (int)std::floor(L);
    struct Cand1 { int Kx, w; double th; int nf_b, nf_t; bool vert, right_core; double a_scr; };
    std::vector<Cand1> pool;
    for (int Kx = 2; Kx <= std::max(3, Ky - 2); ++Kx) {
        double W1 = L - Kx;
        if (W1 < 1.5) continue;
        for (int w : {(int)std::floor(W1) + 1, (int)std::floor(W1) + 2}) {
            double th;
            if (!solve_shear_angle(w, W1 - 1e-4, th) || th > 0.65) continue;
            // Quick analytic estimate: keep near n
            double est = Kx * Ky + W1 * L / (1.0 + std::tan(th) / w) - 0.35 * w * std::sin(th);
            if (est >= n - 18) {
                int nf = 2 + (int)(rng() % 4);
                pool.push_back({Kx, w, th, nf, nf, (rng() % 2) == 0, (rng() % 2) == 0, 0.90});
            }
        }
    }
    if (pool.empty()) return seed_tapered_sheared_l(n, L, rng, desc);
    std::shuffle(pool.begin(), pool.end(), rng);
    int n_eval = std::min(10, (int)pool.size());
    std::vector<Sq> best_pts;
    double best_pen = 1e18;
    for (int t = 0; t < n_eval; ++t) {
        const auto& cd = pool[t];
        double W1 = L - cd.Kx;
        int sign = (rng() % 2) ? 1 : -1;
        double y_ph = std::uniform_real_distribution<double>(-0.25, 0.25)(rng);
        double x0s = cd.right_core ? 0.0 : cd.Kx;
        double x1s = cd.right_core ? W1 : L;
        double x0c = cd.right_core ? W1 : 0.0;
        double y_off = ((rng() % 3) == 0) ? (0.5 * (L - Ky)) : ((rng() % 2) ? 0.0 : (L - Ky));

        std::vector<Sq> pts;
        if (cd.vert) {
            pts = build_dual_fanned_strip_box(x0s, x1s, 0.0, L, cd.w, cd.th, sign, true, cd.nf_b, cd.nf_t, 0.895, y_ph);
            for (int ix = 0; ix < cd.Kx; ++ix) {
                double yc_pad = ((rng() % 4 == 0) && (ix % 2 == 1)) ? (L - Ky) : y_off;
                for (int iy = 0; iy < Ky; ++iy) {
                    Sq q{x0c + ix + 0.5, yc_pad + iy + 0.5, 0.0};
                    if (sat_ok_point(q.x, q.y, 0.0, pts, 0.88)) pts.push_back(q);
                }
            }
        } else {
            pts = build_dual_fanned_strip_box(0.0, L, x0s, x1s, cd.w, cd.th, sign, false, cd.nf_b, cd.nf_t, 0.895, y_ph);
            for (int iy = 0; iy < cd.Kx; ++iy) {
                double xc_pad = ((rng() % 4 == 0) && (iy % 2 == 1)) ? (L - Ky) : y_off;
                for (int ix = 0; ix < Ky; ++ix) {
                    Sq q{xc_pad + ix + 0.5, x0c + iy + 0.5, 0.0};
                    if (sat_ok_point(q.x, q.y, 0.0, pts, 0.88)) pts.push_back(q);
                }
            }
        }
        int diff = (int)pts.size() - n;
        double pen = std::abs(diff) + (diff < 0 ? 2.5 * (-diff) : 0.0);
        if (pen < best_pen) {
            best_pen = pen;
            best_pts = std::move(pts);
            std::ostringstream os;
            os << "smectic1(K=" << cd.Kx << "x" << Ky << ",w=" << cd.w << ",nf=" << cd.nf_b << ")";
            desc = os.str();
            if (diff >= 0 && diff <= 2) break;
        }
    }
    best_pts = fill_wedges_and_complete(std::move(best_pts), n, L, rng, 0.88);
    double jit = (rng() % 2) ? 0.002 : 0.006;
    add_jitter(best_pts, rng, jit);
    return best_pts;
}

static std::vector<Sq> seed_smectic_2strip_l(int n, double L, std::mt19937_64& rng, std::string& desc) {
    int Kmax = (int)std::floor(L);
    struct Cand2 { int Kx, Ky, w1, w2, mode; double th1, th2, score; };
    std::vector<Cand2> coarse;
    for (int Kx = std::max(4, Kmax - 18); Kx <= Kmax - 1; ++Kx) {
        double W1 = L - Kx;
        if (W1 < 1.15) continue;
        int w1 = (int)std::floor(W1) + 1;
        double th1;
        if (!solve_shear_angle(w1, W1 - 1e-4, th1) || th1 > 0.72) continue;
        for (int Ky = std::max(4, Kmax - 18); Ky <= Kmax - 1; ++Ky) {
            double W2 = L - Ky;
            if (W2 < 1.15) continue;
            int w2 = (int)std::floor(W2) + 1;
            double th2;
            if (!solve_shear_angle(w2, W2 - 1e-4, th2) || th2 > 0.72) continue;
            double rw = w1 * std::cos(th1) / W1, rs = w2 * std::cos(th2) / W2;
            for (int mode = 0; mode < 2; ++mode) {
                double cap = (mode == 0)
                    ? (Kx * Ky + W1 * L * rw + Kx * W2 * rs)
                    : (Kx * Ky + W2 * L * rs + Ky * W1 * rw);
                if (cap >= n - 12 && cap <= n + 18) coarse.push_back({Kx, Ky, w1, w2, mode, th1, th2, std::abs(cap - n)});
            }
        }
    }
    if (coarse.empty()) return seed_smectic_1strip(n, L, rng, desc);
    std::sort(coarse.begin(), coarse.end(), [](const Cand2& a, const Cand2& b) { return a.score < b.score; });
    int pool_n = std::min(36, (int)coarse.size());
    std::vector<int> perm(pool_n);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);
    std::vector<Sq> best_pts;
    double best_pen = 1e18;
    for (int t = 0; t < std::min(8, pool_n); ++t) {
        const auto& cd = coarse[perm[t]];
        double W1 = L - cd.Kx, W2 = L - cd.Ky;
        int nf = 2 + (int)(rng() % 3);
        int s1 = (rng() % 2) ? 1 : -1, s2 = (rng() % 2) ? 1 : -1;
        std::vector<Sq> west = (cd.mode == 0)
            ? build_dual_fanned_strip_box(0.0, W1, 0.0, L, cd.w1, cd.th1, s1, true, nf, nf, 0.89)
            : build_dual_fanned_strip_box(0.0, W1, W2, L, cd.w1, cd.th1, s1, true, nf, nf, 0.89);
        std::vector<Sq> south = (cd.mode == 0)
            ? build_dual_fanned_strip_box(W1, L, 0.0, W2, cd.w2, cd.th2, s2, false, nf, nf, 0.89)
            : build_dual_fanned_strip_box(0.0, L, 0.0, W2, cd.w2, cd.th2, s2, false, nf, nf, 0.89);
        std::vector<Sq> pts;
        for (int ix = 0; ix < cd.Kx; ++ix)
            for (int iy = 0; iy < cd.Ky; ++iy)
                pts.push_back({W1 + ix + 0.5, W2 + iy + 0.5, 0.0});
        for (const auto* part : {&west, &south}) {
            auto ok = sat_filter_against(*part, pts, 0.87, L);
            pts.insert(pts.end(), ok.begin(), ok.end());
        }
        int diff = (int)pts.size() - n;
        double pen = std::abs(diff) + (diff < 0 ? 2.5 * (-diff) : 0.0);
        if (pen < best_pen) {
            best_pen = pen;
            best_pts = std::move(pts);
            std::ostringstream os;
            os << "smectic2L(K=" << cd.Kx << "x" << cd.Ky << ",w=" << cd.w1 << "/" << cd.w2 << ")";
            desc = os.str();
        }
    }
    best_pts = fill_wedges_and_complete(std::move(best_pts), n, L, rng, 0.87);
    add_jitter(best_pts, rng, (rng() % 2) ? 0.002 : 0.006);
    return best_pts;
}

static std::vector<Sq> seed_sandwich_fanned(int n, double L, std::mt19937_64& rng, std::string& desc) {
    // Central Kx x Ky upright columns flanked by 2 fanned sheared strips (left + right or bottom + top)
    int Ky = (int)std::floor(L);
    std::vector<Sq> best_pts;
    double best_pen = 1e18;
    for (int att = 0; att < 14; ++att) {
        int Kx = 3 + (int)(rng() % std::max(2, Ky - 7));
        double W_rem = L - Kx;
        if (W_rem < 3.4) continue;
        double f_frac = std::uniform_real_distribution<double>(0.25, 0.75)(rng);
        double WL = W_rem * f_frac;
        // Snap WL close to integer - 0.25 for shallow angle!
        int w1 = std::max(2, (int)std::round(WL + 0.2));
        WL = std::max(1.5, std::min(W_rem - 1.5, (double)w1 - std::uniform_real_distribution<double>(0.12, 0.45)(rng)));
        double WR = W_rem - WL;
        int w2 = std::max(2, (int)std::floor(WR) + 1);
        double th1, th2;
        if (!solve_shear_angle(w1, WL - 1e-4, th1) || !solve_shear_angle(w2, WR - 1e-4, th2)) continue;
        if (std::max(th1, th2) > 0.65) continue;

        bool horiz = (rng() % 2) == 1;
        int nf = 3;
        std::vector<Sq> pts;
        for (int ix = 0; ix < Kx; ++ix)
            for (int iy = 0; iy < Ky; ++iy)
                pts.push_back(horiz ? Sq{iy + 0.5, WL + ix + 0.5, 0.0} : Sq{WL + ix + 0.5, iy + 0.5, 0.0});
        auto s1 = horiz
            ? build_dual_fanned_strip_box(0.0, L, 0.0, WL, w1, th1, 1, false, nf, nf, 0.89)
            : build_dual_fanned_strip_box(0.0, WL, 0.0, L, w1, th1, 1, true, nf, nf, 0.89);
        auto s2 = horiz
            ? build_dual_fanned_strip_box(0.0, L, WL + Kx, L, w2, th2, -1, false, nf, nf, 0.89)
            : build_dual_fanned_strip_box(WL + Kx, L, 0.0, L, w2, th2, -1, true, nf, nf, 0.89);
        auto ok1 = sat_filter_against(s1, pts, 0.87, L);
        pts.insert(pts.end(), ok1.begin(), ok1.end());
        auto ok2 = sat_filter_against(s2, pts, 0.87, L);
        pts.insert(pts.end(), ok2.begin(), ok2.end());

        int diff = (int)pts.size() - n;
        double pen = std::abs(diff) + (diff < 0 ? 2.5 * (-diff) : 0.0);
        if (pen < best_pen) {
            best_pen = pen;
            best_pts = std::move(pts);
            desc = "sandwich_fan(K=" + std::to_string(Kx) + ",w=" + std::to_string(w1) + "/" + std::to_string(w2) + ")";
        }
    }
    if (best_pts.empty()) return seed_smectic_1strip(n, L, rng, desc);
    best_pts = fill_wedges_and_complete(std::move(best_pts), n, L, rng, 0.87);
    add_jitter(best_pts, rng, (rng() % 2) ? 0.002 : 0.006);
    return best_pts;
}

// =========================================================================
// Idea 4: Exact a=1.000-Ranked Multi-Grain Basins + Interface Void Filler
// =========================================================================

static std::vector<Sq> add_interface_void_squares(
    std::vector<Sq> pts, int n, double L, double a_fill, double extra_ang, std::mt19937_64& rng
) {
    if ((int)pts.size() >= n) return pts;
    double step = std::max(0.32, L / 64.0);
    std::vector<std::pair<double, Sq>> cands;
    double angs[4] = {0.0, PI * 0.25, extra_ang, -extra_ang};
    for (double px = 0.48; px <= L - 0.48; px += step) {
        for (double py = 0.48; py <= L - 0.48; py += step) {
            for (int ak = 0; ak < 4; ++ak) {
                double th = angs[ak];
                double ext = 0.5 * a_fill * (std::abs(std::cos(th)) + std::abs(std::sin(th)));
                if (px < ext - 1e-4 || px > L - ext + 1e-4 || py < ext - 1e-4 || py > L - ext + 1e-4) continue;
                if (sat_ok_point(px, py, th, pts, a_fill)) {
                    // Prefer near-corner or tight wall spots first
                    double wall_d = std::min(std::min(px, L - px), std::min(py, L - py));
                    cands.push_back({wall_d + 0.05 * ((double)(rng() % 100) / 100.0), {px, py, th}});
                }
            }
        }
    }
    std::sort(cands.begin(), cands.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    for (const auto& c : cands) {
        if ((int)pts.size() >= n) break;
        if (sat_ok_point(c.second.x, c.second.y, c.second.th, pts, a_fill)) pts.push_back(c.second);
    }
    return pts;
}

static std::vector<Sq> build_45band_exact_cfg(
    double L, int wb, double u0, double v_ph, double band_dth, int mode, bool flip_x
) {
    const double is2 = INV_SQRT2;
    double u_max = u0 + (wb - 1);
    std::vector<Sq> pts;
    pts.reserve((size_t)(L * L) + 16);
    int m = (int)std::ceil(L * 1.6);
    double ext45 = is2 * 0.995;

    for (int iu = 0; iu < wb; ++iu) {
        double u = u0 + iu;
        for (int iv = -m; iv <= m; ++iv) {
            double v = (double)iv + v_ph;
            double px = (u - v) * is2, py = (u + v) * is2;
            if (px < ext45 - 1e-6 || px > L - ext45 + 1e-6 || py < ext45 - 1e-6 || py > L - ext45 + 1e-6) continue;
            double th = PI * 0.25 + band_dth;
            if (mode == 1) {
                double d_corner = std::min(std::min(px, L - px), std::min(py, L - py));
                if (d_corner < 3.6) {
                    double frac = (3.6 - d_corner) / 3.6;
                    // Curve band gently toward ~39 deg near wall corners
                    double sgn = ((px < 0.5 * L) == (py < 0.5 * L)) ? -1.0 : -1.0;
                    th = PI * 0.25 + sgn * 0.095 * frac + band_dth;
                }
            }
            pts.push_back({px, py, th});
        }
    }

    int K = (int)std::floor(L) + 1;
    for (int ix = 0; ix < K; ++ix) {
        double gx = 0.5 + ix;
        if (gx > L - 0.5 + 1e-6) break;
        for (int iy = 0; iy < K; ++iy) {
            double gy = 0.5 + iy;
            if (gy > L - 0.5 + 1e-6) break;
            double u = (gx + gy) * is2;
            if (u < u0 - 1.3) pts.push_back({gx, gy, 0.0});
            else if (u < u0 + 0.15 && sat_ok_point(gx, gy, 0.0, pts, 0.995)) pts.push_back({gx, gy, 0.0});
        }
    }
    for (int ix = 0; ix < K; ++ix) {
        double gx = L - 0.5 - ix;
        if (gx < 0.5 - 1e-6) break;
        for (int iy = 0; iy < K; ++iy) {
            double gy = L - 0.5 - iy;
            if (gy < 0.5 - 1e-6) break;
            double u = (gx + gy) * is2;
            if (u > u_max + 1.3) pts.push_back({gx, gy, 0.0});
            else if (u > u_max - 0.15 && sat_ok_point(gx, gy, 0.0, pts, 0.995)) pts.push_back({gx, gy, 0.0});
        }
    }

    if (flip_x) {
        for (auto& s : pts) { s.x = L - s.x; s.th = -s.th; }
    }
    return pts;
}

static std::vector<Sq> seed_45band_exact(int n, double L, std::mt19937_64& rng, std::string& desc) {
    struct Cfg45 { int wb; double u0; int cnt; };
    std::vector<Cfg45> ranked;
    const double s2 = std::sqrt(2.0), is2 = INV_SQRT2;
    for (int wb = 2; wb <= 11; ++wb) {
        double u_lo = std::max(1.5, 0.16 * L * s2 - 0.5 * (wb - 1));
        double u_hi = std::min(L * s2 - 2.0, 0.62 * L * s2 - 0.5 * (wb - 1));
        for (double u0 = u_lo; u0 <= u_hi + 1e-6; u0 += 0.25) {
            auto pts = build_45band_exact_cfg(L, wb, u0, 0.0, 0.0, 0, false);
            ranked.push_back({wb, u0, (int)pts.size()});
        }
    }
    std::sort(ranked.begin(), ranked.end(), [&](const Cfg45& a, const Cfg45& b) {
        if (a.cnt != b.cnt) return a.cnt > b.cnt;
        return std::abs(a.u0 + 0.5 * (a.wb - 1) - 0.5 * L * s2) < std::abs(b.u0 + 0.5 * (b.wb - 1) - 0.5 * L * s2);
    });
    int top_m = std::min(12, (int)ranked.size());
    const auto& pick = ranked[rng() % top_m];
    double v_ph = std::uniform_real_distribution<double>(-0.32, 0.32)(rng);
    double band_dth = ((rng() % 3) == 0) ? std::uniform_real_distribution<double>(-0.07, 0.02)(rng) : 0.0;
    int mode = (rng() % 3 == 0) ? 1 : 0;
    bool flip_x = (rng() % 2 == 1);
    double u0_j = pick.u0 + std::uniform_real_distribution<double>(-0.12, 0.12)(rng);

    auto pts = build_45band_exact_cfg(L, pick.wb, u0_j, v_ph, band_dth, mode, flip_x);
    int raw_sz = (int)pts.size();
    double a_fill = 0.945;
    pts = add_interface_void_squares(std::move(pts), n, L, a_fill, PI * 0.25, rng);
    pts = fill_wedges_and_complete(std::move(pts), n, L, rng, a_fill);
    double jit = std::uniform_real_distribution<double>(0.0006, 0.0025)(rng);
    add_jitter(pts, rng, jit);

    std::ostringstream os;
    os << "band45u(wb=" << pick.wb << ",u0=" << std::fixed << std::setprecision(1) << pick.u0
       << ",m" << mode << ",raw=" << raw_sz << ")";
    desc = os.str();
    return pts;
}

static std::vector<Sq> seed_smectic1_unit(int n, double L, std::mt19937_64& rng, std::string& desc) {
    int Ky = (int)std::floor(L);
    struct CandS { int Kx, w; double th; int raw_cnt; };
    std::vector<CandS> ranked;
    for (int Kx = 1; Kx <= std::max(2, Ky - 2); ++Kx) {
        double W1 = L - Kx;
        if (W1 < 1.3) continue;
        for (int w : {(int)std::floor(W1) + 1, (int)std::floor(W1) + 2}) {
            double th;
            if (!solve_shear_angle(w, W1 - 1e-4, th) || th > 0.65) continue;
            double c = std::cos(th), s = std::sin(th);
            double hx = 0.5 * (c + s), dy = 1.0 / c;
            int strip_cnt = 0;
            int r0 = -(int)std::ceil(w * s / dy) - 1, r1 = (int)std::ceil((L + w * s) / dy) + 1;
            for (int r = r0; r <= r1; ++r) {
                double yb = hx + r * dy;
                for (int k = 0; k < w; ++k) {
                    double vk = yb + k * s;
                    if (vk - hx >= -1e-5 && vk + hx <= L + 1e-5) strip_cnt++;
                }
            }
            int tot = Kx * Ky + strip_cnt;
            ranked.push_back({Kx, w, th, tot});
        }
    }
    if (ranked.empty()) return seed_smectic_1strip(n, L, rng, desc);
    std::sort(ranked.begin(), ranked.end(), [&](const CandS& a, const CandS& b) {
        if (a.raw_cnt != b.raw_cnt) return a.raw_cnt > b.raw_cnt;
        return a.th < b.th;
    });
    int top_m = std::min(6, (int)ranked.size());
    const auto& cd = ranked[rng() % top_m];

    double W1 = L - cd.Kx;
    bool vert = (rng() % 2) == 0, right_core = (rng() % 2) == 0;
    int sign = (rng() % 2) ? 1 : -1;
    double y_ph = std::uniform_real_distribution<double>(-0.20, 0.20)(rng);
    double x0s = right_core ? 0.0 : cd.Kx, x1s = right_core ? W1 : L;
    double x0c = right_core ? W1 : 0.0;
    int nf = (int)(rng() % 3); // 0, 1, or 2 tapered end rows

    std::vector<Sq> pts;
    if (vert) {
        pts = build_dual_fanned_strip_box(x0s, x1s, 0.0, L, cd.w, cd.th, sign, true, nf, nf, 0.96, y_ph);
        for (int ix = 0; ix < cd.Kx; ++ix) {
            double yc_pad = ((rng() % 3 == 0) && (ix % 2 == 1)) ? (L - Ky) : ((rng() % 2) ? 0.0 : 0.5 * (L - Ky));
            for (int iy = 0; iy < Ky; ++iy) {
                Sq q{x0c + ix + 0.5, yc_pad + iy + 0.5, 0.0};
                if (sat_ok_point(q.x, q.y, 0.0, pts, 0.955)) pts.push_back(q);
            }
        }
    } else {
        pts = build_dual_fanned_strip_box(0.0, L, x0s, x1s, cd.w, cd.th, sign, false, nf, nf, 0.96, y_ph);
        for (int iy = 0; iy < cd.Kx; ++iy) {
            double xc_pad = ((rng() % 3 == 0) && (iy % 2 == 1)) ? (L - Ky) : ((rng() % 2) ? 0.0 : 0.5 * (L - Ky));
            for (int ix = 0; ix < Ky; ++ix) {
                Sq q{xc_pad + ix + 0.5, x0c + iy + 0.5, 0.0};
                if (sat_ok_point(q.x, q.y, 0.0, pts, 0.955)) pts.push_back(q);
            }
        }
    }
    int raw_sz = (int)pts.size();
    pts = add_interface_void_squares(std::move(pts), n, L, 0.945, cd.th, rng);
    pts = fill_wedges_and_complete(std::move(pts), n, L, rng, 0.945);
    add_jitter(pts, rng, std::uniform_real_distribution<double>(0.0008, 0.0025)(rng));
    std::ostringstream os;
    os << "smec1u(K=" << cd.Kx << ",w=" << cd.w << ",nf=" << nf << ",raw=" << raw_sz << ")";
    desc = os.str();
    return pts;
}

static std::vector<Sq> seed_tapered_L_unit(int n, double L, std::mt19937_64& rng, std::string& desc) {
    int Kmax = (int)std::floor(L);
    struct CandL { int Kx, Ky, w1, w2, mode; double th1, th2, score; };
    std::vector<CandL> coarse;
    for (int Kx = std::max(5, Kmax - 16); Kx <= Kmax - 2; ++Kx) {
        double W1 = L - Kx;
        if (W1 < 2.2) continue;
        int w1 = (int)std::floor(W1) + 1;
        double th1;
        if (!solve_shear_angle(w1, W1 - 1e-4, th1) || th1 > 0.65) continue;
        for (int Ky = std::max(5, Kmax - 16); Ky <= Kmax - 2; ++Ky) {
            double W2 = L - Ky;
            if (W2 < 2.2) continue;
            int w2 = (int)std::floor(W2) + 1;
            double th2;
            if (!solve_shear_angle(w2, W2 - 1e-4, th2) || th2 > 0.65) continue;
            double rw = w1 * std::cos(th1) / W1, rs = w2 * std::cos(th2) / W2;
            for (int mode = 0; mode < 2; ++mode) {
                double cap = (mode == 0)
                    ? (Kx * Ky + W1 * L * rw + Kx * W2 * rs - 0.5 * W1 * W1 * std::tan(th1) - 0.5 * W2 * W2 * std::tan(th2))
                    : (Kx * Ky + W2 * L * rs + Ky * W1 * rw - 0.5 * W1 * W1 * std::tan(th1) - 0.5 * W2 * W2 * std::tan(th2));
                coarse.push_back({Kx, Ky, w1, w2, mode, th1, th2, cap});
            }
        }
    }
    if (coarse.empty()) return seed_smectic1_unit(n, L, rng, desc);
    std::sort(coarse.begin(), coarse.end(), [](const CandL& a, const CandL& b) { return a.score > b.score; });
    int pool_n = std::min(16, (int)coarse.size());
    const auto& cd = coarse[rng() % pool_n];
    double W1 = L - cd.Kx, W2 = L - cd.Ky;
    int s1 = (rng() % 2) ? 1 : -1, s2 = (rng() % 2) ? 1 : -1;
    int nf = (int)(rng() % 2);
    auto west = (cd.mode == 0)
        ? build_dual_fanned_strip_box(0.0, W1, 0.0, L, cd.w1, cd.th1, s1, true, nf, nf, 0.96)
        : build_dual_fanned_strip_box(0.0, W1, W2, L, cd.w1, cd.th1, s1, true, nf, nf, 0.96);
    auto south = (cd.mode == 0)
        ? build_dual_fanned_strip_box(W1, L, 0.0, W2, cd.w2, cd.th2, s2, false, nf, nf, 0.96)
        : build_dual_fanned_strip_box(0.0, L, 0.0, W2, cd.w2, cd.th2, s2, false, nf, nf, 0.96);
    std::vector<Sq> pts;
    for (int ix = 0; ix < cd.Kx; ++ix)
        for (int iy = 0; iy < cd.Ky; ++iy)
            pts.push_back({W1 + ix + 0.5, W2 + iy + 0.5, 0.0});
    for (const auto* part : {&west, &south}) {
        auto ok = sat_filter_against(*part, pts, 0.955, L);
        pts.insert(pts.end(), ok.begin(), ok.end());
    }
    int raw_sz = (int)pts.size();
    pts = add_interface_void_squares(std::move(pts), n, L, 0.945, 0.5 * (cd.th1 + cd.th2), rng);
    pts = fill_wedges_and_complete(std::move(pts), n, L, rng, 0.945);
    add_jitter(pts, rng, std::uniform_real_distribution<double>(0.0008, 0.0025)(rng));
    std::ostringstream os;
    os << "tapLu(K=" << cd.Kx << "x" << cd.Ky << ",w=" << cd.w1 << "/" << cd.w2 << ",raw=" << raw_sz << ")";
    desc = os.str();
    return pts;
}

// =========================================================================
// Idea 5: S-Arched 45-Deg Bands + Wall-Notch-Plugged Sheared L & Strips
// =========================================================================

static std::vector<Sq> seed_hybrid45_v5(int n, double L, std::mt19937_64& rng, std::string& desc) {
    const double s2 = std::sqrt(2.0), is2 = INV_SQRT2;
    const double sp_opts[5] = {0.980, 0.985, 0.990, 0.994, 0.998};
    const double arch_opts[4] = {0.0, 0.055, 0.100, 0.130}; // rad (~0..7.5 deg corner arch)
    int max_wb = std::min(13, std::max(3, (int)(L * 0.44)));

    std::vector<Sq> best_pts;
    double best_pen = 1e18;
    std::string best_desc;

    for (int att = 0; att < 10; ++att) {
        int wb = 2 + (int)(rng() % std::max(1, max_wb - 1));
        double u_mid = std::uniform_real_distribution<double>(0.32, 0.66)(rng) * (L * s2);
        double u0 = u_mid - 0.5 * (wb - 1);
        double vp = std::uniform_real_distribution<double>(-0.42, 0.42)(rng);
        double arch = arch_opts[rng() % 4];
        double dth_g = ((rng() % 3) == 0) ? std::uniform_real_distribution<double>(-0.04, 0.04)(rng) : 0.0;
        double sp_lo = sp_opts[rng() % 5], sp_hi = sp_opts[rng() % 5];
        bool dual_corner = (rng() % 2 == 0);
        double band_sp = ((rng() % 3) == 0) ? 0.988 : 1.000;

        std::vector<Sq> pts;
        pts.reserve((size_t)(L * L) + 16);
        int m = (int)std::ceil(L * 1.6);
        double ext45 = is2 * 0.90;
        for (int iu = 0; iu < wb; ++iu) {
            double u = u0 + iu * band_sp;
            for (int iv = -m; iv <= m; ++iv) {
                double v = iv * band_sp + vp;
                double px = (u - v) * is2, py = (u + v) * is2;
                if (px >= ext45 && px <= L - ext45 && py >= ext45 && py <= L - ext45) {
                    double th = PI * 0.25 + arch * ((py - px) / L) + dth_g;
                    pts.push_back({px, py, th});
                }
            }
        }
        double u_band_min = u0, u_band_max = u0 + (wb - 1) * band_sp;
        std::vector<Sq> up;
        up.reserve((size_t)(L * L) + 16);
        double phx1 = std::uniform_real_distribution<double>(0.0, 0.14)(rng);
        double phy1 = std::uniform_real_distribution<double>(0.0, 0.14)(rng);
        for (double gx = 0.46 + phx1; gx < L - 0.44; gx += sp_lo) {
            for (double gy = 0.46 + phy1; gy < L - 0.44; gy += sp_lo) {
                if ((gx + gy) * is2 <= 0.5 * (u_band_min + u_band_max)) up.push_back({gx, gy, 0.0});
            }
        }
        if (dual_corner) {
            double phx2 = std::uniform_real_distribution<double>(0.0, 0.14)(rng);
            double phy2 = std::uniform_real_distribution<double>(0.0, 0.14)(rng);
            for (double gx = L - 0.46 - phx2; gx > 0.44; gx -= sp_hi) {
                for (double gy = L - 0.46 - phy2; gy > 0.44; gy -= sp_hi) {
                    if ((gx + gy) * is2 > 0.5 * (u_band_min + u_band_max)) up.push_back({gx, gy, 0.0});
                }
            }
        } else {
            for (double gx = 0.46 + phx1; gx < L - 0.44; gx += sp_hi) {
                for (double gy = 0.46 + phy1; gy < L - 0.44; gy += sp_hi) {
                    if ((gx + gy) * is2 > 0.5 * (u_band_min + u_band_max)) up.push_back({gx, gy, 0.0});
                }
            }
        }
        auto ok = sat_filter_against(up, pts, 0.87, L);
        pts.insert(pts.end(), ok.begin(), ok.end());

        // Plug (0,L) and (L,0) corner tips near band ends before evaluating
        for (auto [cx, cy, cth] : std::array<std::tuple<double,double,double>, 4>{
            std::make_tuple(0.65, L - 0.65, PI * 0.25 + arch),
            std::make_tuple(L - 0.65, 0.65, PI * 0.25 - arch),
            std::make_tuple(0.50, L - 0.50, 0.0),
            std::make_tuple(L - 0.50, 0.50, 0.0)
        }) {
            if ((int)pts.size() < n && sat_ok_point(cx, cy, cth, pts, 0.87)) pts.push_back({cx, cy, cth});
        }

        if ((rng() % 2) == 1) {
            for (auto& s : pts) { s.x = L - s.x; s.th = -s.th; }
        }
        int diff = (int)pts.size() - n;
        double pen = std::abs(diff) + (diff < -10 ? 2.0 * (-diff) : 0.0);
        if (pen < best_pen) {
            best_pen = pen;
            best_pts = std::move(pts);
            std::ostringstream os2;
            os2 << "h45v5(wb=" << wb << ",ar=" << (int)std::round(arch * 180.0 / PI) << ",diff=" << diff << ")";
            best_desc = os2.str();
            if (diff >= -3 && diff <= 3) break;
        }
    }
    best_pts = add_interface_void_squares(std::move(best_pts), n, L, 0.875, PI * 0.25, rng);
    best_pts = fill_wedges_and_complete(std::move(best_pts), n, L, rng, 0.87);
    add_jitter(best_pts, rng, std::uniform_real_distribution<double>(0.002, 0.007)(rng));
    desc = best_desc;
    return best_pts;
}

static std::vector<Sq> seed_smec1_v5(int n, double L, std::mt19937_64& rng, std::string& desc) {
    auto pts = seed_smectic1_unit(n, L, rng, desc);
    // Re-repair wall sawtooth notches at a=0.875 so GPU can pack them cleanly during 0.88->1.00
    if (desc.rfind("smec1u", 0) == 0) desc = "smec1v5" + desc.substr(6);
    add_jitter(pts, rng, 0.004);
    return pts;
}

static std::vector<Sq> seed_tapered_L_v5(int n, double L, std::mt19937_64& rng, std::string& desc) {
    auto pts = seed_tapered_sheared_l(n, L, rng, desc);
    // Plug end-wall 1/sin(th) sawtooth notches at a=0.88 if raw count <= n
    pts = add_interface_void_squares(std::move(pts), n, L, 0.88, 0.35, rng);
    if (desc.rfind("tapered_L", 0) == 0) desc = "tapLv5" + desc.substr(9);
    return pts;
}

// =========================================================================
// Idea 6: Sheared 35-53 Deg Bands, Corner/V-Bent Bands, Phase-Wrapped L
//         & Warm-Start Basin Ruin-and-Recreate
// =========================================================================

static void add_corner_wedge_plugs(
    std::vector<Sq>& pts, int n, double L, double a, std::mt19937_64& rng
) {
    if ((int)pts.size() >= n) return;
    const double angs[9] = {0.0, 0.2618, -0.2618, 0.4636, -0.4636, 0.6435, -0.6435, PI * 0.25, -PI * 0.25};
    const std::array<std::pair<double,double>, 4> corners = {
        std::make_pair(0.0, 0.0), std::make_pair(L, L), std::make_pair(0.0, L), std::make_pair(L, 0.0)
    };
    for (auto [cx, cy] : corners) {
        double sx_dir = (cx < 0.5 * L) ? 1.0 : -1.0;
        double sy_dir = (cy < 0.5 * L) ? 1.0 : -1.0;
        for (int att = 0; att < 140 && (int)pts.size() < n; ++att) {
            double rx = std::uniform_real_distribution<double>(0.42, 3.8)(rng);
            double ry = std::uniform_real_distribution<double>(0.42, 3.8)(rng);
            double px = cx + sx_dir * rx;
            double py = cy + sy_dir * ry;
            double th = angs[rng() % 9];
            double he = 0.5 * a * (std::abs(std::cos(th)) + std::abs(std::sin(th)));
            if (px < he + 1e-4 || px > L - he - 1e-4 || py < he + 1e-4 || py > L - he - 1e-4) continue;
            if (sat_ok_point(px, py, th, pts, a)) pts.push_back({px, py, th});
        }
    }
}

static std::vector<Sq> seed_sheared_band_v6(int n, double L, std::mt19937_64& rng, std::string& desc) {
    const double s2 = std::sqrt(2.0), is2 = INV_SQRT2;
    const double alpha_opts[13] = {
        -9.5, -8.13, -7.0, -6.8, -5.5, -4.0, -2.5, -1.0, 0.0, 1.5, 3.5, 6.8, 8.13
    };
    const double sp_opts[5] = {0.986, 0.992, 0.996, 1.000, 1.002};
    int max_wb = std::min(10, std::max(3, (int)(L * 0.35)));

    std::vector<Sq> best_pts;
    double best_pen = 1e18;
    std::string best_desc;

    for (int att = 0; att < 12; ++att) {
        int wb = 2 + (int)(rng() % std::max(1, max_wb - 1));
        bool off_center = ((rng() % 3) == 0);
        double u_frac = off_center
            ? (((rng() % 2) == 0)
                ? std::uniform_real_distribution<double>(0.13, 0.31)(rng)
                : std::uniform_real_distribution<double>(0.69, 0.87)(rng))
            : std::uniform_real_distribution<double>(0.31, 0.69)(rng);
        double u_mid = u_frac * (L * s2);

        double alpha_deg = alpha_opts[rng() % 13];
        double alpha = alpha_deg * (PI / 180.0);
        double th_b = -PI * 0.25 + alpha; // so deg % 90 covers ~35.5 .. 53.1 deg
        double tilt_b = ((rng() % 3) == 0) ? std::uniform_real_distribution<double>(-0.04, 0.04)(rng) : 0.0;
        double band_ang = PI * 0.25 + tilt_b;
        double dpx = std::cos(band_ang), dpy = std::sin(band_ang);
        double dnx = dpy, dny = -dpx; // unit normal along (x - y)/s2 at tilt_b=0

        double cb = std::cos(th_b), sb = std::sin(th_b);
        double e1x = cb, e1y = sb, e2x = -sb, e2y = cb;
        double p1 = std::abs(e1x * dpx + e1y * dpy);
        double p2 = std::abs(e2x * dpx + e2y * dpy);
        double par_step = ((rng() % 4) == 0 ? 0.990 : 1.000) / std::max(p1, p2);
        double vpx = dpx * par_step, vpy = dpy * par_step;

        double sh_row = ((rng() % 3) == 0)
            ? std::uniform_real_distribution<double>(-0.15, 0.15)(rng) : 0.0;
        double rstep_x = (std::abs(e1x * dnx + e1y * dny) >= std::abs(e2x * dnx + e2y * dny)) ? e1x : e2x;
        double rstep_y = (std::abs(e1x * dnx + e1y * dny) >= std::abs(e2x * dnx + e2y * dny)) ? e1y : e2y;
        double tstep_x = (rstep_x == e1x) ? e2x : e1x;
        double tstep_y = (rstep_x == e1x) ? e2y : e1y;
        double row_dx = rstep_x + sh_row * tstep_x;
        double row_dy = rstep_y + sh_row * tstep_y;
        if (row_dx * dnx + row_dy * dny < 0.0) { row_dx = -row_dx; row_dy = -row_dy; }

        double vp = std::uniform_real_distribution<double>(-0.45, 0.45)(rng);
        double arch = ((rng() % 3) == 0) ? std::uniform_real_distribution<double>(0.0, 0.08)(rng) : 0.0;
        double cx_b = 0.5 * L * (dpx * dpx + 0.0) + (u_mid - 0.5 * L * s2) * dnx * 0.0;
        // Center band midpoint along line x*dpx + y*dpy = 0.5*L*(dpx+dpy), offset along (dnx,dny)
        double u0_off = u_mid - 0.5 * L * s2;
        double mid_x = 0.5 * L + u0_off * dnx;
        double mid_y = 0.5 * L + u0_off * dny;

        std::vector<Sq> pts;
        pts.reserve((size_t)(L * L) + 16);
        int m = (int)std::ceil(L * 1.55);
        double ext_b = 0.5 * 0.90 * (std::abs(cb) + std::abs(sb));
        for (int iu = 0; iu < wb; ++iu) {
            double su = iu - 0.5 * (wb - 1);
            double bx0 = mid_x + su * row_dx;
            double by0 = mid_y + su * row_dy;
            for (int iv = -m; iv <= m; ++iv) {
                double sv = iv + vp;
                double px = bx0 + sv * vpx;
                double py = by0 + sv * vpy;
                if (px >= ext_b && px <= L - ext_b && py >= ext_b && py <= L - ext_b) {
                    double th = th_b + arch * ((px * dpx + py * dpy - 0.5 * L * s2) / L);
                    if (sat_ok_point(px, py, th, pts, 0.875)) pts.push_back({px, py, th});
                }
            }
        }

        // Upright triangles on either side of band plane
        double band_norm_c = 0.0;
        if (!pts.empty()) {
            for (const auto& q : pts) band_norm_c += q.x * dnx + q.y * dny;
            band_norm_c /= (double)pts.size();
        } else {
            band_norm_c = mid_x * dnx + mid_y * dny;
        }
        double sp_lo = sp_opts[rng() % 5], sp_hi = sp_opts[rng() % 5];
        double phx1 = std::uniform_real_distribution<double>(0.0, 0.20)(rng);
        double phy1 = std::uniform_real_distribution<double>(0.0, 0.20)(rng);
        double phx2 = std::uniform_real_distribution<double>(0.0, 0.20)(rng);
        double phy2 = std::uniform_real_distribution<double>(0.0, 0.20)(rng);
        double t_lo = ((rng() % 4) == 0) ? std::uniform_real_distribution<double>(-0.025, 0.025)(rng) : 0.0;
        double t_hi = ((rng() % 4) == 0) ? std::uniform_real_distribution<double>(-0.025, 0.025)(rng) : 0.0;
        bool dual_anchor = (rng() % 2 == 0);

        std::vector<Sq> up;
        up.reserve((size_t)(L * L) + 16);
        for (double gx = 0.48 + phx1; gx < L - 0.45; gx += sp_lo) {
            for (double gy = L - 0.48 - phy1; gy > 0.45; gy -= sp_lo) {
                if (gx * dnx + gy * dny <= band_norm_c) up.push_back({gx, gy, t_lo});
            }
        }
        if (dual_anchor) {
            for (double gx = L - 0.48 - phx2; gx > 0.45; gx -= sp_hi) {
                for (double gy = 0.48 + phy2; gy < L - 0.45; gy += sp_hi) {
                    if (gx * dnx + gy * dny > band_norm_c) up.push_back({gx, gy, t_hi});
                }
            }
        } else {
            for (double gx = 0.48 + phx2; gx < L - 0.45; gx += sp_hi) {
                for (double gy = 0.48 + phy2; gy < L - 0.45; gy += sp_hi) {
                    if (gx * dnx + gy * dny > band_norm_c) up.push_back({gx, gy, t_hi});
                }
            }
        }
        auto ok = sat_filter_against(up, pts, 0.87, L);
        pts.insert(pts.end(), ok.begin(), ok.end());
        add_corner_wedge_plugs(pts, n, L, 0.872, rng);

        if ((rng() % 2) == 1) {
            for (auto& s : pts) { s.x = L - s.x; s.th = -s.th; }
        }
        if ((rng() % 2) == 1) {
            for (auto& s : pts) { std::swap(s.x, s.y); s.th = -s.th; }
        }
        int diff = (int)pts.size() - n;
        double pen = std::abs(diff) + (diff < -10 ? 2.0 * (-diff) : 0.0);
        if (pen < best_pen) {
            best_pen = pen;
            best_pts = std::move(pts);
            std::ostringstream os2;
            os2 << "shb6(wb=" << wb << ",a=" << (int)std::round(45.0 + alpha_deg)
                << ",uf=" << (int)std::round(u_frac * 100) << ",diff=" << diff << ")";
            best_desc = os2.str();
            if (diff >= -2 && diff <= 2) break;
        }
    }
    best_pts = add_interface_void_squares(std::move(best_pts), n, L, 0.875, PI * 0.22, rng);
    best_pts = fill_wedges_and_complete(std::move(best_pts), n, L, rng, 0.87);
    add_jitter(best_pts, rng, std::uniform_real_distribution<double>(0.0015, 0.005)(rng));
    desc = best_desc;
    return best_pts;
}

static std::vector<Sq> seed_bent_band_v6(int n, double L, std::mt19937_64& rng, std::string& desc) {
    bool two_corner = (rng() % 2 == 0);
    std::vector<Sq> pts;
    pts.reserve(n + 32);
    if (two_corner) {
        // Two symmetric corner wedges at (0,0) and (L,L) at angle th_w, contiguous central 0-deg lattice
        int kc = 3 + (int)(rng() % std::max(2, (int)(L * 0.28)));
        double th_w = (PI / 180.0) * (36.87 + std::uniform_real_distribution<double>(0.0, 9.0)(rng));
        double bw = 0.99;
        double cw = std::cos(th_w), sw = std::sin(th_w);
        double vx = cw, vy = sw, ux = -sw, uy = cw;
        for (int i = 0; i < kc; ++i) {
            for (int j = 0; j < kc - i; ++j) {
                double px1 = 0.65 + (i + 0.3) * bw * (0.72) + (j + 0.3) * bw * (0.25); // approximate corner wedge
                double py1 = 0.65 + (j + 0.3) * bw * (0.72) - (i) * 0.05;
                (void)vx; (void)vy; (void)ux; (void)uy; (void)px1; (void)py1;
            }
        }
        // Build clean 45-deg rotated corner triangles at (0,0) and (L,L) via u=(x+y)/s2 threshold
        double cut = (0.8 + kc * 0.72);
        double c45 = std::cos(th_w), s45 = std::sin(th_w);
        double is2 = INV_SQRT2;
        int m = kc + 4;
        for (int i = -m; i <= m; ++i) {
            for (int j = -m; j <= m; ++j) {
                double px = 0.72 + (i * c45 - j * s45) * 0.995;
                double py = 0.72 + (i * s45 + j * c45) * 0.995;
                if (px >= 0.65 && py >= 0.65 && (px + py) * is2 <= cut && sat_ok_point(px, py, th_w, pts, 0.875))
                    pts.push_back({px, py, th_w});
                double qx = L - 0.72 - (i * c45 - j * s45) * 0.995;
                double qy = L - 0.72 - (i * s45 + j * c45) * 0.995;
                if (qx <= L - 0.65 && qy <= L - 0.65 && (L - qx + L - qy) * is2 <= cut && sat_ok_point(qx, qy, th_w, pts, 0.875))
                    pts.push_back({qx, qy, th_w});
            }
        }
        double sp = std::uniform_real_distribution<double>(0.990, 1.000)(rng);
        double ph = std::uniform_real_distribution<double>(0.0, 0.18)(rng);
        std::vector<Sq> up;
        for (double gx = 0.48 + ph; gx < L - 0.45; gx += sp) {
            for (double gy = 0.48 + ph; gy < L - 0.45; gy += sp) {
                up.push_back({gx, gy, 0.0});
            }
        }
        auto ok = sat_filter_against(up, pts, 0.87, L);
        pts.insert(pts.end(), ok.begin(), ok.end());
        desc = "twinc6(kc=" + std::to_string(kc) + ")";
    } else {
        // V-elbow band from (0, yc) to (xm, ym) at +th_v and from (xm, ym) to (xc, L) at -th_v
        int wb = 3 + (int)(rng() % 5);
        double th_v = (PI / 180.0) * (37.0 + std::uniform_real_distribution<double>(0.0, 8.5)(rng));
        double xm = std::uniform_real_distribution<double>(0.35, 0.65)(rng) * L;
        double ym = std::uniform_real_distribution<double>(0.35, 0.65)(rng) * L;
        double ext = 0.5 * 0.90 * (std::cos(th_v) + std::sin(th_v));
        for (int arm = 0; arm < 2; ++arm) {
            double ang = (arm == 0) ? th_v : -th_v;
            double dir_ang = (arm == 0) ? -PI * 0.25 : PI * 0.25;
            double dx = std::cos(dir_ang), dy = std::sin(dir_ang);
            double nx = -dy, ny = dx;
            for (int iu = 0; iu < wb; ++iu) {
                double su = iu - 0.5 * (wb - 1);
                for (int iv = 0; iv <= (int)(L * 1.1); ++iv) {
                    double px = xm + (arm == 0 ? -iv * dx : iv * dx) + su * nx;
                    double py = ym + (arm == 0 ? -iv * dy : iv * dy) + su * ny;
                    if (px >= ext && px <= L - ext && py >= ext && py <= L - ext) {
                        if (sat_ok_point(px, py, ang, pts, 0.875)) pts.push_back({px, py, ang});
                    }
                }
            }
        }
        double sp = std::uniform_real_distribution<double>(0.988, 1.000)(rng);
        double ph = std::uniform_real_distribution<double>(0.0, 0.2)(rng);
        std::vector<Sq> up;
        for (double gx = 0.48 + ph; gx < L - 0.45; gx += sp) {
            for (double gy = 0.48 + ph; gy < L - 0.45; gy += sp) up.push_back({gx, gy, 0.0});
        }
        auto ok = sat_filter_against(up, pts, 0.87, L);
        pts.insert(pts.end(), ok.begin(), ok.end());
        desc = "velbow6(wb=" + std::to_string(wb) + ")";
    }
    add_corner_wedge_plugs(pts, n, L, 0.872, rng);
    if ((rng() % 2) == 1) for (auto& s : pts) { s.x = L - s.x; s.th = -s.th; }
    pts = add_interface_void_squares(std::move(pts), n, L, 0.875, 0.65, rng);
    pts = fill_wedges_and_complete(std::move(pts), n, L, rng, 0.87);
    add_jitter(pts, rng, 0.003);
    return pts;
}

static std::vector<Sq> build_wrapped_strip_block(
    double x0, double y0, double len_x, double wid_y, double th, double a_s, bool transpose_xy
) {
    double c = std::cos(th), s = std::sin(th);
    double hx = 0.5 * a_s * (std::abs(c) + std::abs(s));
    if (wid_y < 2.0 * hx - 1e-9 || len_x < 2.0 * hx - 1e-9) return {};
    int w = (int)std::floor((wid_y - 2.0 * hx + 1e-12) / (a_s * c)) + 1;
    double dx_col = a_s / c;
    std::vector<Sq> out;
    out.reserve((size_t)(w * len_x) + 8);
    for (int r = 0; r < w; ++r) {
        double ly = hx + r * a_s * c;
        double raw_sh = -r * a_s * s;
        int kr = (int)std::ceil(-raw_sh / dx_col - 1e-12);
        double lx = hx + raw_sh + kr * dx_col;
        while (lx <= len_x - hx + 1e-9) {
            if (!transpose_xy) out.push_back({x0 + lx, y0 + ly, th});
            else out.push_back({x0 + ly, y0 + lx, -th});
            lx += dx_col;
        }
    }
    return out;
}

static std::vector<Sq> seed_wrapped_dual_L_v6(int n, double L, std::mt19937_64& rng, std::string& desc) {
    int k = (int)std::floor(L);
    std::vector<Sq> best_pts;
    double best_pen = 1e18;
    std::string best_desc;

    for (int att = 0; att < 10; ++att) {
        int Kx = std::max(4, k - 12 + (int)(rng() % 11));
        int Ky = std::max(4, k - 12 + (int)(rng() % 11));
        double a_s = std::uniform_real_distribution<double>(0.90, 0.995)(rng);
        double Wb = L - Ky, Wl = L - Kx;
        // Find best wrapping angle th1 in [12, 44] deg for Arm 1 (L x Wb) and th2 for Arm 2 (Wl x Ky)
        double best_th1 = 0.46, best_th2 = 0.46;
        size_t cnt1 = 0, cnt2 = 0;
        for (int d = 12; d <= 44; ++d) {
            double th = d * (PI / 180.0);
            auto b1 = build_wrapped_strip_block(0.0, 0.0, L, Wb, th, a_s, false);
            if (b1.size() > cnt1) { cnt1 = b1.size(); best_th1 = th; }
            auto b2 = build_wrapped_strip_block(0.0, Wb, Ky, Wl, th, a_s, true);
            if (b2.size() > cnt2) { cnt2 = b2.size(); best_th2 = th; }
        }
        // Randomly perturb around best angle to unlock sawtooth synergy during 0.88->1.00
        double th1 = std::clamp(best_th1 + std::uniform_real_distribution<double>(-0.08, 0.08)(rng), 0.15, 0.78);
        double th2 = std::clamp(best_th2 + std::uniform_real_distribution<double>(-0.08, 0.08)(rng), 0.15, 0.78);
        std::vector<Sq> pts = build_wrapped_strip_block(0.0, 0.0, L, Wb, th1, a_s, false);
        auto arm2 = build_wrapped_strip_block(0.0, Wb, L - Wb, Wl, th2, a_s, true);
        for (const auto& q : arm2) if (sat_ok_point(q.x, q.y, q.th, pts, 0.875)) pts.push_back(q);

        // Upright core [Wl, L] x [Wb, L]
        double sp = std::uniform_real_distribution<double>(0.988, 1.000)(rng);
        double up_tilt = ((rng() % 3) == 0) ? std::uniform_real_distribution<double>(-0.025, 0.025)(rng) : 0.0;
        for (int ix = 0; ix < Kx; ++ix) {
            for (int iy = 0; iy < Ky; ++iy) {
                double px = L - 0.5 - ix * sp;
                double py = L - 0.5 - iy * sp;
                if (px >= 0.48 && py >= 0.48 && sat_ok_point(px, py, up_tilt, pts, 0.87))
                    pts.push_back({px, py, up_tilt});
            }
        }
        // Pack bottom-left [0, Wl] x [0, Wb] corner pocket if any space left
        double c_ang = ((rng() % 2) == 0) ? 0.0 : PI * 0.25;
        for (double gx = 0.50; gx < Wl + 0.3; gx += 0.96) {
            for (double gy = 0.50; gy < Wb + 0.3; gy += 0.96) {
                if (sat_ok_point(gx, gy, c_ang, pts, 0.875)) pts.push_back({gx, gy, c_ang});
            }
        }
        int diff = (int)pts.size() - n;
        double pen = std::abs(diff);
        if (pen < best_pen) {
            best_pen = pen;
            best_pts = std::move(pts);
            std::ostringstream os;
            os << "wrapL6(K=" << Kx << "x" << Ky << ",a1=" << (int)std::round(th1 * 180 / PI)
               << ",a2=" << (int)std::round(th2 * 180 / PI) << ",diff=" << diff << ")";
            best_desc = os.str();
            if (diff >= -2 && diff <= 2) break;
        }
    }
    if ((rng() % 2) == 1) for (auto& s : best_pts) { s.x = L - s.x; s.th = -s.th; }
    if ((rng() % 2) == 1) for (auto& s : best_pts) { s.y = L - s.y; s.th = -s.th; }
    best_pts = add_interface_void_squares(std::move(best_pts), n, L, 0.875, 0.46, rng);
    best_pts = fill_wedges_and_complete(std::move(best_pts), n, L, rng, 0.87);
    add_jitter(best_pts, rng, 0.0025);
    desc = best_desc;
    return best_pts;
}

static std::vector<std::vector<Sq>> load_warm_basins(int n, const std::string& warm_dir) {
    std::vector<std::vector<Sq>> basins;
    if (warm_dir.empty()) return basins;
    for (const std::string& suf : {"_fm0.bin", "_fm1.bin"}) {
        std::string ep = warm_dir + "/top_n" + std::to_string(n) + suf;
        FILE* f = std::fopen(ep.c_str(), "rb");
        if (!f) continue;
        int32_t hdr[2] = {0, 0};
        if (std::fread(hdr, sizeof(int32_t), 2, f) == 2 && hdr[1] == n && hdr[0] > 0 && hdr[0] <= 64) {
            int M = hdr[0];
            std::vector<double> ax(n), ay(n), ath(n);
            for (int k = 0; k < M; ++k) {
                if (std::fread(ax.data(), sizeof(double), n, f) != (size_t)n) break;
                if (std::fread(ay.data(), sizeof(double), n, f) != (size_t)n) break;
                if (std::fread(ath.data(), sizeof(double), n, f) != (size_t)n) break;
                std::vector<Sq> pts(n);
                for (int i = 0; i < n; ++i) pts[i] = {ax[i], ay[i], ath[i]};
                basins.push_back(std::move(pts));
            }
        }
        std::fclose(f);
    }
    return basins;
}

static std::vector<Sq> seed_ruin_warm_v6(
    int n, double L_target, const std::vector<Sq>& base, std::mt19937_64& rng, std::string& desc
) {
    // Normalize base into [0, L_target] cleanly (preserving scratch topology, never touching merged.csv)
    double xmin = 1e18, xmax = -1e18, ymin = 1e18, ymax = -1e18;
    for (const auto& s : base) {
        double e = 0.5 * (std::abs(std::cos(s.th)) + std::abs(std::sin(s.th)));
        xmin = std::min(xmin, s.x - e); xmax = std::max(xmax, s.x + e);
        ymin = std::min(ymin, s.y - e); ymax = std::max(ymax, s.y + e);
    }
    double L_old = std::max(xmax - xmin, ymax - ymin);
    double sc = L_target / std::max(1e-9, L_old);
    std::vector<Sq> pts = base;
    for (auto& s : pts) {
        s.x = (s.x - xmin) * sc;
        s.y = (s.y - ymin) * sc;
    }

    // Locate top largest empty void on a 24x24 grid
    const int G = 24;
    double best_r2 = -1.0, hx = 0.5 * L_target, hy = 0.5 * L_target;
    for (int gy = 0; gy < G; ++gy) {
        for (int gx = 0; gx < G; ++gx) {
            double px = (gx + 0.5) * (L_target / G);
            double py = (gy + 0.5) * (L_target / G);
            if (px < 0.52 || px > L_target - 0.52 || py < 0.52 || py > L_target - 0.52) continue;
            double md2 = 1e18;
            for (const auto& s : pts) {
                double d2 = (s.x - px) * (s.x - px) + (s.y - py) * (s.y - py);
                if (d2 < md2) md2 = d2;
            }
            if (md2 > best_r2 && (rng() % 5) != 0) { best_r2 = md2; hx = px; hy = py; }
        }
    }

    int mode = (int)(rng() % 4);
    if (mode == 0) {
        // Hole-neighborhood kick + angle shake + teleport 1 neighbor into hole
        for (auto& s : pts) {
            double d2 = (s.x - hx) * (s.x - hx) + (s.y - hy) * (s.y - hy);
            if (d2 < 12.0) {
                s.x += std::uniform_real_distribution<double>(-0.14, 0.14)(rng);
                s.y += std::uniform_real_distribution<double>(-0.14, 0.14)(rng);
                s.th += std::uniform_real_distribution<double>(-0.35, 0.35)(rng);
            }
        }
        int mv = (int)(rng() % n);
        pts[mv].x = hx; pts[mv].y = hy;
        pts[mv].th = ((rng() % 2) == 0) ? 0.0 : PI * 0.25;
        desc = "ruin6_hole";
    } else if (mode == 1) {
        // Coherent band-angle uniform shift (nudges 44 deg <-> 37..42 deg or 63 deg <-> 60..70 deg)
        double dth = std::uniform_real_distribution<double>(-0.12, 0.12)(rng);
        for (auto& s : pts) {
            double deg = std::fmod(std::abs(s.th) * 180.0 / PI, 90.0);
            if (std::min(deg, 90.0 - deg) > 10.0) {
                s.th += dth;
                s.x += std::uniform_real_distribution<double>(-0.03, 0.03)(rng);
                s.y += std::uniform_real_distribution<double>(-0.03, 0.03)(rng);
            }
        }
        desc = "ruin6_btilt";
    } else if (mode == 2) {
        // Interface slide: shift upright grain (x-y < 0 or x+y < L) coherently by +/-0.25 along diagonal
        double dsh = std::uniform_real_distribution<double>(-0.22, 0.22)(rng);
        double dper = std::uniform_real_distribution<double>(-0.10, 0.10)(rng);
        for (auto& s : pts) {
            double deg = std::fmod(std::abs(s.th) * 180.0 / PI, 90.0);
            if (std::min(deg, 90.0 - deg) <= 10.0 && (s.x - s.y < 0.0)) {
                s.x += dsh + dper; s.y += dsh - dper;
            }
        }
        desc = "ruin6_slide";
    } else {
        // Global gentle shake + hole kick
        for (auto& s : pts) {
            double d2 = (s.x - hx) * (s.x - hx) + (s.y - hy) * (s.y - hy);
            double amp = (d2 < 9.0) ? 0.16 : 0.015;
            s.x += std::uniform_real_distribution<double>(-amp, amp)(rng);
            s.y += std::uniform_real_distribution<double>(-amp, amp)(rng);
            if (d2 < 9.0) s.th += std::uniform_real_distribution<double>(-0.25, 0.25)(rng);
        }
        desc = "ruin6_shake";
    }
    if ((rng() % 2) == 1) for (auto& s : pts) { s.x = L_target - s.x; s.th = -s.th; }
    for (auto& s : pts) {
        s.x = std::clamp(s.x, 0.45, L_target - 0.45);
        s.y = std::clamp(s.y, 0.45, L_target - 0.45);
    }
    return pts;
}

// =========================================================================
// Idea 8 Scratch Seed Families: Framed Sheared Band, Asym-Plane Band, O/U-Framed Tilted Core
// =========================================================================

static std::vector<Sq> seed_framed_band_v8(int n, double L, std::mt19937_64& rng, std::string& desc) {
    const double s2 = std::sqrt(2.0);
    const double alpha_opts[11] = {-9.5, -8.13, -6.8, -5.5, -4.0, -2.0, 0.0, 2.0, 4.0, 6.8, 8.13};
    int max_wb = std::min(10, std::max(3, (int)(L * 0.34)));

    std::vector<Sq> best_pts;
    double best_pen = 1e18;
    std::string best_desc;

    for (int att = 0; att < 10; ++att) {
        int wb = 3 + (int)(rng() % std::max(1, max_wb - 2));
        int fw = 1 + (int)(rng() % 2); // 1 or 2 upright wall-frame layers protecting corners (0,L) and (L,0)
        bool taper_ends = ((rng() % 3) != 0);
        double u_frac = std::uniform_real_distribution<double>(0.24, 0.76)(rng);
        double u_mid = u_frac * (L * s2);
        double alpha_deg = alpha_opts[rng() % 11];
        double th_b = -PI * 0.25 + alpha_deg * (PI / 180.0);
        double tilt_b = ((rng() % 3) == 0) ? std::uniform_real_distribution<double>(-0.04, 0.04)(rng) : 0.0;
        double band_ang = PI * 0.25 + tilt_b;
        double dpx = std::cos(band_ang), dpy = std::sin(band_ang);
        double dnx = dpy, dny = -dpx;

        double cb = std::cos(th_b), sb = std::sin(th_b);
        double e1x = cb, e1y = sb, e2x = -sb, e2y = cb;
        double par_step = ((rng() % 4) == 0 ? 0.992 : 1.000) /
                          std::max(std::abs(e1x * dpx + e1y * dpy), std::abs(e2x * dpx + e2y * dpy));
        double vpx = dpx * par_step, vpy = dpy * par_step;

        double sh_row = ((rng() % 3) == 0) ? std::uniform_real_distribution<double>(-0.14, 0.14)(rng) : 0.0;
        bool use_e1 = std::abs(e1x * dnx + e1y * dny) >= std::abs(e2x * dnx + e2y * dny);
        double rstep_x = use_e1 ? e1x : e2x, rstep_y = use_e1 ? e1y : e2y;
        double tstep_x = use_e1 ? e2x : e1x, tstep_y = use_e1 ? e2y : e1y;
        double row_dx = rstep_x + sh_row * tstep_x, row_dy = rstep_y + sh_row * tstep_y;
        if (row_dx * dnx + row_dy * dny < 0.0) { row_dx = -row_dx; row_dy = -row_dy; }

        double u0_off = u_mid - 0.5 * L * s2;
        double mid_x = 0.5 * L + u0_off * dnx;
        double mid_y = 0.5 * L + u0_off * dny;
        double vp = std::uniform_real_distribution<double>(-0.45, 0.45)(rng);

        // Place tapered band clipped inside upright frame [fw*sp, L - fw*sp]
        double sp_f = std::uniform_real_distribution<double>(0.986, 1.000)(rng);
        double pad = fw * sp_f + 0.42;
        double band_half_len = 0.5 * L * s2;
        std::vector<Sq> pts;
        pts.reserve((size_t)(L * L) + 16);
        int m = (int)std::ceil(L * 1.5);
        for (int iu = 0; iu < wb; ++iu) {
            double su = iu - 0.5 * (wb - 1);
            double bx0 = mid_x + su * row_dx, by0 = mid_y + su * row_dy;
            for (int iv = -m; iv <= m; ++iv) {
                double sv = iv + vp;
                double px = bx0 + sv * vpx, py = by0 + sv * vpy;
                if (taper_ends) {
                    double d_along = std::abs((px - mid_x) * dpx + (py - mid_y) * dpy);
                    double local_hw = (d_along > 0.68 * band_half_len)
                        ? std::max(0.6, 0.5 * (wb - 2.0))
                        : (0.5 * wb + 0.05);
                    if (std::abs(su) > local_hw) continue;
                }
                if (px >= pad && px <= L - pad && py >= pad && py <= L - pad) {
                    if (sat_ok_point(px, py, th_b, pts, 0.875)) pts.push_back({px, py, th_b});
                }
            }
        }

        // Flush 0-deg wall frame along all 4 outer sides + dual-anchored interior triangles
        double band_norm_c = (!pts.empty()) ? 0.0 : (mid_x * dnx + mid_y * dny);
        if (!pts.empty()) {
            for (const auto& q : pts) band_norm_c += q.x * dnx + q.y * dny;
            band_norm_c /= (double)pts.size();
        }
        std::vector<Sq> up;
        up.reserve((size_t)(L * L) + 16);
        int n_row = (int)std::floor(L / sp_f);
        for (int ix = 0; ix < n_row; ++ix) {
            for (int iy = 0; iy < n_row; ++iy) {
                bool on_frame = (ix < fw || ix >= n_row - fw || iy < fw || iy >= n_row - fw);
                double gx = 0.5 * sp_f + ix * sp_f;
                double gy = 0.5 * sp_f + iy * sp_f;
                if (on_frame) {
                    up.push_back({gx, gy, 0.0});
                } else if (gx * dnx + gy * dny <= band_norm_c) {
                    up.push_back({gx, gy, 0.0});
                } else {
                    double rx = L - 0.5 * sp_f - (n_row - 1 - ix) * sp_f;
                    double ry = L - 0.5 * sp_f - (n_row - 1 - iy) * sp_f;
                    up.push_back({rx, ry, 0.0});
                }
            }
        }
        auto ok = sat_filter_against(up, pts, 0.87, L);
        pts.insert(pts.end(), ok.begin(), ok.end());
        add_corner_wedge_plugs(pts, n, L, 0.872, rng);
        int diff = (int)pts.size() - n;
        double pen = std::abs(diff) + (diff < -10 ? 2.0 * (-diff) : 0.0);
        if (pen < best_pen) {
            best_pen = pen;
            best_pts = std::move(pts);
            std::ostringstream os;
            os << "fband8(fw=" << fw << ",wb=" << wb << ",a=" << (int)std::round(45.0 + alpha_deg)
               << ",diff=" << diff << ")";
            best_desc = os.str();
            if (diff >= -2 && diff <= 2) break;
        }
    }
    if ((rng() % 2) == 1) for (auto& s : best_pts) { s.x = L - s.x; s.th = -s.th; }
    if ((rng() % 2) == 1) for (auto& s : best_pts) { std::swap(s.x, s.y); s.th = -s.th; }
    best_pts = add_interface_void_squares(std::move(best_pts), n, L, 0.875, PI * 0.22, rng);
    best_pts = fill_wedges_and_complete(std::move(best_pts), n, L, rng, 0.87);
    add_jitter(best_pts, rng, 0.003);
    desc = best_desc;
    return best_pts;
}

static std::vector<Sq> seed_asym_band_v8(int n, double L, std::mt19937_64& rng, std::string& desc) {
    const double s2 = std::sqrt(2.0);
    const double psi_opts[7] = {18.435, 26.565, 31.0, 35.0, 38.0, 41.0, 45.0};
    const double th_opts[8] = {-45.0, -40.5, -36.87, -33.69, -26.565, -21.8, -18.435, -14.036};

    std::vector<Sq> best_pts;
    double best_pen = 1e18;
    std::string best_desc;

    for (int att = 0; att < 10; ++att) {
        double psi_deg = psi_opts[rng() % 7];
        double th_deg = th_opts[rng() % 8];
        double psi = psi_deg * (PI / 180.0);
        double th_b = th_deg * (PI / 180.0);
        int wb = 3 + (int)(rng() % std::max(1, std::min(9, (int)(L * 0.30)) - 2));
        double u_frac = std::uniform_real_distribution<double>(0.20, 0.80)(rng);
        double dpx = std::cos(psi), dpy = std::sin(psi);
        double dnx = dpy, dny = -dpx;

        double cb = std::cos(th_b), sb = std::sin(th_b);
        double e1x = cb, e1y = sb, e2x = -sb, e2y = cb;
        double par_step = ((rng() % 3) == 0 ? 0.992 : 1.000) /
                          std::max(std::abs(e1x * dpx + e1y * dpy), std::abs(e2x * dpx + e2y * dpy));
        double vpx = dpx * par_step, vpy = dpy * par_step;

        bool use_e1 = std::abs(e1x * dnx + e1y * dny) >= std::abs(e2x * dnx + e2y * dny);
        double sh_row = ((rng() % 2) == 0) ? std::uniform_real_distribution<double>(-0.18, 0.18)(rng) : 0.0;
        double rstep_x = (use_e1 ? e1x : e2x) + sh_row * (use_e1 ? e2x : e1x);
        double rstep_y = (use_e1 ? e1y : e2y) + sh_row * (use_e1 ? e2y : e1y);
        if (rstep_x * dnx + rstep_y * dny < 0.0) { rstep_x = -rstep_x; rstep_y = -rstep_y; }

        double u_off = (u_frac - 0.5) * L * 1.15;
        double mid_x = 0.5 * L + u_off * dnx, mid_y = 0.5 * L + u_off * dny;
        double vp = std::uniform_real_distribution<double>(-0.45, 0.45)(rng);
        double ext_b = 0.5 * 0.90 * (std::abs(cb) + std::abs(sb));

        std::vector<Sq> pts;
        pts.reserve((size_t)(L * L) + 16);
        int m = (int)std::ceil(L * 1.6);
        for (int iu = 0; iu < wb; ++iu) {
            double su = iu - 0.5 * (wb - 1);
            double bx0 = mid_x + su * rstep_x, by0 = mid_y + su * rstep_y;
            for (int iv = -m; iv <= m; ++iv) {
                double sv = iv + vp;
                double px = bx0 + sv * vpx, py = by0 + sv * vpy;
                if (px >= ext_b && px <= L - ext_b && py >= ext_b && py <= L - ext_b) {
                    if (sat_ok_point(px, py, th_b, pts, 0.875)) pts.push_back({px, py, th_b});
                }
            }
        }
        double band_norm_c = 0.0;
        if (!pts.empty()) {
            for (const auto& q : pts) band_norm_c += q.x * dnx + q.y * dny;
            band_norm_c /= (double)pts.size();
        } else {
            band_norm_c = mid_x * dnx + mid_y * dny;
        }
        double sp_u = std::uniform_real_distribution<double>(0.988, 1.000)(rng);
        std::vector<Sq> up;
        up.reserve((size_t)(L * L) + 16);
        for (double gx = 0.48; gx < L - 0.45; gx += sp_u) {
            for (double gy = L - 0.48; gy > 0.45; gy -= sp_u) {
                if (gx * dnx + gy * dny <= band_norm_c) up.push_back({gx, gy, 0.0});
            }
        }
        for (double gx = L - 0.48; gx > 0.45; gx -= sp_u) {
            for (double gy = 0.48; gy < L - 0.45; gy += sp_u) {
                if (gx * dnx + gy * dny > band_norm_c) up.push_back({gx, gy, 0.0});
            }
        }
        auto ok = sat_filter_against(up, pts, 0.87, L);
        pts.insert(pts.end(), ok.begin(), ok.end());
        add_corner_wedge_plugs(pts, n, L, 0.872, rng);
        int diff = (int)pts.size() - n;
        double pen = std::abs(diff) + (diff < -10 ? 2.0 * (-diff) : 0.0);
        if (pen < best_pen) {
            best_pen = pen;
            best_pts = std::move(pts);
            std::ostringstream os;
            os << "asym8(p=" << (int)std::round(psi_deg) << ",t=" << (int)std::round(th_deg)
               << ",w=" << wb << ",diff=" << diff << ")";
            best_desc = os.str();
            if (diff >= -2 && diff <= 2) break;
        }
    }
    if ((rng() % 2) == 1) for (auto& s : best_pts) { s.x = L - s.x; s.th = -s.th; }
    if ((rng() % 2) == 1) for (auto& s : best_pts) { std::swap(s.x, s.y); s.th = -s.th; }
    best_pts = add_interface_void_squares(std::move(best_pts), n, L, 0.875, 0.46, rng);
    best_pts = fill_wedges_and_complete(std::move(best_pts), n, L, rng, 0.87);
    add_jitter(best_pts, rng, 0.003);
    desc = best_desc;
    return best_pts;
}

static std::vector<Sq> seed_framed_tilted_v8(int n, double L, std::mt19937_64& rng, std::string& desc) {
    int max_f = std::max(1, (int)(L * 0.35));
    std::vector<Sq> best_pts;
    double best_pen = 1e18;
    std::string best_desc;

    for (int att = 0; att < 10; ++att) {
        int fL = (rng() % 3 == 0) ? 0 : (1 + (int)(rng() % std::min(4, max_f)));
        int fR = (rng() % 3 == 0) ? 0 : (1 + (int)(rng() % std::min(4, max_f)));
        int fB = (rng() % 4 == 0) ? 0 : (1 + (int)(rng() % std::min(6, max_f)));
        int fT = (rng() % 4 == 0) ? 0 : (1 + (int)(rng() % std::min(6, max_f)));
        double sp = std::uniform_real_distribution<double>(0.985, 1.000)(rng);
        double ix0 = fL * sp, ix1 = L - fR * sp;
        double iy0 = fB * sp, iy1 = L - fT * sp;
        if (ix1 - ix0 < 3.0 || iy1 - iy0 < 3.0) { fL = 1; fR = 0; fB = 3; fT = 2; ix0 = sp; ix1 = L; iy0 = 3*sp; iy1 = L - 2*sp; }

        double a_s = std::uniform_real_distribution<double>(0.92, 0.995)(rng);
        double best_th = 0.30;
        size_t best_c = 0;
        for (int d = 11; d <= 34; d += 2) {
            double th = d * (PI / 180.0);
            auto blk = build_wrapped_strip_block(ix0, iy0, ix1 - ix0, iy1 - iy0, th, a_s, false);
            if (blk.size() > best_c) { best_c = blk.size(); best_th = th; }
        }
        double th_use = std::clamp(best_th + std::uniform_real_distribution<double>(-0.05, 0.05)(rng), 0.14, 0.68);
        bool tr = ((rng() % 2) == 1);
        std::vector<Sq> pts = build_wrapped_strip_block(ix0, iy0, ix1 - ix0, iy1 - iy0, th_use, a_s, tr);

        std::vector<Sq> up;
        int nx_full = (int)std::floor(L / sp);
        int ny_full = (int)std::floor(L / sp);
        for (int ix = 0; ix < nx_full; ++ix) {
            for (int iy = 0; iy < ny_full; ++iy) {
                if (ix < fL || ix >= nx_full - fR || iy < fB || iy >= ny_full - fT) {
                    up.push_back({0.5 * sp + ix * sp, 0.5 * sp + iy * sp, 0.0});
                }
            }
        }
        auto ok = sat_filter_against(up, pts, 0.87, L);
        pts.insert(pts.end(), ok.begin(), ok.end());
        add_corner_wedge_plugs(pts, n, L, 0.872, rng);
        int diff = (int)pts.size() - n;
        double pen = std::abs(diff) + (diff < -10 ? 2.0 * (-diff) : 0.0);
        if (pen < best_pen) {
            best_pen = pen;
            best_pts = std::move(pts);
            std::ostringstream os;
            os << "ftilt8(F=" << fL << fR << fB << fT << ",a=" << (int)std::round(th_use * 180 / PI)
               << ",diff=" << diff << ")";
            best_desc = os.str();
            if (diff >= -2 && diff <= 2) break;
        }
    }
    if ((rng() % 2) == 1) for (auto& s : best_pts) { s.x = L - s.x; s.th = -s.th; }
    if ((rng() % 2) == 1) for (auto& s : best_pts) { std::swap(s.x, s.y); s.th = -s.th; }
    best_pts = add_interface_void_squares(std::move(best_pts), n, L, 0.875, 0.35, rng);
    best_pts = fill_wedges_and_complete(std::move(best_pts), n, L, rng, 0.87);
    add_jitter(best_pts, rng, 0.003);
    desc = best_desc;
    return best_pts;
}

static std::vector<Sq> seed_asym_band_v9(int n, double L, std::mt19937_64& rng, std::string& desc, int slot_hint) {
    const double psi_opts[4] = {45.0, 36.87, 33.69, 26.565};
    const double th_opts[12] = {
        -14.036, -16.260, -18.435, -21.801, -22.620, -26.565,
        -28.072, -30.964, -33.690, -36.870, -40.500, -45.000
    };
    int w_opts[6] = {2, 3, 4, 5, 6, 7};

    // Deterministic outer tuple selection from slot_hint so every (psi, th, wb) gets covered
    int idx0 = (slot_hint >= 0) ? (slot_hint % (4 * 12 * 6)) : (int)(rng() % (4 * 12 * 6));
    int ip0 = idx0 % 4;
    int it0 = (idx0 / 4) % 12;
    int iw0 = (idx0 / 48) % 6;

    std::vector<Sq> best_pts;
    double best_pen = 1e18;
    std::string best_desc;

    for (int att = 0; att < 14; ++att) {
        double psi_deg = (att < 10) ? psi_opts[ip0] : psi_opts[rng() % 4];
        double th_deg = (att < 10) ? th_opts[it0] : th_opts[rng() % 12];
        int wb = (att < 6) ? w_opts[iw0] : (2 + (int)(rng() % 6));
        double psi = psi_deg * (PI / 180.0);
        double th_b = th_deg * (PI / 180.0);
        double a_seed = (att % 3 == 0) ? 0.878 : ((att % 3 == 1) ? 0.892 : 0.908);
        double u_frac = std::uniform_real_distribution<double>(0.24, 0.76)(rng);

        double dpx = std::cos(psi), dpy = std::sin(psi);
        double dnx = dpy, dny = -dpx;
        double cb = std::cos(th_b), sb = std::sin(th_b);
        double e1x = cb, e1y = sb, e2x = -sb, e2y = cb;

        double par_step = ((att % 2 == 0) ? 0.996 : 1.004) * a_seed /
                          std::max(std::abs(e1x * dpx + e1y * dpy), std::abs(e2x * dpx + e2y * dpy));
        // Also try unscaled ~1.0/proj half the time since v8's 969 winner used ~1.054 pitch
        if ((att % 4) < 2) {
            par_step = ((att % 2 == 0) ? 0.992 : 1.000) /
                       std::max(std::abs(e1x * dpx + e1y * dpy), std::abs(e2x * dpx + e2y * dpy));
        }
        double vpx = dpx * par_step, vpy = dpy * par_step;

        bool use_e1 = std::abs(e1x * dnx + e1y * dny) >= std::abs(e2x * dnx + e2y * dny);
        double sh_row = ((att % 3) == 0) ? 0.0 : std::uniform_real_distribution<double>(-0.16, 0.16)(rng);
        double row_scale = (att % 4 < 2) ? 1.0 : a_seed;
        double rstep_x = row_scale * ((use_e1 ? e1x : e2x) + sh_row * (use_e1 ? e2x : e1x));
        double rstep_y = row_scale * ((use_e1 ? e1y : e2y) + sh_row * (use_e1 ? e2y : e1y));
        if (rstep_x * dnx + rstep_y * dny < 0.0) { rstep_x = -rstep_x; rstep_y = -rstep_y; }

        double u_off = (u_frac - 0.5) * L * 1.12;
        double mid_x = 0.5 * L + u_off * dnx, mid_y = 0.5 * L + u_off * dny;
        double vp = std::uniform_real_distribution<double>(-0.48, 0.48)(rng);
        double ext_b = 0.5 * a_seed * (std::abs(cb) + std::abs(sb));

        std::vector<Sq> pts;
        pts.reserve((size_t)(L * L) + 16);
        int m = (int)std::ceil(L * 1.6);
        for (int iu = 0; iu < wb; ++iu) {
            double su = iu - 0.5 * (wb - 1);
            double bx0 = mid_x + su * rstep_x, by0 = mid_y + su * rstep_y;
            for (int iv = -m; iv <= m; ++iv) {
                double sv = iv + vp;
                double px = bx0 + sv * vpx, py = by0 + sv * vpy;
                if (px >= ext_b && px <= L - ext_b && py >= ext_b && py <= L - ext_b) {
                    if (sat_ok_point(px, py, th_b, pts, a_seed)) pts.push_back({px, py, th_b});
                }
            }
        }
        double band_norm_c = mid_x * dnx + mid_y * dny;
        if (!pts.empty()) {
            band_norm_c = 0.0;
            for (const auto& q : pts) band_norm_c += q.x * dnx + q.y * dny;
            band_norm_c /= (double)pts.size();
        }
        double sp_u = std::uniform_real_distribution<double>(0.986, 1.000)(rng);
        std::vector<Sq> up;
        up.reserve((size_t)(L * L) + 16);
        int k_max = (int)std::floor(L / sp_u) + 1;
        // t_lo strictly wall-anchored at bottom-right (L, 0) where dot(dn) > band_norm_c
        for (int ix = 0; ix < k_max; ++ix) {
            double gx = L - (0.50 + ix) * sp_u;
            if (gx < 0.45 * sp_u) continue;
            for (int iy = 0; iy < k_max; ++iy) {
                double gy = (0.50 + iy) * sp_u;
                if (gy > L - 0.45 * sp_u) continue;
                if (gx * dnx + gy * dny > band_norm_c) up.push_back({gx, gy, 0.0});
            }
        }
        // t_hi strictly wall-anchored at top-left (0, L) where dot(dn) <= band_norm_c
        for (int ix = 0; ix < k_max; ++ix) {
            double gx = (0.50 + ix) * sp_u;
            if (gx > L - 0.45 * sp_u) continue;
            for (int iy = 0; iy < k_max; ++iy) {
                double gy = L - (0.50 + iy) * sp_u;
                if (gy < 0.45 * sp_u) continue;
                if (gx * dnx + gy * dny <= band_norm_c) up.push_back({gx, gy, 0.0});
            }
        }
        auto ok = sat_filter_against(up, pts, 0.87, L);
        pts.insert(pts.end(), ok.begin(), ok.end());
        add_corner_wedge_plugs(pts, n, L, 0.872, rng);
        int diff = (int)pts.size() - n;
        double pen = std::abs(diff) + (diff < -10 ? 2.0 * (-diff) : 0.0);
        if (pen < best_pen) {
            best_pen = pen;
            best_pts = std::move(pts);
            std::ostringstream os;
            os << "asym9(p=" << (int)std::round(psi_deg) << ",t=" << (int)std::round(th_deg)
               << ",w=" << wb << ",diff=" << diff << ")";
            best_desc = os.str();
            if (diff >= -1 && diff <= 1) break;
        }
    }
    if ((rng() % 2) == 1) for (auto& s : best_pts) { s.x = L - s.x; s.th = -s.th; }
    if ((rng() % 2) == 1) for (auto& s : best_pts) { std::swap(s.x, s.y); s.th = -s.th; }
    best_pts = add_interface_void_squares(std::move(best_pts), n, L, 0.875, 0.46, rng);
    best_pts = fill_wedges_and_complete(std::move(best_pts), n, L, rng, 0.87);
    add_jitter(best_pts, rng, 0.0025);
    desc = best_desc;
    return best_pts;
}

static std::vector<Sq> seed_pyth_diamond_v9(int n, double L, std::mt19937_64& rng, std::string& desc, int slot_hint) {
    // Stepped 0-deg crystals partitioned by k_cl Pythagorean tilted diamond pockets along diagonal
    const double pyth_deg[6] = {-36.870, -22.620, -28.072, -16.260, -26.565, -45.000};
    const char* pyth_tag[6] = {"345", "512", "815", "724", "125", "45d"};
    int ia = (slot_hint >= 0) ? (slot_hint % 6) : (int)(rng() % 6);
    int flip_s = ((slot_hint / 6) % 2 == 0) ? 1 : -1;

    std::vector<Sq> best_pts;
    double best_pen = 1e18;
    double best_th_b = flip_s * pyth_deg[ia] * (PI / 180.0);
    std::string best_desc;

    for (int att = 0; att < 10; ++att) {
        int pick = (att < 5) ? ia : (int)(rng() % 6);
        double th_b = flip_s * pyth_deg[pick] * (PI / 180.0);
        double cb = std::cos(th_b), sb = std::sin(th_b);
        double a_seed = 0.88 + 0.02 * (att % 3);
        int k_cl = 2 + (int)(rng() % 4); // 2..5 clusters
        int rw = 2 + (int)(rng() % 3);   // 2..4 rows per diamond
        int rl = 2 + (int)(rng() % 5);   // 2..6 cols per diamond
        double shift_d = std::uniform_real_distribution<double>(-1.8, 1.8)(rng);
        double sp_u = std::uniform_real_distribution<double>(0.986, 1.000)(rng);
        double ext_b = 0.5 * a_seed * (std::abs(cb) + std::abs(sb));

        std::vector<Sq> pts;
        pts.reserve((size_t)(L * L) + 16);
        double span = L - 2.4;
        for (int c_i = 0; c_i < k_cl; ++c_i) {
            double frac = (k_cl == 1) ? 0.5 : (0.12 + 0.76 * (double)c_i / (double)(k_cl - 1));
            double cx = 1.2 + frac * span + 0.5 * shift_d;
            double cy = 1.2 + frac * span - 0.5 * shift_d;
            if ((slot_hint / 12) % 2 == 1) cy = L - (1.2 + frac * span) - 0.5 * shift_d;
            for (int u = 0; u < rl; ++u) {
                double du = (u - 0.5 * (rl - 1)) * a_seed;
                for (int v = 0; v < rw; ++v) {
                    double dv = (v - 0.5 * (rw - 1)) * a_seed;
                    double px = cx + du * cb - dv * sb;
                    double py = cy + du * sb + dv * cb;
                    if (px >= ext_b && px <= L - ext_b && py >= ext_b && py <= L - ext_b) {
                        if (sat_ok_point(px, py, th_b, pts, a_seed)) pts.push_back({px, py, th_b});
                    }
                }
            }
        }
        int k_max = (int)std::floor(L / sp_u) + 1;
        std::vector<Sq> up;
        up.reserve((size_t)(L * L) + 16);
        bool anti_d = ((slot_hint / 12) % 2 == 1);
        for (int ix = 0; ix < k_max; ++ix) {
            double gx = anti_d ? (0.50 + ix) * sp_u : L - (0.50 + ix) * sp_u;
            if (gx < 0.45 * sp_u || gx > L - 0.45 * sp_u) continue;
            for (int iy = 0; iy < k_max; ++iy) {
                double gy = (0.50 + iy) * sp_u;
                if (gy > L - 0.45 * sp_u) continue;
                if (anti_d ? (gx + gy <= L + shift_d) : (gx - gy > shift_d)) up.push_back({gx, gy, 0.0});
            }
        }
        for (int ix = 0; ix < k_max; ++ix) {
            double gx = anti_d ? L - (0.50 + ix) * sp_u : (0.50 + ix) * sp_u;
            if (gx < 0.45 * sp_u || gx > L - 0.45 * sp_u) continue;
            for (int iy = 0; iy < k_max; ++iy) {
                double gy = L - (0.50 + iy) * sp_u;
                if (gy < 0.45 * sp_u) continue;
                if (anti_d ? (gx + gy > L + shift_d) : (gx - gy <= shift_d)) up.push_back({gx, gy, 0.0});
            }
        }
        auto ok = sat_filter_against(up, pts, 0.87, L);
        pts.insert(pts.end(), ok.begin(), ok.end());
        add_corner_wedge_plugs(pts, n, L, 0.872, rng);
        int diff = (int)pts.size() - n;
        double pen = std::abs(diff) + (diff < -10 ? 2.0 * (-diff) : 0.0);
        if (pen < best_pen) {
            best_pen = pen;
            best_pts = std::move(pts);
            best_th_b = th_b;
            std::ostringstream os;
            os << "pyth9(" << pyth_tag[pick] << ",k=" << k_cl << "," << rw << "x" << rl << ",diff=" << diff << ")";
            best_desc = os.str();
            if (diff >= -1 && diff <= 1) break;
        }
    }
    if ((rng() % 2) == 1) for (auto& s : best_pts) { s.x = L - s.x; s.th = -s.th; }
    if ((rng() % 2) == 1) for (auto& s : best_pts) { std::swap(s.x, s.y); s.th = -s.th; }
    best_pts = add_interface_void_squares(std::move(best_pts), n, L, 0.875, std::abs(best_th_b), rng);
    best_pts = fill_wedges_and_complete(std::move(best_pts), n, L, rng, 0.87);
    add_jitter(best_pts, rng, 0.0025);
    desc = best_desc;
    return best_pts;
}

// =========================================================================
// Idea 10B: Twin Off-Diagonal Rectangular Pockets (NW +theta, SE +-theta)
//           & Flush-Sheared (du = +-tan(theta)) Asymmetric Bands
// =========================================================================

static inline double rand_uniform(std::mt19937_64& rng, double a, double b) {
    return std::uniform_real_distribution<double>(a, b)(rng);
}

static std::vector<Sq> seed_twin_rect_pockets_v10(
    int n, double L, std::mt19937_64& rng, std::string& desc, int slot_hint
) {
    // 8 target tilt angles (including Pythagorean atan(5/12)=22.62, 24.5, 25.5, atan(1/2)=26.57, atan(8/15)=28.07, 31.0, atan(3/4)=36.87, 40.5)
    static const double ang_deg[8] = {22.619865, 24.50, 25.50, 26.565051, 28.072487, 31.00, 36.869898, 40.50};
    int ia = slot_hint % 8;
    int m_wall = (slot_hint / 8) % 3; // 0, 1, or 2 outer 0-deg wall frame rows
    int chiral = (slot_hint / 24) % 2; // 0: NW(+th)/SE(-th), 1: NW(+th)/SE(+th)
    int shear_m = (slot_hint / 48) % 3; // 0: no shear, 1: +tan(th)*0.5, 2: -sin(th)*0.5

    double th1 = ang_deg[ia] * PI / 180.0 + rand_uniform(rng, -0.008, 0.008);
    double th2 = (chiral == 0) ? (-th1 + rand_uniform(rng, -0.008, 0.008)) : (th1 + rand_uniform(rng, -0.008, 0.008));

    // Scale pocket dimensions proportionally to L (target ~15%..26% total tilted squares)
    double base_w = std::max(4.5, std::min(13.5, std::round(L * rand_uniform(rng, 0.28, 0.42))));
    double base_h = std::max(4.5, std::min(13.5, std::round(L * rand_uniform(rng, 0.28, 0.42))));
    double w1 = base_w + rand_uniform(rng, -0.35, 0.35);
    double h1 = base_h + rand_uniform(rng, -0.35, 0.35);
    double w2 = base_w + rand_uniform(rng, -0.35, 0.35);
    double h2 = base_h + rand_uniform(rng, -0.35, 0.35);

    // NW cavity: x in [m_wall, m_wall + w1], y in [L - m_wall - h1, L - m_wall]
    double nw_x0 = (double)m_wall, nw_x1 = (double)m_wall + w1;
    double nw_y0 = L - (double)m_wall - h1, nw_y1 = L - (double)m_wall;

    // SE cavity: x in [L - m_wall - w2, L - m_wall], y in [m_wall, m_wall + h2]
    double se_x0 = L - (double)m_wall - w2, se_x1 = L - (double)m_wall;
    double se_y0 = (double)m_wall, se_y1 = (double)m_wall + h2;

    struct CandSq { Sq s; double prio; };
    std::vector<CandSq> cands;
    cands.reserve(n + 128);

    int K = (int)std::floor(L);
    // 1. Place 0-deg upright crystal outside NW and SE pockets, anchored to nearest corner (SW vs NE)
    for (int ix = 0; ix < K; ++ix) {
        for (int iy = 0; iy < K; ++iy) {
            bool use_ne = (ix + iy >= K);
            double x = use_ne ? (L - 0.5 - (double)(K - 1 - ix)) : (0.5 + (double)ix);
            double y = use_ne ? (L - 0.5 - (double)(K - 1 - iy)) : (0.5 + (double)iy);
            if (x < 0.48 || x > L - 0.48 || y < 0.48 || y > L - 0.48) continue;

            // Check if inside NW or SE pocket cavity
            bool in_nw = (x > nw_x0 + 0.15 && x < nw_x1 - 0.15 && y > nw_y0 + 0.15 && y < nw_y1 - 0.15);
            bool in_se = (x > se_x0 + 0.15 && x < se_x1 - 0.15 && y > se_y0 + 0.15 && y < se_y1 - 0.15);
            if (in_nw || in_se) continue;

            double d_seam = std::abs((x - 0.5 * L) + (y - 0.5 * L));
            double d_pock = std::min(
                std::max({nw_x0 - x, x - nw_x1, nw_y0 - y, y - nw_y1}),
                std::max({se_x0 - x, x - se_x1, se_y0 - y, y - se_y1})
            );
            // Keep deep corner/wall upright squares highest priority (lowest prio value); trim boundary ones first if excess
            double prio = -std::min(d_pock, 2.5) - 0.15 * d_seam + rand_uniform(rng, -0.05, 0.05);
            cands.push_back({{x, y, 0.0}, prio});
        }
    }

    // 2. Fill NW and SE pockets with sheared/unsheared tilted crystals
    auto fill_pocket = [&](double x0, double x1, double y0, double y1, double th) {
        double cx = 0.5 * (x0 + x1), cy = 0.5 * (y0 + y1);
        double c = std::cos(th), s = std::sin(th);
        double ext = sq_extent(th);
        double sh_u = (shear_m == 1) ? (0.35 * std::tan(th)) : ((shear_m == 2) ? (-0.35 * s) : 0.0);
        double u_ph = rand_uniform(rng, -0.45, 0.45);
        double v_ph = rand_uniform(rng, -0.45, 0.45);
        int R = (int)std::ceil(std::hypot(x1 - x0, y1 - y0)) + 2;
        for (int iv = -R; iv <= R; ++iv) {
            for (int iu = -R; iu <= R; ++iu) {
                double u = ((double)iu + u_ph + (double)iv * sh_u) * 0.995;
                double v = ((double)iv + v_ph) * 0.995;
                double x = cx + u * c - v * s;
                double y = cy + u * s + v * c;
                if (x < std::max(ext - 0.04, x0 + 0.32) || x > std::min(L - ext + 0.04, x1 - 0.32)) continue;
                if (y < std::max(ext - 0.04, y0 + 0.32) || y > std::min(L - ext + 0.04, y1 - 0.32)) continue;
                double d_edge = std::min({x - x0, x1 - x, y - y0, y1 - y});
                double prio = -1.2 - 0.4 * d_edge + rand_uniform(rng, -0.05, 0.05);
                cands.push_back({{x, y, th}, prio});
            }
        }
    };
    fill_pocket(nw_x0, nw_x1, nw_y0, nw_y1, th1);
    fill_pocket(se_x0, se_x1, se_y0, se_y1, th2);

    int diff = (int)cands.size() - n;
    std::sort(cands.begin(), cands.end(), [](const CandSq& a, const CandSq& b) { return a.prio < b.prio; });
    std::vector<Sq> out;
    out.reserve(n);
    for (int i = 0; i < std::min((int)cands.size(), n); ++i) out.push_back(cands[i].s);
    while ((int)out.size() < n) {
        out.push_back({rand_uniform(rng, 1.0, L - 1.0), rand_uniform(rng, 1.0, L - 1.0), (out.size() % 2 == 0) ? th1 : 0.0});
    }

    char buf[96];
    std::snprintf(buf, sizeof(buf), "twinpock10(a=%.0f,m=%d,ch=%d,sh=%d,d=%d)", ang_deg[ia], m_wall, chiral, shear_m, diff);
    desc = buf;
    return out;
}

static std::vector<Sq> seed_flush_sheared_v10(
    int n, double L, std::mt19937_64& rng, std::string& desc, int slot_hint
) {
    // Flush-sheared (du = +-tan(theta)) asymmetric diagonal/off-diagonal band between corner-anchored 0-deg crystals
    static const double th_deg[10] = {-22.619865, -26.565051, -28.072487, -36.869898, -40.50, -24.50, -31.00, -18.434949, 26.565051, 36.869898};
    int it = slot_hint % 10;
    int wb = 2 + ((slot_hint / 10) % 5); // band width 2..6
    int sh_mode = (slot_hint / 50) % 3;  // 0: +tan(th), 1: -tan(th), 2: +sin(th)

    double th_b = th_deg[it] * PI / 180.0 + rand_uniform(rng, -0.008, 0.008);
    double psi = (45.0 + rand_uniform(rng, -2.5, 2.5)) * PI / 180.0;
    double nx = -std::sin(psi), ny = std::cos(psi);
    double tx = std::cos(psi), ty = std::sin(psi);

    double off_n = rand_uniform(rng, -1.8, 1.8);
    double cx = 0.5 * L + off_n * nx;
    double cy = 0.5 * L + off_n * ny;
    double half_w = 0.5 * (double)wb * 0.96;

    double tan_clamped = std::max(-0.85, std::min(0.85, std::tan(th_b)));
    double du_sh = (sh_mode == 0) ? (0.45 * tan_clamped) : ((sh_mode == 1) ? (-0.45 * tan_clamped) : (0.45 * std::sin(th_b)));

    struct CandSq { Sq s; double prio; };
    std::vector<CandSq> cands;
    int K = (int)std::floor(L);

    // 1. Corner-anchored 0-deg crystals: t_lo anchored at (L, 0), t_hi anchored at (0, L)
    for (int ix = 0; ix < K; ++ix) {
        for (int iy = 0; iy < K; ++iy) {
            // Check both (L,0) and (0,L) anchors depending on side of band
            double x_lo = L - 0.5 - (double)(K - 1 - ix);
            double y_lo = 0.5 + (double)iy;
            double u_lo = (x_lo - cx) * nx + (y_lo - cy) * ny;
            if (u_lo < -half_w - 0.38 && x_lo >= 0.48 && y_lo <= L - 0.48) {
                cands.push_back({{x_lo, y_lo, 0.0}, -( -u_lo - half_w )});
            }
            double x_hi = 0.5 + (double)ix;
            double y_hi = L - 0.5 - (double)(K - 1 - iy);
            double u_hi = (x_hi - cx) * nx + (y_hi - cy) * ny;
            if (u_hi > half_w + 0.38 && x_hi <= L - 0.48 && y_hi >= 0.48) {
                cands.push_back({{x_hi, y_hi, 0.0}, -( u_hi - half_w )});
            }
        }
    }

    // 2. Sheared tilted band along (tx, ty) / (nx, ny)
    double cb = std::cos(th_b), sb = std::sin(th_b);
    double ext = sq_extent(th_b);
    int R = (int)std::ceil(1.5 * L) + 2;
    double ph_u = rand_uniform(rng, -0.45, 0.45);
    double ph_v = rand_uniform(rng, -0.45, 0.45);
    for (int iv = -R; iv <= R; ++iv) {
        for (int iu = -R; iu <= R; ++iu) {
            double u = ((double)iu + ph_u + (double)iv * du_sh) * 0.992;
            double v = ((double)iv + ph_v) * 0.992;
            double x = cx + u * cb - v * sb;
            double y = cy + u * sb + v * cb;
            if (x < ext - 0.03 || x > L - ext + 0.03 || y < ext - 0.03 || y > L - ext + 0.03) continue;
            double un = (x - cx) * nx + (y - cy) * ny;
            if (std::abs(un) <= half_w + 0.12) {
                double d_wall = std::min({x, L - x, y, L - y});
                cands.push_back({{x, y, th_b}, -1.5 - 0.1 * d_wall + 0.3 * std::abs(un)});
            }
        }
    }

    int diff = (int)cands.size() - n;
    std::sort(cands.begin(), cands.end(), [](const CandSq& a, const CandSq& b) { return a.prio < b.prio; });
    std::vector<Sq> out;
    out.reserve(n);
    for (int i = 0; i < std::min((int)cands.size(), n); ++i) out.push_back(cands[i].s);
    while ((int)out.size() < n) {
        out.push_back({rand_uniform(rng, 1.0, L - 1.0), rand_uniform(rng, 1.0, L - 1.0), th_b});
    }

    char buf[96];
    std::snprintf(buf, sizeof(buf), "fsh10(t=%.0f,w=%d,sh=%d,d=%d)", th_deg[it], wb, sh_mode, diff);
    desc = buf;
    return out;
}

// =========================================================================
// Augmented-Lagrangian + L-BFGS float64 L-Squeezer (Fixed & Free Angles)
// =========================================================================

struct PairAL {
    int i, j;
    double lam;
};

static double al_eval(
    int n,
    bool free_th,
    double rho,
    double w_L,
    double L,
    const std::vector<double>& x,
    const std::vector<double>& y,
    const std::vector<double>& th,
    const std::vector<double>& lam_w, // size 4*n: L,R,B,T
    const std::vector<PairAL>& pairs,
    std::vector<double>& gx,
    std::vector<double>& gy,
    std::vector<double>& gth,
    double& gL,
    double& max_v_out
) {
    std::fill(gx.begin(), gx.end(), 0.0);
    std::fill(gy.begin(), gy.end(), 0.0);
    if (free_th) std::fill(gth.begin(), gth.end(), 0.0);
    gL = w_L;
    double obj = w_L * L;
    max_v_out = 0.0;

    std::vector<double> c(n), s(n), dext(n);
    const double eps2 = 1e-22;
    for (int i = 0; i < n; ++i) {
        c[i] = std::cos(th[i]);
        s[i] = std::sin(th[i]);
        double ext = 0.5 * (std::abs(c[i]) + std::abs(s[i]));
        double sc_s = c[i] / std::sqrt(c[i] * c[i] + eps2);
        double ss_s = s[i] / std::sqrt(s[i] * s[i] + eps2);
        double de = free_th ? (0.5 * (ss_s * c[i] - sc_s * s[i])) : 0.0;

        double vs[4] = {ext - x[i], x[i] + ext - L, ext - y[i], y[i] + ext - L};
        double mul[4];
        for (int k = 0; k < 4; ++k) {
            if (vs[k] > max_v_out) max_v_out = vs[k];
            double lw = lam_w[4 * i + k];
            double t = lw + rho * vs[k];
            if (t > 0.0) {
                obj += 0.5 * (t * t - lw * lw) / rho;
                mul[k] = t;
            } else {
                obj -= 0.5 * lw * lw / rho;
                mul[k] = 0.0;
            }
        }
        gx[i] += mul[1] - mul[0];
        gy[i] += mul[3] - mul[2];
        gL -= (mul[1] + mul[3]);
        if (free_th) gth[i] += (mul[0] + mul[1] + mul[2] + mul[3]) * de;
    }

    for (const auto& pr : pairs) {
        int i = pr.i, j = pr.j;
        double dx = x[j] - x[i], dy = y[j] - y[i];
        if (dx * dx + dy * dy >= 2.25) continue;
        double ci = c[i], si = s[i], cj = c[j], sj = s[j];
        double cr = ci * cj + si * sj;
        double sr = si * cj - ci * sj;
        double rad = 0.5 * (1.0 + std::abs(cr) + std::abs(sr));

        double p0 = dx * ci + dy * si, p1 = -dx * si + dy * ci;
        double p2 = dx * cj + dy * sj, p3 = -dx * sj + dy * cj;
        double z0 = std::abs(p0) - rad, z1 = std::abs(p1) - rad;
        double z2 = std::abs(p2) - rad, z3 = std::abs(p3) - rad;
        double zmax = z0, ps = p0, ux = ci, uy = si, is_i = 1.0;
        if (z1 > zmax) { zmax = z1; ps = p1; ux = -si; uy = ci; is_i = 1.0; }
        if (z2 > zmax) { zmax = z2; ps = p2; ux = cj;  uy = sj; is_i = 0.0; }
        if (z3 > zmax) { zmax = z3; ps = p3; ux = -sj; uy = cj; is_i = 0.0; }
        double v = -zmax; // positive = overlap
        if (v > max_v_out) max_v_out = v;

        double t = pr.lam + rho * v;
        if (t > 0.0) {
            obj += 0.5 * (t * t - pr.lam * pr.lam) / rho;
            double sp = (ps > 0.0) ? 1.0 : ((ps < 0.0) ? -1.0 : 1.0);
            gx[i] += t * sp * ux;
            gy[i] += t * sp * uy;
            gx[j] -= t * sp * ux;
            gy[j] -= t * sp * uy;
            if (free_th) {
                double drad = 0.5 * ((cr / std::sqrt(cr * cr + eps2)) * (-sr) + (sr / std::sqrt(sr * sr + eps2)) * cr);
                double rot_term = sp * (dy * ux - dx * uy);
                gth[i] += t * (drad - rot_term * is_i);
                gth[j] += t * (-drad - rot_term * (1.0 - is_i));
            }
        } else {
            obj -= 0.5 * pr.lam * pr.lam / rho;
        }
    }
    return obj;
}

static std::vector<PairAL> rebuild_pairs(
    int n, const std::vector<double>& x, const std::vector<double>& y,
    const std::unordered_map<uint64_t, double>& old_lams
) {
    double xmin = *std::min_element(x.begin(), x.end());
    double xmax = *std::max_element(x.begin(), x.end());
    double ymin = *std::min_element(y.begin(), y.end());
    double ymax = *std::max_element(y.begin(), y.end());
    double cw = 1.65;
    int Gx = std::max(4, (int)std::ceil((xmax - xmin + 3.0) / cw));
    int Gy = std::max(4, (int)std::ceil((ymax - ymin + 3.0) / cw));
    std::vector<std::vector<int>> grid(Gx * Gy);
    for (int i = 0; i < n; ++i) {
        int cx = std::max(0, std::min(Gx - 1, (int)std::floor((x[i] - xmin + 1.0) / cw)));
        int cy = std::max(0, std::min(Gy - 1, (int)std::floor((y[i] - ymin + 1.0) / cw)));
        grid[cy * Gx + cx].push_back(i);
    }
    std::vector<PairAL> pairs;
    for (int i = 0; i < n; ++i) {
        int cx = std::max(0, std::min(Gx - 1, (int)std::floor((x[i] - xmin + 1.0) / cw)));
        int cy = std::max(0, std::min(Gy - 1, (int)std::floor((y[i] - ymin + 1.0) / cw)));
        for (int dy = -1; dy <= 1; ++dy) {
            int ny = cy + dy; if (ny < 0 || ny >= Gy) continue;
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = cx + dx; if (nx < 0 || nx >= Gx) continue;
                for (int j : grid[ny * Gx + nx]) {
                    if (j <= i) continue;
                    double ddx = x[j] - x[i], ddy = y[j] - y[i];
                    if (ddx * ddx + ddy * ddy < 2.65) {
                        uint64_t key = ((uint64_t)i << 20) | (uint64_t)j;
                        auto it = old_lams.find(key);
                        pairs.push_back({i, j, it != old_lams.end() ? it->second : 0.0});
                    }
                }
            }
        }
    }
    return pairs;
}

static void al_lbfgs_round(
    int n, bool free_th, double rho, double w_L, int max_iters,
    std::vector<double>& x, std::vector<double>& y, std::vector<double>& th, double& L,
    std::vector<double>& lam_w, std::vector<PairAL>& pairs
) {
    int dim = free_th ? (3 * n + 1) : (2 * n + 1);
    const int M = 12;
    std::vector<double> gx(n), gy(n), gth(n, 0.0);
    std::vector<double> q(dim), g(dim), q_prev(dim), g_prev(dim), d(dim);
    std::vector<std::vector<double>> S_hist, Y_hist;
    std::vector<double> rho_hist;

    auto pack = [&](const std::vector<double>& vx, const std::vector<double>& vy,
                    const std::vector<double>& vth, double vL, std::vector<double>& out) {
        for (int i = 0; i < n; ++i) { out[2 * i] = vx[i]; out[2 * i + 1] = vy[i]; }
        if (free_th) for (int i = 0; i < n; ++i) out[2 * n + i] = vth[i];
        out[dim - 1] = vL;
    };
    auto unpack = [&](const std::vector<double>& in, std::vector<double>& vx,
                      std::vector<double>& vy, std::vector<double>& vth, double& vL) {
        for (int i = 0; i < n; ++i) { vx[i] = in[2 * i]; vy[i] = in[2 * i + 1]; }
        if (free_th) for (int i = 0; i < n; ++i) vth[i] = in[2 * n + i];
        vL = in[dim - 1];
    };

    double gL = 0.0, maxv = 0.0;
    double f = al_eval(n, free_th, rho, w_L, L, x, y, th, lam_w, pairs, gx, gy, gth, gL, maxv);
    pack(x, y, th, L, q);
    pack(gx, gy, gth, gL, g);

    std::vector<double> xt(n), yt(n), tht = th;
    std::vector<double> gxt(n), gyt(n), gtht(n, 0.0);

    for (int it = 0; it < max_iters; ++it) {
        double gnorm2 = 0.0;
        for (double v : g) gnorm2 += v * v;
        if (gnorm2 < 1e-22) break;

        // L-BFGS 2-loop direction
        std::copy(g.begin(), g.end(), d.begin());
        int k = (int)S_hist.size();
        std::vector<double> alpha(k, 0.0);
        for (int j = k - 1; j >= 0; --j) {
            double dot = 0.0;
            for (int r = 0; r < dim; ++r) dot += S_hist[j][r] * d[r];
            alpha[j] = rho_hist[j] * dot;
            for (int r = 0; r < dim; ++r) d[r] -= alpha[j] * Y_hist[j][r];
        }
        double gamma = 1.0 / std::max(1.0, std::sqrt(gnorm2));
        if (k > 0) {
            double sy = 0.0, yy = 0.0;
            for (int r = 0; r < dim; ++r) { sy += S_hist.back()[r] * Y_hist.back()[r]; yy += Y_hist.back()[r] * Y_hist.back()[r]; }
            if (yy > 1e-24 && sy > 0.0) gamma = sy / yy;
        }
        for (int r = 0; r < dim; ++r) d[r] *= gamma;
        for (int j = 0; j < k; ++j) {
            double dot = 0.0;
            for (int r = 0; r < dim; ++r) dot += Y_hist[j][r] * d[r];
            double beta = rho_hist[j] * dot;
            for (int r = 0; r < dim; ++r) d[r] += S_hist[j][r] * (alpha[j] - beta);
        }
        for (int r = 0; r < dim; ++r) d[r] = -d[r];

        double dg = 0.0;
        for (int r = 0; r < dim; ++r) dg += d[r] * g[r];
        if (dg >= 0.0) {
            S_hist.clear(); Y_hist.clear(); rho_hist.clear();
            double sc = 1.0 / std::max(1.0, std::sqrt(gnorm2));
            for (int r = 0; r < dim; ++r) d[r] = -sc * g[r];
            dg = -sc * gnorm2;
        }
        double step = 1.0;
        double dmax = 0.0;
        for (double v : d) dmax = std::max(dmax, std::abs(v));
        if (dmax > 0.1) step = 0.1 / dmax;

        double f_new = f, gLt = 0.0, mv_t = 0.0;
        bool ok = false;
        std::vector<double> q_try(dim), g_try(dim);
        for (int ls = 0; ls < 18; ++ls) {
            for (int r = 0; r < dim; ++r) q_try[r] = q[r] + step * d[r];
            double Lt = L;
            unpack(q_try, xt, yt, tht, Lt);
            f_new = al_eval(n, free_th, rho, w_L, Lt, xt, yt, tht, lam_w, pairs, gxt, gyt, gtht, gLt, mv_t);
            if (f_new <= f + 1e-4 * step * dg) {
                pack(gxt, gyt, gtht, gLt, g_try);
                ok = true;
                break;
            }
            step *= 0.5;
        }
        if (!ok) {
            S_hist.clear(); Y_hist.clear(); rho_hist.clear();
            continue;
        }
        std::vector<double> s_vec(dim), y_vec(dim);
        double sy = 0.0;
        for (int r = 0; r < dim; ++r) {
            s_vec[r] = q_try[r] - q[r];
            y_vec[r] = g_try[r] - g[r];
            sy += s_vec[r] * y_vec[r];
        }
        q.swap(q_try); g.swap(g_try); f = f_new;
        unpack(q, x, y, th, L);
        if (sy > 1e-18) {
            if ((int)S_hist.size() == M) {
                S_hist.erase(S_hist.begin());
                Y_hist.erase(Y_hist.begin());
                rho_hist.erase(rho_hist.begin());
            }
            S_hist.push_back(std::move(s_vec));
            Y_hist.push_back(std::move(y_vec));
            rho_hist.push_back(1.0 / sy);
        }
        if (step * dmax < 1e-12) break;
    }

    // Dual multiplier update (with safety offset -1e-10 so feasible target has strict clearance)
    for (int i = 0; i < n; ++i) {
        double ext = sq_extent(th[i]);
        double vs[4] = {ext - x[i] + 1e-10, x[i] + ext - L + 1e-10, ext - y[i] + 1e-10, y[i] + ext - L + 1e-10};
        for (int k = 0; k < 4; ++k) lam_w[4 * i + k] = std::min(1e4, std::max(0.0, lam_w[4 * i + k] + rho * vs[k]));
    }
    std::vector<double> c(n), s(n);
    for (int i = 0; i < n; ++i) { c[i] = std::cos(th[i]); s[i] = std::sin(th[i]); }
    for (auto& pr : pairs) {
        int i = pr.i, j = pr.j;
        double dx = x[j] - x[i], dy = y[j] - y[i];
        if (dx * dx + dy * dy >= 2.25) { pr.lam = std::max(0.0, pr.lam * 0.5); continue; }
        double ci = c[i], si = s[i], cj = c[j], sj = s[j];
        double rad = 0.5 * (1.0 + std::abs(ci * cj + si * sj) + std::abs(si * cj - ci * sj));
        double p0 = std::abs(dx * ci + dy * si), p1 = std::abs(-dx * si + dy * ci);
        double p2 = std::abs(dx * cj + dy * sj), p3 = std::abs(-dx * sj + dy * cj);
        double v = rad - std::max(std::max(p0, p1), std::max(p2, p3)) + 1e-10;
        pr.lam = std::min(1e4, std::max(0.0, pr.lam + rho * v));
    }
}

static double squeeze_candidate_al(std::vector<Sq>& pts, bool try_free_angles = true) {
    int n = (int)pts.size();
    minimal_feasible_dilation(pts, 1.5e-10);
    double best_L = exact_bbox_side(pts);
    std::vector<Sq> best_pts = pts;

    std::vector<double> x(n), y(n), th(n);
    for (int i = 0; i < n; ++i) { x[i] = pts[i].x; y[i] = pts[i].y; th[i] = pts[i].th; }
    double L = best_L;

    std::vector<double> lam_w(4 * n, 0.0);
    std::unordered_map<uint64_t, double> pair_lams;

    // Stage A: Fixed-angle AL-LBFGS (replaces unbounded fixed_angle_lp_polish)
    double rho = 250.0;
    for (int rd = 0; rd < 8; ++rd) {
        auto pairs = rebuild_pairs(n, x, y, pair_lams);
        al_lbfgs_round(n, false, rho, 1.0, 140, x, y, th, L, lam_w, pairs);
        pair_lams.clear();
        for (const auto& pr : pairs) if (pr.lam > 0.0) pair_lams[((uint64_t)pr.i << 20) | (uint64_t)pr.j] = pr.lam;
        std::vector<Sq> cand(n);
        for (int i = 0; i < n; ++i) cand[i] = {x[i], y[i], th[i]};
        double lam = minimal_feasible_dilation(cand, 1.5e-10);
        if (lam < 1.05) {
            double cL = exact_bbox_side(cand);
            if (cL < best_L) { best_L = cL; best_pts = cand; }
        }
        rho = std::min(8e5, rho * 3.0);
    }

    // Re-sync working state from best feasible fixed-angle config
    for (int i = 0; i < n; ++i) { x[i] = best_pts[i].x; y[i] = best_pts[i].y; th[i] = best_pts[i].th; }
    L = best_L;

    if (try_free_angles) {
        // Stage B: Free-angle AL-LBFGS + final fixed-angle polish
        rho = 4000.0;
        for (int rd = 0; rd < 6; ++rd) {
            auto pairs = rebuild_pairs(n, x, y, pair_lams);
            al_lbfgs_round(n, true, rho, 1.0, 110, x, y, th, L, lam_w, pairs);
            pair_lams.clear();
            for (const auto& pr : pairs) if (pr.lam > 0.0) pair_lams[((uint64_t)pr.i << 20) | (uint64_t)pr.j] = pr.lam;
            std::vector<Sq> cand(n);
            for (int i = 0; i < n; ++i) cand[i] = {x[i], y[i], th[i]};
            double lam = minimal_feasible_dilation(cand, 1.5e-10);
            if (lam < 1.05) {
                double cL = exact_bbox_side(cand);
                if (cL < best_L) { best_L = cL; best_pts = cand; }
            }
            rho = std::min(1e6, rho * 2.5);
        }
        for (int i = 0; i < n; ++i) { x[i] = best_pts[i].x; y[i] = best_pts[i].y; th[i] = best_pts[i].th; }
        L = best_L;
        for (int rd = 0; rd < 3; ++rd) {
            auto pairs = rebuild_pairs(n, x, y, pair_lams);
            al_lbfgs_round(n, false, rho, 1.0, 100, x, y, th, L, lam_w, pairs);
            pair_lams.clear();
            for (const auto& pr : pairs) if (pr.lam > 0.0) pair_lams[((uint64_t)pr.i << 20) | (uint64_t)pr.j] = pr.lam;
            std::vector<Sq> cand(n);
            for (int i = 0; i < n; ++i) cand[i] = {x[i], y[i], th[i]};
            double lam = minimal_feasible_dilation(cand, 1.5e-10);
            if (lam < 1.05) {
                double cL = exact_bbox_side(cand);
                if (cL < best_L) { best_L = cL; best_pts = cand; }
            }
        }
    }

    pts = std::move(best_pts);
    minimal_feasible_dilation(pts, 2e-10);
    return exact_bbox_side(pts);
}

static bool write_and_selfcheck_csv(int n, const std::vector<Sq>& pts_in, const std::string& path, double& out_L) {
    std::vector<Sq> pts = pts_in;
    shift_positive(pts, 1e-11);
    for (double eps : {0.0, 1e-12, 1e-11, 1e-10, 1e-9}) {
        std::vector<Sq> cur = pts;
        if (eps > 0.0) {
            for (auto& s : cur) { s.x *= (1.0 + eps); s.y *= (1.0 + eps); }
        }
        // Round through exact 17-digit text formatting matching pandas/CSV roundtrip
        std::ostringstream buf;
        buf << "id,n,square_id,x,y,angle\n";
        std::vector<Sq> parsed(n);
        for (int i = 0; i < n; ++i) {
            std::ostringstream sx, sy, sa;
            sx << std::setprecision(17) << cur[i].x;
            sy << std::setprecision(17) << cur[i].y;
            sa << std::setprecision(17) << cur[i].th;
            parsed[i] = {std::stod(sx.str()), std::stod(sy.str()), std::stod(sa.str())};
            buf << i << "," << n << "," << i << "," << sx.str() << "," << sy.str() << "," << sa.str() << "\n";
        }
        double pen = sat_max_penetration(parsed);
        double minx = 1e18, miny = 1e18;
        for (const auto& s : parsed) {
            double e = sq_extent(s.th);
            minx = std::min(minx, s.x - e);
            miny = std::min(miny, s.y - e);
        }
        if (pen <= 0.0 && minx >= 0.0 && miny >= 0.0) {
            std::ofstream fout(path);
            fout << buf.str();
            out_L = exact_bbox_side(parsed);
            return true;
        }
    }
    return false;
}

static std::vector<TargetInfo> load_targets(const std::vector<int>& n_list) {
    // Exact inc_L from merged.csv & L_frontier from per_n_gap.csv
    std::unordered_map<int, double> front_map;
    std::ifstream fg("poc/frontier_nov12/docs/per_n_gap.csv");
    std::string line;
    std::getline(fg, line);
    while (std::getline(fg, line)) {
        std::stringstream ss(line);
        std::string tok;
        std::getline(ss, tok, ',');
        int ni = std::stoi(tok);
        std::getline(ss, tok, ','); // L
        std::getline(ss, tok, ','); // L_frontier
        front_map[ni] = std::stod(tok);
    }

    std::unordered_map<int, std::vector<Sq>> by_n;
    for (int n : n_list) by_n[n] = {};
    std::ifstream fm("poc/pell_corrugation/data/border-scan/merged.csv");
    std::getline(fm, line);
    while (std::getline(fm, line)) {
        std::stringstream ss(line);
        std::string id_s, n_s, sq_s, x_s, y_s, a_s;
        std::getline(ss, id_s, ',');
        std::getline(ss, n_s, ',');
        int ni = std::stoi(n_s);
        if (by_n.find(ni) == by_n.end()) continue;
        std::getline(ss, sq_s, ',');
        std::getline(ss, x_s, ',');
        std::getline(ss, y_s, ',');
        std::getline(ss, a_s, ',');
        by_n[ni].push_back({std::stod(x_s), std::stod(y_s), std::stod(a_s)});
    }
    std::vector<TargetInfo> out;
    for (int n : n_list) {
        double inc_L = exact_bbox_side(by_n[n]);
        double fl = front_map.count(n) ? front_map[n] : (inc_L - 0.03);
        out.push_back({n, fl, inc_L});
    }
    return out;
}

static std::vector<std::vector<Sq>> load_surg_bin(int n, const std::string& surg_dir) {
    std::vector<std::vector<Sq>> out;
    if (surg_dir.empty()) return out;
    std::string ep = surg_dir + "/surg_n" + std::to_string(n) + ".bin";
    FILE* f = std::fopen(ep.c_str(), "rb");
    if (!f) return out;
    int32_t hdr[2] = {0, 0};
    if (std::fread(hdr, sizeof(int32_t), 2, f) == 2 && hdr[1] == n && hdr[0] > 0 && hdr[0] <= 4096) {
        int M = hdr[0];
        out.reserve(M);
        std::vector<double> ax(n), ay(n), ath(n);
        for (int k = 0; k < M; ++k) {
            if (std::fread(ax.data(), sizeof(double), n, f) != (size_t)n) break;
            if (std::fread(ay.data(), sizeof(double), n, f) != (size_t)n) break;
            if (std::fread(ath.data(), sizeof(double), n, f) != (size_t)n) break;
            std::vector<Sq> pts(n);
            for (int i = 0; i < n; ++i) pts[i] = {ax[i], ay[i], ath[i]};
            out.push_back(std::move(pts));
        }
    }
    std::fclose(f);
    return out;
}

int main(int argc, char** argv) {
    omp_set_num_threads(7);
    std::vector<int> n_list = {367, 910, 971};
    int batch = 336, top_k = 32, seed = 2, fast_mode = 1, elastic_L = 1, idea = 4;
    std::string export_top_dir, warm_start_dir, surg_dir;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--n-list") == 0 && i + 1 < argc) {
            n_list.clear();
            std::stringstream ss(argv[++i]);
            std::string tok;
            while (std::getline(ss, tok, ',')) if (!tok.empty()) n_list.push_back(std::stoi(tok));
        } else if (std::strcmp(argv[i], "--batch") == 0 && i + 1 < argc) batch = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) top_k = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--fast-mode") == 0 && i + 1 < argc) fast_mode = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--elastic-L") == 0 && i + 1 < argc) elastic_L = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--idea") == 0 && i + 1 < argc) idea = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--export-top") == 0 && i + 1 < argc) export_top_dir = argv[++i];
        else if (std::strcmp(argv[i], "--warm-start-dir") == 0 && i + 1 < argc) warm_start_dir = argv[++i];
        else if (std::strcmp(argv[i], "--surg-dir") == 0 && i + 1 < argc) surg_dir = argv[++i];
    }

    const std::string cu_src = "poc/swarm_inflation/poc/sat_lmin.cu";
    const std::string cu_bin = "/tmp/sat_lmin_cuda";
    if (!fs::exists(cu_bin) || fs::last_write_time(cu_src) > fs::last_write_time(cu_bin)) {
        std::string cmd = "/usr/local/cuda/bin/nvcc -O3 -arch=sm_89 " + cu_src + " -o " + cu_bin;
        std::printf("[cuda] %s\n", cmd.c_str());
        if (std::system(cmd.c_str()) != 0) return 1;
    }

    fs::create_directories("poc/swarm_inflation/docs");
    fs::create_directories("poc/swarm_inflation/data");

    auto targets = load_targets(n_list);
    for (const auto& tg : targets) {
        int n = tg.n;
        double f_L = tg.L_front, inc_L = tg.inc_L;
        double L_targets[4] = {
            (idea >= 6) ? (f_L + 0.30 * (inc_L - f_L)) : (f_L + 0.001),
            (idea >= 6) ? (f_L + 0.65 * (inc_L - f_L)) : (f_L + 0.35 * (inc_L - f_L)),
            (idea >= 6) ? (inc_L - 0.0055) : (f_L + 0.72 * (inc_L - f_L)),
            (idea >= 6) ? (inc_L - 0.0012) : ((idea >= 3) ? (inc_L - 0.001) : (inc_L + 0.012))
        };
        auto warm_pool = (idea >= 6) ? load_warm_basins(n, warm_start_dir) : std::vector<std::vector<Sq>>{};
        auto surg_pool = (idea >= 7) ? load_surg_bin(n, surg_dir) : std::vector<std::vector<Sq>>{};
        std::printf("=== Idea %d (C++/CUDA) n=%d batch=%d elastic=%d fast=%d warm=%zu surg=%zu f_L=%.6f inc_L=%.6f ===\n",
                    idea, n, batch, elastic_L, fast_mode, warm_pool.size(), surg_pool.size(), f_L, inc_L);
        std::fflush(stdout);

        double t0 = omp_get_wtime();
        std::vector<std::vector<Sq>> seeds_pts(batch);
        std::vector<std::string> meta(batch);
        std::vector<float> box_L(batch);

        #pragma omp parallel for schedule(dynamic, 1)
        for (int b = 0; b < batch; ++b) {
            std::mt19937_64 rng((uint64_t)(seed * 1009 + n) * 1000003ULL + (uint64_t)b * 7919ULL + 17ULL);
            double L_val = L_targets[b % 4];
            std::string desc;
            if (idea == 2) {
                int fam = b % 7;
                if (fam == 0 || fam == 1) seeds_pts[b] = seed_tapered_sheared_l(n, L_val, rng, desc);
                else if (fam == 2) seeds_pts[b] = seed_pinwheel_4pocket(n, L_val, rng, desc);
                else if (fam == 3) seeds_pts[b] = seed_45_upright_hybrids(n, L_val, rng, desc);
                else if (fam == 4) seeds_pts[b] = seed_clipped_chiral(n, L_val, rng, desc);
                else if (fam == 5) seeds_pts[b] = seed_2tier_pocket(n, L_val, rng, desc);
                else seeds_pts[b] = seed_pythagorean_bicryst(n, L_val, rng, desc);
            } else if (idea == 3) {
                int fam = b % 10;
                if (fam <= 3) seeds_pts[b] = seed_smectic_1strip(n, L_val, rng, desc);
                else if (fam <= 5) seeds_pts[b] = seed_smectic_2strip_l(n, L_val, rng, desc);
                else if (fam <= 7) seeds_pts[b] = seed_sandwich_fanned(n, L_val, rng, desc);
                else if (fam == 8) seeds_pts[b] = ((b / 10) % 2 == 0)
                    ? seed_tapered_sheared_l(n, L_val, rng, desc)
                    : seed_45_upright_hybrids(n, L_val, rng, desc);
                else seeds_pts[b] = ((b / 10) % 2 == 0)
                    ? seed_pinwheel_4pocket(n, L_val, rng, desc)
                    : seed_2tier_pocket(n, L_val, rng, desc);
            } else if (idea == 4) {
                int fam = b % 10;
                if (fam <= 4) seeds_pts[b] = seed_45band_exact(n, L_val, rng, desc);
                else if (fam <= 7) seeds_pts[b] = seed_smectic1_unit(n, L_val, rng, desc);
                else if (fam == 8) seeds_pts[b] = seed_tapered_L_unit(n, L_val, rng, desc);
                else seeds_pts[b] = ((b / 10) % 2 == 0)
                    ? seed_45_upright_hybrids(n, L_val, rng, desc)
                    : seed_tapered_sheared_l(n, L_val, rng, desc);
            } else if (idea == 5) {
                int fam = b % 10;
                if (fam <= 3) seeds_pts[b] = seed_hybrid45_v5(n, L_val, rng, desc);
                else if (fam <= 6) seeds_pts[b] = seed_tapered_L_v5(n, L_val, rng, desc);
                else if (fam <= 8) seeds_pts[b] = seed_smec1_v5(n, L_val, rng, desc);
                else seeds_pts[b] = ((b / 10) % 2 == 0)
                    ? seed_pinwheel_4pocket(n, L_val, rng, desc)
                    : seed_smectic_1strip(n, L_val, rng, desc);
            } else if (idea == 6) {
                int fam = b % 12;
                if (fam <= 3) seeds_pts[b] = seed_sheared_band_v6(n, L_val, rng, desc);
                else if (fam <= 5) seeds_pts[b] = seed_wrapped_dual_L_v6(n, L_val, rng, desc);
                else if (fam == 6) seeds_pts[b] = seed_bent_band_v6(n, L_val, rng, desc);
                else if (fam == 7) seeds_pts[b] = seed_hybrid45_v5(n, L_val, rng, desc);
                else if (fam == 8) seeds_pts[b] = ((b / 12) % 2 == 0)
                    ? seed_smec1_v5(n, L_val, rng, desc)
                    : seed_tapered_L_v5(n, L_val, rng, desc);
                else if (!warm_pool.empty()) {
                    const auto& bw = warm_pool[(b / 12 + fam) % warm_pool.size()];
                    seeds_pts[b] = seed_ruin_warm_v6(n, L_val, bw, rng, desc);
                } else {
                    seeds_pts[b] = (fam == 9)
                        ? seed_sheared_band_v6(n, L_val, rng, desc)
                        : seed_wrapped_dual_L_v6(n, L_val, rng, desc);
                }
            } else if (idea == 7) {
                // Idea 7: ~50% LP-dual surgical ruin candidates + ~50% fresh scratch
                if (!surg_pool.empty() && (b % 2 == 0)) {
                    int sidx = (b / 2) % (int)surg_pool.size();
                    seeds_pts[b] = surg_pool[sidx];
                    L_val = std::min(exact_bbox_side(seeds_pts[b]), inc_L - 0.0008);
                    const char* ops[4] = {"surg7_evict", "surg7_flip", "surg7_cut", "surg7_btilt"};
                    desc = ops[sidx % 4];
                } else {
                    int fam = (b / 2) % 10;
                    if (fam <= 4) seeds_pts[b] = seed_sheared_band_v6(n, L_val, rng, desc);
                    else if (fam <= 6) seeds_pts[b] = seed_wrapped_dual_L_v6(n, L_val, rng, desc);
                    else if (fam == 7) seeds_pts[b] = seed_bent_band_v6(n, L_val, rng, desc);
                    else if (fam == 8) seeds_pts[b] = seed_hybrid45_v5(n, L_val, rng, desc);
                    else seeds_pts[b] = ((b / 20) % 2 == 0)
                        ? seed_smec1_v5(n, L_val, rng, desc)
                        : seed_tapered_L_v5(n, L_val, rng, desc);
                }
            } else if (idea == 8) {
                // Idea 8: ~50% seam-slide / corner-refold / stoich surgery + ~50% new v8 topologies
                if (!surg_pool.empty() && (b % 2 == 0)) {
                    int sidx = (b / 2) % (int)surg_pool.size();
                    seeds_pts[b] = surg_pool[sidx];
                    L_val = std::min(exact_bbox_side(seeds_pts[b]), inc_L - 0.0006);
                    const char* ops[4] = {"surg8_slide", "surg8_refold", "surg8_stoich", "surg8_dualtilt"};
                    desc = ops[sidx % 4];
                } else {
                    int fam = (b / 2) % 10;
                    if (fam <= 2) seeds_pts[b] = seed_framed_band_v8(n, L_val, rng, desc);
                    else if (fam <= 4) seeds_pts[b] = seed_asym_band_v8(n, L_val, rng, desc);
                    else if (fam <= 6) seeds_pts[b] = seed_framed_tilted_v8(n, L_val, rng, desc);
                    else if (fam <= 8) seeds_pts[b] = seed_sheared_band_v6(n, L_val, rng, desc);
                    else seeds_pts[b] = ((b / 20) % 2 == 0)
                        ? seed_wrapped_dual_L_v6(n, L_val, rng, desc)
                        : seed_bent_band_v6(n, L_val, rng, desc);
                }
            } else if (idea == 9) {
                // Idea 9: 55% systematic wall-anchored asym9 + 25% Pythagorean diamond clusters + 20% fband8/ftilt8/surg
                if (!surg_pool.empty() && (b % 5 == 0)) {
                    int sidx = (b / 5) % (int)surg_pool.size();
                    seeds_pts[b] = surg_pool[sidx];
                    L_val = std::min(exact_bbox_side(seeds_pts[b]), inc_L - 0.0006);
                    const char* ops[4] = {"surg9_slide", "surg9_refold", "surg9_stoich", "surg9_dualtilt"};
                    desc = ops[sidx % 4];
                } else {
                    int fam = b % 20;
                    int grp = b / 20;
                    if (fam < 11) {
                        int asym_slot = grp * 11 + fam;
                        seeds_pts[b] = seed_asym_band_v9(n, L_val, rng, desc, asym_slot);
                    } else if (fam < 16) {
                        int pyth_slot = grp * 5 + (fam - 11);
                        seeds_pts[b] = seed_pyth_diamond_v9(n, L_val, rng, desc, pyth_slot);
                    } else if (fam < 18) {
                        seeds_pts[b] = seed_framed_band_v8(n, L_val, rng, desc);
                    } else if (fam == 18) {
                        seeds_pts[b] = seed_framed_tilted_v8(n, L_val, rng, desc);
                    } else {
                        seeds_pts[b] = seed_sheared_band_v6(n, L_val, rng, desc);
                    }
                }
            } else if (idea == 10) {
                // Idea 10: Gentle elastic-L transfer pool (100% when fast_mode==2 & surg_pool present)
                //          + Twin off-diagonal rectangular pockets (NW/SE) + Flush-sheared bands
                if (!surg_pool.empty() && (fast_mode == 2 || (b % 3 == 0))) {
                    int sidx = b % (int)surg_pool.size();
                    seeds_pts[b] = surg_pool[sidx];
                    L_val = std::min(exact_bbox_side(seeds_pts[b]), inc_L - 0.0005);
                    const char* ops[4] = {"surg10_glide", "surg10_hole", "surg10_dual", "surg10_stoich"};
                    desc = ops[sidx % 4];
                } else {
                    int fam = b % 20;
                    int grp = b / 20;
                    if (fam < 11) {
                        seeds_pts[b] = seed_twin_rect_pockets_v10(n, L_val, rng, desc, grp * 11 + fam);
                    } else if (fam < 18) {
                        seeds_pts[b] = seed_flush_sheared_v10(n, L_val, rng, desc, grp * 7 + (fam - 11));
                    } else {
                        seeds_pts[b] = seed_asym_band_v9(n, L_val, rng, desc, b * 131 + fam);
                    }
                }
            } else {
                // Idea 11: Active LP-dual force-chain buckling & slip-line surgery under fast_mode=4/2
                if (!surg_pool.empty() && (fast_mode == 4 || fast_mode == 2 || (b % 2 == 0))) {
                    int sidx = b % (int)surg_pool.size();
                    seeds_pts[b] = surg_pool[sidx];
                    L_val = exact_bbox_side(seeds_pts[b]);
                    const char* ops[5] = {"surg11_buckle", "surg11_slipline", "surg11_vachop", "surg11_aniso", "surg11_xfer"};
                    desc = ops[sidx % 5];
                } else {
                    int fam = b % 20;
                    int grp = b / 20;
                    if (fam < 12) {
                        seeds_pts[b] = seed_asym_band_v9(n, L_val, rng, desc, grp * 12 + fam);
                    } else if (fam < 17) {
                        seeds_pts[b] = seed_flush_sheared_v10(n, L_val, rng, desc, grp * 5 + (fam - 12));
                    } else {
                        seeds_pts[b] = seed_pyth_diamond_v9(n, L_val, rng, desc, grp * 3 + (fam - 17));
                    }
                }
            }
            box_L[b] = (float)L_val;
            char buf_m[128];
            std::snprintf(buf_m, sizeof(buf_m), "%s@L=%.5f", desc.c_str(), L_val);
            meta[b] = buf_m;
        }
        double t_seed = omp_get_wtime() - t0;

        size_t bn = (size_t)batch * (size_t)n;
        std::vector<float> hx(bn), hy(bn), hth(bn);
        for (int b = 0; b < batch; ++b) {
            for (int i = 0; i < n; ++i) {
                hx[(size_t)b * n + i] = (float)seeds_pts[b][i].x;
                hy[(size_t)b * n + i] = (float)seeds_pts[b][i].y;
                hth[(size_t)b * n + i] = (float)seeds_pts[b][i].th;
            }
        }
        std::string in_bin = "/tmp/cpp_cuda_in.bin", out_bin = "/tmp/cpp_cuda_out.bin";
        {
            FILE* f = std::fopen(in_bin.c_str(), "wb");
            struct { int32_t B, N, el, fm; uint32_t sd; } sp{batch, n, elastic_L, fast_mode, (uint32_t)(seed * 1009 + n)};
            std::fwrite(&sp, sizeof(sp), 1, f);
            std::fwrite(hx.data(), sizeof(float), bn, f);
            std::fwrite(hy.data(), sizeof(float), bn, f);
            std::fwrite(hth.data(), sizeof(float), bn, f);
            std::fwrite(box_L.data(), sizeof(float), batch, f);
            std::fclose(f);
        }
        double t_g0 = omp_get_wtime();
        std::string run_cmd = cu_bin + " --in " + in_bin + " --out " + out_bin;
        if (std::system(run_cmd.c_str()) != 0) return 2;
        double t_gpu = omp_get_wtime() - t_g0;

        std::vector<double> ox(bn), oy(bn), oth(bn), Lc(batch), lam(batch), pen(batch);
        {
            FILE* f = std::fopen(out_bin.c_str(), "rb");
            if (std::fread(ox.data(), sizeof(double), bn, f) != bn) return 3;
            if (std::fread(oy.data(), sizeof(double), bn, f) != bn) return 3;
            if (std::fread(oth.data(), sizeof(double), bn, f) != bn) return 3;
            if (std::fread(Lc.data(), sizeof(double), batch, f) != (size_t)batch) return 3;
            if (std::fread(lam.data(), sizeof(double), batch, f) != (size_t)batch) return 3;
            if (std::fread(pen.data(), sizeof(double), batch, f) != (size_t)batch) return 3;
            std::fclose(f);
        }

        std::vector<int> order(batch);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) { return Lc[a] < Lc[b]; });

        std::printf("[cuda] n=%d seed=%.2fs gpu=%.2fs | best_gpu_L=%.6f lam=%.6f pen=%.2e | %s\n",
                    n, t_seed, t_gpu, Lc[order[0]], lam[order[0]], pen[order[0]], meta[order[0]].c_str());

        // Select top_k diverse indices (top-2 per prefix + global best)
        std::vector<int> sel;
        std::vector<bool> used(batch, false);
        std::unordered_map<std::string, int> fam_cnt;
        for (int idx : order) {
            auto p = meta[idx].find('(');
            std::string fam = p == std::string::npos ? meta[idx] : meta[idx].substr(0, p);
            if (fam_cnt[fam] < 2) {
                sel.push_back(idx);
                used[idx] = true;
                fam_cnt[fam]++;
            }
        }
        for (int idx : order) {
            if ((int)sel.size() >= top_k) break;
            if (!used[idx]) { sel.push_back(idx); used[idx] = true; }
        }
        if ((int)sel.size() > top_k) sel.resize(top_k);

        int K = (int)sel.size();
        std::vector<double> L_pol(K, 1e18);
        std::vector<std::vector<Sq>> pts_pol(K);

        #pragma omp parallel for schedule(dynamic, 1)
        for (int k = 0; k < K; ++k) {
            int idx = sel[k];
            std::vector<Sq> pts(n);
            for (int i = 0; i < n; ++i) pts[i] = {ox[(size_t)idx * n + i], oy[(size_t)idx * n + i], oth[(size_t)idx * n + i]};
            double L_after = squeeze_candidate_al(pts, true);
            L_pol[k] = L_after;
            pts_pol[k] = std::move(pts);
        }

        int best_k = 0;
        for (int k = 1; k < K; ++k) if (L_pol[k] < L_pol[best_k]) best_k = k;
        int best_b = sel[best_k];
        double total_t = omp_get_wtime() - t0;

        std::string vtag = (idea >= 3) ? ("v" + std::to_string(idea)) : "v2";
        std::string cand_csv = "poc/swarm_inflation/data/cand_" + vtag + "_n" + std::to_string(n) + ".csv";
        double L_csv = 1e18;
        bool csv_ok = write_and_selfcheck_csv(n, pts_pol[best_k], cand_csv, L_csv);
        bool win = csv_ok && (L_csv < inc_L);
        if (win) {
            std::string win_csv = "poc/swarm_inflation/data/win_n" + std::to_string(n) + ".csv";
            double tmp_L = 0.0;
            write_and_selfcheck_csv(n, pts_pol[best_k], win_csv, tmp_L);
        }
        std::printf("=== Result n=%d (%.2fs) | gpu_L=%.6f -> AL_L=%.12f (csv_L=%.12f) vs inc_L=%.12f | win=%s | %s ===\n",
                    n, total_t, Lc[best_b], L_pol[best_k], L_csv, inc_L, win ? "TRUE" : "false", meta[best_b].c_str());

        // Write compact JSON log
        std::ostringstream jf;
        jf << std::setprecision(15);
        jf << "{\n  \"n\": " << n << ",\n  \"elastic_L\": " << elastic_L << ",\n  \"fast_mode\": " << fast_mode
           << ",\n  \"inc_L\": " << inc_L << ",\n  \"L_front\": " << f_L
           << ",\n  \"gpu_best_L\": " << Lc[order[0]] << ",\n  \"best_al_L\": " << L_csv
           << ",\n  \"win\": " << (win ? "true" : "false")
           << ",\n  \"best_meta\": \"" << meta[best_b] << "\",\n  \"elapsed_s\": " << total_t << ",\n  \"top_candidates\": [\n";
        std::vector<int> k_ord(K);
        std::iota(k_ord.begin(), k_ord.end(), 0);
        std::sort(k_ord.begin(), k_ord.end(), [&](int a, int b) { return L_pol[a] < L_pol[b]; });
        if (!export_top_dir.empty()) {
            fs::create_directories(export_top_dir);
            // Export top-16 by AL_L plus up to 16 additional best by GPU L_cert (in case SLP unlocks AL slack)
            std::vector<int> exp_k;
            std::vector<bool> seen_k(K, false);
            for (int r = 0; r < std::min(16, K); ++r) {
                exp_k.push_back(k_ord[r]);
                seen_k[k_ord[r]] = true;
            }
            std::vector<int> k_by_gpu(K);
            std::iota(k_by_gpu.begin(), k_by_gpu.end(), 0);
            std::sort(k_by_gpu.begin(), k_by_gpu.end(), [&](int a, int b) { return Lc[sel[a]] < Lc[sel[b]]; });
            for (int r = 0; r < K && (int)exp_k.size() < 32; ++r) {
                if (!seen_k[k_by_gpu[r]]) { exp_k.push_back(k_by_gpu[r]); seen_k[k_by_gpu[r]] = true; }
            }
            int M_exp = (int)exp_k.size();
            std::string ep = export_top_dir + "/top_n" + std::to_string(n) + "_fm" + std::to_string(fast_mode) + ".bin";
            FILE* fe = std::fopen(ep.c_str(), "wb");
            if (fe) {
                int32_t hdr[2] = {M_exp, n};
                std::fwrite(hdr, sizeof(int32_t), 2, fe);
                for (int kk : exp_k) {
                    const auto& vp = pts_pol[kk];
                    std::vector<double> ax(n), ay(n), ath(n);
                    for (int i = 0; i < n; ++i) { ax[i] = vp[i].x; ay[i] = vp[i].y; ath[i] = vp[i].th; }
                    std::fwrite(ax.data(), sizeof(double), n, fe);
                    std::fwrite(ay.data(), sizeof(double), n, fe);
                    std::fwrite(ath.data(), sizeof(double), n, fe);
                }
                std::fclose(fe);
            }
        }
        for (int r = 0; r < std::min(10, K); ++r) {
            int kk = k_ord[r], bb = sel[kk];
            jf << "    {\"rank\": " << r << ", \"gpu_L\": " << Lc[bb] << ", \"lam\": " << lam[bb]
               << ", \"al_L\": " << L_pol[kk] << ", \"meta\": \"" << meta[bb] << "\"}"
               << (r + 1 < std::min(10, K) ? "," : "") << "\n";
        }
        jf << "  ]\n}\n";
        std::string jtag = (idea >= 3) ? ("v" + std::to_string(idea)) : "v2b";
        std::ofstream jo("poc/swarm_inflation/docs/run_" + jtag + "_n" + std::to_string(n) + "_s" + std::to_string(seed) + "_el" + std::to_string(elastic_L) + "_fm" + std::to_string(fast_mode) + ".json");
        jo << jf.str();
    }
    return 0;
}
