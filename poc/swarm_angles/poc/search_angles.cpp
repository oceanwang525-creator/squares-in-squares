// Fast OpenMP C++ search for Ideas 7B & 8 with precomputed 4-corner grid + exact Degree-1/B&B MIS + SAT polish.
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

inline bool sat_clear_pair(const PreSq& a, const PreSq& b, double min_gap) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    double hb = a.hx + b.hx + min_gap;
    if (std::abs(dx) >= hb || std::abs(dy) >= hb) return true;
    double rad = 0.5 + 0.5 * (std::abs(a.c * b.c + a.s * b.s) + std::abs(a.c * b.s - a.s * b.c)) + min_gap;
    if (std::abs(dx * a.c + dy * a.s) >= rad) return true;
    if (std::abs(-dx * a.s + dy * a.c) >= rad) return true;
    if (std::abs(dx * b.c + dy * b.s) >= rad) return true;
    if (std::abs(-dx * b.s + dy * b.c) >= rad) return true;
    return false;
}

std::pair<double, double> sat_max_overlap_and_bbox(const std::vector<Square>& sq) {
    int n = (int)sq.size();
    std::vector<PreSq> p(n);
    double minx = 1e18, maxx = -1e18, miny = 1e18, maxy = -1e18;
    for (int i = 0; i < n; ++i) {
        p[i] = make_pre(sq[i].x, sq[i].y, sq[i].th);
        minx = std::min(minx, p[i].x - p[i].hx);
        maxx = std::max(maxx, p[i].x + p[i].hx);
        miny = std::min(miny, p[i].y - p[i].hx);
        maxy = std::max(maxy, p[i].y + p[i].hx);
    }
    double max_pen = 0.0;
    for (int i = 0; i < n; ++i) {
        const auto& a = p[i];
        for (int j = i + 1; j < n; ++j) {
            const auto& b = p[j];
            double dx = b.x - a.x;
            if (std::abs(dx) >= a.hx + b.hx) continue;
            double dy = b.y - a.y;
            if (std::abs(dy) >= a.hx + b.hx) continue;
            double rad = 0.5 + 0.5 * (std::abs(a.c * b.c + a.s * b.s) + std::abs(a.c * b.s - a.s * b.c));
            double pen = std::min({
                rad - std::abs(dx * a.c + dy * a.s),
                rad - std::abs(-dx * a.s + dy * a.c),
                rad - std::abs(dx * b.c + dy * b.s),
                rad - std::abs(-dx * b.s + dy * b.c),
            });
            if (pen > max_pen) max_pen = pen;
        }
    }
    return {max_pen, std::max(maxx - minx, maxy - miny)};
}

double sat_energy_grad(
    const std::vector<double>& x,
    const std::vector<double>& y,
    const std::vector<double>& th,
    double L,
    double w_wall,
    double w_pair,
    std::vector<double>& gx,
    std::vector<double>& gy,
    std::vector<double>& gth
) {
    int n = (int)x.size();
    std::fill(gx.begin(), gx.end(), 0.0);
    std::fill(gy.begin(), gy.end(), 0.0);
    std::fill(gth.begin(), gth.end(), 0.0);
    std::vector<double> c(n), s(n), hx(n);
    for (int i = 0; i < n; ++i) {
        c[i] = std::cos(th[i]);
        s[i] = std::sin(th[i]);
        hx[i] = 0.5 * (std::abs(c[i]) + std::abs(s[i]));
    }
    double energy = 0.0;
    for (int i = 0; i < n; ++i) {
        double sgnc = (c[i] >= 0.0) ? 1.0 : -1.0;
        double sgns = (s[i] >= 0.0) ? 1.0 : -1.0;
        double dhx = 0.5 * (-sgnc * s[i] + sgns * c[i]);
        double v1 = hx[i] - x[i];
        if (v1 > 0.0) { energy += w_wall * v1 * v1; gx[i] -= 2.0 * w_wall * v1; gth[i] += 2.0 * w_wall * v1 * dhx; }
        double v2 = x[i] + hx[i] - L;
        if (v2 > 0.0) { energy += w_wall * v2 * v2; gx[i] += 2.0 * w_wall * v2; gth[i] += 2.0 * w_wall * v2 * dhx; }
        double v3 = hx[i] - y[i];
        if (v3 > 0.0) { energy += w_wall * v3 * v3; gy[i] -= 2.0 * w_wall * v3; gth[i] += 2.0 * w_wall * v3 * dhx; }
        double v4 = y[i] + hx[i] - L;
        if (v4 > 0.0) { energy += w_wall * v4 * v4; gy[i] += 2.0 * w_wall * v4; gth[i] += 2.0 * w_wall * v4 * dhx; }
    }
    for (int i = 0; i < n; ++i) {
        double xi = x[i], yi = y[i], ci = c[i], si = s[i], hxi = hx[i];
        for (int j = i + 1; j < n; ++j) {
            double dx = x[j] - xi;
            double bound = hxi + hx[j];
            if (std::abs(dx) >= bound) continue;
            double dy = y[j] - yi;
            if (std::abs(dy) >= bound) continue;
            double cj = c[j], sj = s[j];
            double dot_cc = ci * cj + si * sj;
            double cross_cs = ci * sj - si * cj;
            double rad = 0.5 + 0.5 * (std::abs(dot_cc) + std::abs(cross_cs));
            double sc1 = (dot_cc >= 0.0) ? 1.0 : -1.0;
            double sc2 = (cross_cs >= 0.0) ? 1.0 : -1.0;
            double p[4] = {dx * ci + dy * si, -dx * si + dy * ci, dx * cj + dy * sj, -dx * sj + dy * cj};
            double m = rad - std::abs(p[0]);
            int best_k = 0;
            for (int k = 1; k < 4; ++k) {
                double mk = rad - std::abs(p[k]);
                if (mk < m) { m = mk; best_k = k; }
            }
            if (m <= 0.0) continue;
            energy += w_pair * m * m;
            double w2 = 2.0 * w_pair * m;
            double drad_dti = 0.5 * (sc1 * cross_cs - sc2 * dot_cc);
            double drad_dtj = -drad_dti;
            double sgnp = (p[best_k] >= 0.0) ? 1.0 : -1.0;
            double dpk_dxj = 0, dpk_dyj = 0, dpk_dti = 0, dpk_dtj = 0;
            if (best_k == 0) { dpk_dxj = ci; dpk_dyj = si; dpk_dti = -dx * si + dy * ci; }
            else if (best_k == 1) { dpk_dxj = -si; dpk_dyj = ci; dpk_dti = -dx * ci - dy * si; }
            else if (best_k == 2) { dpk_dxj = cj; dpk_dyj = sj; dpk_dtj = -dx * sj + dy * cj; }
            else { dpk_dxj = -sj; dpk_dyj = cj; dpk_dtj = -dx * cj - dy * sj; }
            double dm_dxj = -sgnp * dpk_dxj, dm_dyj = -sgnp * dpk_dyj;
            double dm_dti = drad_dti - sgnp * dpk_dti, dm_dtj = drad_dtj - sgnp * dpk_dtj;
            gx[i] -= w2 * dm_dxj; gy[i] -= w2 * dm_dyj;
            gx[j] += w2 * dm_dxj; gy[j] += w2 * dm_dyj;
            gth[i] += w2 * dm_dti; gth[j] += w2 * dm_dtj;
        }
    }
    return energy;
}

// Precomputed 4-corner upright grid for fixed (L, min_gap)
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
        for (int m = 0; m < N_ORD; ++m) {
            fast_ord[m].resize(N);
            std::iota(fast_ord[m].begin(), fast_ord[m].end(), 0);
            std::sort(fast_ord[m].begin(), fast_ord[m].end(), [&](int a, int b) {
                int ka = 0, kb = 0;
                if (m == 0) { ka = cheb[a] * 100 + d[a]; kb = cheb[b] * 100 + d[b]; }
                else if (m == 1) { ka = d[a] * 100 + cheb[a]; kb = d[b] * 100 + cheb[b]; }
                else if (m == 2) { ka = std::max(ci[a] * 10, cj[a] * 7); kb = std::max(ci[b] * 10, cj[b] * 7); }
                else if (m == 3) { ka = std::max(ci[a] * 7, cj[a] * 10); kb = std::max(ci[b] * 7, cj[b] * 10); }
                else if (m == 4) { ka = cheb[a] + ((corner[a] == 1 || corner[a] == 2) ? 1000 : 0); kb = cheb[b] + ((corner[b] == 1 || corner[b] == 2) ? 1000 : 0); }
                else if (m == 5) { ka = cheb[a] + ((corner[a] == 0 || corner[a] == 3) ? 1000 : 0); kb = cheb[b] + ((corner[b] == 0 || corner[b] == 3) ? 1000 : 0); }
                else if (m == 6) { ka = d[a] + ((corner[a] == 1 || corner[a] == 2) ? 1000 : 0); kb = d[b] + ((corner[b] == 1 || corner[b] == 2) ? 1000 : 0); }
                else { ka = d[a] + ((corner[a] == 0 || corner[a] == 3) ? 1000 : 0); kb = d[b] + ((corner[b] == 0 || corner[b] == 3) ? 1000 : 0); }
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

    // Exact Degree-0/1 reduction + Structured + ILS (1-2 + local kick) + Exact B&B (M <= 48)
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
            for (int w : adj[v]) {
                if (!removed[w]) {
                    --deg[w];
                    if (deg[w] <= 1) q.push_back(w);
                }
            }
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
            std::vector<int> comp;
            std::vector<int> bfs = {root};
            vis[root] = 1;
            for (size_t bi = 0; bi < bfs.size(); ++bi) {
                int u = bfs[bi];
                comp.push_back(u);
                for (int v : adj[u]) if (!removed[v] && !vis[v]) { vis[v] = 1; bfs.push_back(v); }
            }
            int M = (int)comp.size();
            std::vector<int> local_id(N, -1);
            for (int i = 0; i < M; ++i) local_id[comp[i]] = i;
            std::vector<std::vector<int>> ladj(M);
            for (int i = 0; i < M; ++i) {
                for (int v : adj[comp[i]]) if (!removed[v]) ladj[i].push_back(local_id[v]);
            }

            auto local_climb = [&](std::vector<int>& sel, std::vector<int>& ccnt, int& cnt) {
                bool imp = true;
                while (imp) {
                    imp = false;
                    for (int u = 0; u < M; ++u) {
                        if (!sel[u] && ccnt[u] == 0) {
                            sel[u] = 1; ++cnt; for (int w : ladj[u]) ++ccnt[w]; imp = true;
                        }
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
            std::mt19937 rng(42);
            for (int trial = 0; trial < 36; ++trial) {
                if (trial < 8) {
                    std::sort(ord.begin(), ord.end(), [&](int a, int b) {
                        int ga = comp[a], gb = comp[b], ka = 0, kb = 0;
                        if (trial == 0) { ka = cheb[ga]*100 + d[ga]; kb = cheb[gb]*100 + d[gb]; }
                        else if (trial == 1) { ka = d[ga]*100 + cheb[ga]; kb = d[gb]*100 + cheb[gb]; }
                        else if (trial == 2) { ka = std::max(ci[ga]*10, cj[ga]*7); kb = std::max(ci[gb]*10, cj[gb]*7); }
                        else if (trial == 3) { ka = std::max(ci[ga]*7, cj[ga]*10); kb = std::max(ci[gb]*7, cj[gb]*10); }
                        else if (trial == 4) { ka = cheb[ga] + ((corner[ga]==0||corner[ga]==3)?0:1000); kb = cheb[gb] + ((corner[gb]==0||corner[gb]==3)?0:1000); }
                        else if (trial == 5) { ka = cheb[ga] + ((corner[ga]==1||corner[ga]==2)?0:1000); kb = cheb[gb] + ((corner[gb]==1||corner[gb]==2)?0:1000); }
                        else if (trial == 6) { ka = d[ga] + ((corner[ga]==0||corner[ga]==3)?0:1000); kb = d[gb] + ((corner[gb]==0||corner[gb]==3)?0:1000); }
                        else { ka = d[ga] + ((corner[ga]==1||corner[ga]==2)?0:1000); kb = d[gb] + ((corner[gb]==1||corner[gb]==2)?0:1000); }
                        if (ka != kb) return ka < kb;
                        return corner[ga] < corner[gb];
                    });
                } else if (trial < 12) {
                    int cp = trial - 8;
                    std::sort(ord.begin(), ord.end(), [&](int a, int b) {
                        int ca = (corner[comp[a]] == cp) ? 0 : 1;
                        int cb = (corner[comp[b]] == cp) ? 0 : 1;
                        if (ca != cb) return ca < cb;
                        return cheb[comp[a]] < cheb[comp[b]];
                    });
                } else if (trial == 12) {
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

            // Local-neighborhood ILS kicks from best_c (plateau 1-1 shifts + 2-3 escape)
            if (M > 8) {
                for (int kick = 0; kick < 80; ++kick) {
                    std::vector<int> sel = best_c, ccnt = best_ccnt;
                    int cnt = best_sz;
                    std::vector<int> s_nodes;
                    for (int u = 0; u < M; ++u) if (sel[u]) s_nodes.push_back(u);
                    if (s_nodes.empty()) break;
                    int pivot = s_nodes[rng() % s_nodes.size()];
                    // Remove pivot + 1..3 selected neighbors-of-neighbors (or perform 1-1 swap first)
                    std::vector<int> pool;
                    for (int v : ladj[pivot]) for (int w : ladj[v]) if (sel[w]) pool.push_back(w);
                    if (pool.empty()) pool.push_back(pivot);
                    std::shuffle(pool.begin(), pool.end(), rng);
                    int ndrop = std::min((int)pool.size(), 1 + (int)(rng() % 3));
                    for (int d_i = 0; d_i < ndrop; ++d_i) {
                        int r = pool[d_i];
                        if (sel[r]) { sel[r] = 0; --cnt; for (int w : ladj[r]) --ccnt[w]; }
                    }
                    // Also randomly perturb 1 adjacent free vertex in pivot's neighborhood
                    std::vector<int> cand_add;
                    for (int u = 0; u < M; ++u) if (!sel[u] && ccnt[u] == 0) cand_add.push_back(u);
                    std::shuffle(cand_add.begin(), cand_add.end(), rng);
                    for (int u : cand_add) if (ccnt[u] == 0) { sel[u] = 1; ++cnt; for (int w : ladj[u]) ++ccnt[w]; }
                    local_climb(sel, ccnt, cnt);
                    if (cnt > best_sz) { best_sz = cnt; best_c = std::move(sel); best_ccnt = std::move(ccnt); }
                }
            }

            // Step 2: Exact B&B for small/medium components (M <= 48), using ILS best_sz as warm cutoff!
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
                        int u = __builtin_ctzll(tmp);
                        tmp &= tmp - 1;
                        uint64_t nb = (mask_nbr[u] & rem) ^ (1ULL << u);
                        int dg = __builtin_popcountll(nb);
                        if (dg == 0) { self(self, rem ^ (1ULL << u), cur | (1ULL << u), cur_sz + 1); return; }
                        if (dg == 1) { self(self, rem & ~mask_nbr[u], cur | (1ULL << u), cur_sz + 1); return; }
                    }
                    int best_u = -1, max_d = -1;
                    tmp = rem;
                    while (tmp) {
                        int u = __builtin_ctzll(tmp);
                        tmp &= tmp - 1;
                        int dg = __builtin_popcountll(mask_nbr[u] & rem);
                        if (dg > max_d) { max_d = dg; best_u = u; }
                    }
                    self(self, rem & ~mask_nbr[best_u], cur | (1ULL << best_u), cur_sz + 1);
                    self(self, rem ^ (1ULL << best_u), cur, cur_sz);
                };
                dfs(dfs, all_mask, 0ULL, 0);
                if (found_better) {
                    for (int i = 0; i < M; ++i) best_c[i] = (best_mask >> i) & 1ULL;
                }
            }
            for (int i = 0; i < M; ++i) if (best_c[i]) selected[comp[i]] = 1;
        }

        std::vector<Square> out;
        for (int i = 0; i < N; ++i) if (selected[i]) out.push_back({x[i], y[i], 0.0});
        return out;
    }
};

double polish_and_shrink(std::vector<Square>& sq, double L_start, int n_iters = 350) {
    int n = (int)sq.size();
    std::vector<double> x(n), y(n), th(n), gx(n), gy(n), gth(n);
    std::vector<double> mx(n, 0), vx(n, 0), my(n, 0), vy(n, 0), mt(n, 0), vt(n, 0);
    for (int i = 0; i < n; ++i) { x[i] = sq[i].x; y[i] = sq[i].y; th[i] = sq[i].th; }

    auto run_adam = [&](double L, int iters, double lr) {
        std::fill(mx.begin(), mx.end(), 0.0); std::fill(vx.begin(), vx.end(), 0.0);
        std::fill(my.begin(), my.end(), 0.0); std::fill(vy.begin(), vy.end(), 0.0);
        std::fill(mt.begin(), mt.end(), 0.0); std::fill(vt.begin(), vt.end(), 0.0);
        for (int t = 1; t <= iters; ++t) {
            double e = sat_energy_grad(x, y, th, L, 1.0, 1.0, gx, gy, gth);
            if (e < 1e-24) break;
            double b1 = 0.9, b2 = 0.999, eps = 1e-8;
            double bc1 = 1.0 - std::pow(b1, t), bc2 = 1.0 - std::pow(b2, t);
            double cur_lr = lr * (0.2 + 0.8 * (1.0 - double(t) / iters));
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
    run_adam(cur_L, n_iters, 1e-3);
    std::vector<Square> cur(n);
    for (int i = 0; i < n; ++i) cur[i] = {x[i], y[i], th[i]};
    auto [pen, boxL] = sat_max_overlap_and_bbox(cur);
    if (pen > 1e-11) return 1e18;
    for (double step : {1e-3, 3e-4, 1e-4, 3e-5, 1e-5}) {
        for (int tries = 0; tries < 8; ++tries) {
            double trial_L = cur_L - step;
            auto xb = x, yb = y, tb = th;
            run_adam(trial_L, 250, step * 0.5 + 1e-4);
            for (int i = 0; i < n; ++i) cur[i] = {x[i], y[i], th[i]};
            auto [p2, b2] = sat_max_overlap_and_bbox(cur);
            if (p2 <= 1e-11 && b2 <= trial_L + 1e-11) { cur_L = b2; sq = cur; }
            else { x = xb; y = yb; th = tb; break; }
        }
    }
    return cur_L;
}

void build_nw_se(double L, int a1, int a2, std::vector<PreSq>& out_pre) {
    out_pre.clear();
    for (int i = 0; i < a1; ++i)
        for (int j = 0; j < a1 - i; ++j)
            out_pre.push_back({L - 0.5 - i, 0.5 + j, 1.0, 0.0, 0.5});
    for (int i = 0; i < a2; ++i)
        for (int j = 0; j < a2 - i; ++j)
            out_pre.push_back({0.5 + i, L - 0.5 - j, 1.0, 0.0, 0.5});
}

std::vector<Square> pack_jogging_band(
    double L,
    const std::vector<PreSq>& nwse,
    int w_max,
    double th,
    double shear,
    double u_lo,
    double u_hi,
    double v_lo,
    double v_hi,
    int v_mode,
    bool allow_taper,
    double end_bend_deg
) {
    double inv_sq2 = 1.0 / std::sqrt(2.0);
    std::vector<PreSq> placed;
    std::vector<Square> res;
    int w_min = allow_taper ? 1 : w_max;

    auto col_fits = [&](double cx, double cy, int w, double cb, double sb, double hxb, double vxb, double vyb) -> bool {
        double hw = 0.5 * (w - 1);
        for (int m = 0; m < w; ++m) {
            double t = m - hw;
            double qx = cx + t * vxb, qy = cy + t * vyb;
            if (qx - hxb < -1e-4 || qx + hxb > L + 1e-4 || qy - hxb < -1e-4 || qy + hxb > L + 1e-4) return false;
            PreSq q{qx, qy, cb, sb, hxb};
            for (const auto& u : nwse) if (!sat_clear_pair(q, u, -1e-4)) return false;
            for (const auto& p : placed) if (!sat_clear_pair(q, p, -1e-4)) return false;
        }
        return true;
    };

    double dth_opts[3] = {0.0, end_bend_deg * PI / 180.0, -end_bend_deg * PI / 180.0};
    int n_dth = (end_bend_deg > 0.1) ? 3 : 1;
    double u_cur = u_lo;
    int col_idx = 0;
    double v_prev = 0.5 * (v_lo + v_hi);

    while (u_cur <= u_hi && (int)placed.size() + w_max <= 550) {
        bool ok_step = false;
        int w_hi = w_max;
        int w_lo = (col_idx > 0 && u_cur > u_hi - 2.0) ? w_min : w_max;
        double du_max = (col_idx == 0) ? 0.01 : 0.52;
        double vv_step = (v_mode == 2) ? 0.05 : 0.08;
        for (int w = w_hi; w >= w_lo && !ok_step; --w) {
            double hw = 0.5 * (w - 1);
            for (double du = 0.0; du <= du_max && !ok_step; du += 0.04) {
                double u_val = u_cur + du;
                int try_dth = (col_idx == 0 || u_val > u_hi - 2.0) ? n_dth : 1;
                for (int bi = 0; bi < try_dth && !ok_step; ++bi) {
                    double tb = th + dth_opts[bi];
                    double cb = std::cos(tb), sb = std::sin(tb);
                    double hxb = 0.5 * (std::abs(cb) + std::abs(sb));
                    double vxb = -sb + shear * cb, vyb = cb + shear * sb;
                    std::vector<double> vs;
                    for (double vv = v_lo; vv <= v_hi; vv += vv_step) {
                        double cx = (u_val + vv) * inv_sq2, cy = (u_val - vv) * inv_sq2;
                        if (col_fits(cx, cy, w, cb, sb, hxb, vxb, vyb)) vs.push_back(vv);
                    }
                    if (!vs.empty()) {
                        double bv = vs[vs.size() / 2];
                        if (v_mode == 1) bv = (col_idx % 2 == 0) ? vs.front() : vs.back();
                        else if (v_mode == 2) {
                            double tgt = (v_prev + 0.075 <= vs.back()) ? (v_prev + 0.075) : (v_prev - 0.19);
                            double bd = 1e9;
                            for (double vv : vs) {
                                double d = std::abs(vv - tgt);
                                if (d < bd) { bd = d; bv = vv; }
                            }
                        }
                        v_prev = bv;
                        double cx = (u_val + bv) * inv_sq2, cy = (u_val - bv) * inv_sq2;
                        for (int m = 0; m < w; ++m) {
                            double t = m - hw;
                            double qx = cx + t * vxb, qy = cy + t * vyb;
                            placed.push_back({qx, qy, cb, sb, hxb});
                            res.push_back({qx, qy, tb});
                        }
                        ok_step = true;
                        ++col_idx;
                        u_cur = u_val + 0.88;
                    }
                }
            }
        }
        if (!ok_step) u_cur += (col_idx == 0) ? 0.25 : 0.08;
    }
    return res;
}

void save_csv(const std::string& path, int n, const std::vector<Square>& sq) {
    double minx = 1e18, miny = 1e18;
    for (const auto& s : sq) {
        double hx = 0.5 * (std::abs(std::cos(s.th)) + std::abs(std::sin(s.th)));
        minx = std::min(minx, s.x - hx);
        miny = std::min(miny, s.y - hx);
    }
    std::ofstream f(path);
    f << std::setprecision(17) << "id,n,square_id,x,y,angle\n";
    for (int i = 0; i < n; ++i) {
        f << i << "," << n << "," << i << ","
          << (sq[i].x - minx + 1e-12) << "," << (sq[i].y - miny + 1e-12) << "," << sq[i].th << "\n";
    }
}

int main() {
    omp_set_num_threads(7);

    std::map<int, std::vector<Square>> merged;
    {
        std::ifstream f("poc/pell_corrugation/data/border-scan/merged.csv");
        std::string line;
        std::getline(f, line);
        while (std::getline(f, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            std::stringstream ss(line);
            std::string id_s, n_s, sid_s, x_s, y_s, a_s;
            std::getline(ss, id_s, ','); std::getline(ss, n_s, ','); std::getline(ss, sid_s, ',');
            std::getline(ss, x_s, ','); std::getline(ss, y_s, ','); std::getline(ss, a_s, ',');
            merged[std::stoi(n_s)].push_back({std::stod(x_s), std::stod(y_s), std::stod(a_s)});
        }
    }

    // Calibrate Exact MIS on 401, 598, 626
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
        std::cout << "ExactMIS calib n=" << ncal << " L=" << std::fixed << std::setprecision(6) << L
                  << " tilt=" << T.size() << " up=" << up.size() << " tot=" << (T.size() + up.size()) << "\n";
    }

    // Calibrate Idea 7B on n=227
    {
        double L227 = 15.563625652;
        Grid4C g227; g227.init(L227, -1e-4);
        std::vector<PreSq> nwse_p;
        build_nw_se(L227, 13, 12, nwse_p);
        int best_nt = 0, best_tot = 0;
        std::string btag;
        for (double th_d : {-33.88, 33.88, 56.12}) {
            for (double sh : {-0.04, -0.02, 0.0, 0.02, 0.04}) {
                for (int vm : {0, 2}) {
                    for (double u0 : {2.1, 2.5, 2.64, 3.2}) {
                        auto band = pack_jogging_band(L227, nwse_p, 3, th_d * PI / 180.0, sh, u0, std::sqrt(2.0) * L227 - 1.0, -1.6, 1.0, vm, true, 6.0);
                        std::vector<PreSq> T;
                        for (const auto& s : band) T.push_back(make_pre(s.x, s.y, s.th));
                        std::vector<uint8_t> alive; int acnt;
                        g227.compute_alive(T, -1e-4, alive, acnt);
                        auto up = g227.solve_mis_exact(alive);
                        int tot = (int)band.size() + (int)up.size();
                        best_nt = std::max(best_nt, (int)band.size());
                        if (tot > best_tot) {
                            best_tot = tot;
                            std::ostringstream os;
                            os << "th=" << th_d << " sh=" << sh << " vm=" << vm << " u0=" << u0 << " nt=" << band.size() << " up=" << up.size();
                            btag = os.str();
                        }
                    }
                }
            }
        }
        std::cout << "Idea 7B Calib n=227: best_nt=" << best_nt << "/54, best_tot=" << best_tot << "/227 [" << btag << "]\n" << std::flush;
    }

    std::ofstream jout("poc/swarm_angles/docs/cpp_search_7b_8.json");
    jout << "[\n";
    std::vector<int> targets = {401, 442, 530};

    for (size_t ti = 0; ti < targets.size(); ++ti) {
        int n = targets[ti];
        int k = (int)std::floor(std::sqrt((double)(n - 1)));
        auto [pen_inc, L_ours] = sat_max_overlap_and_bbox(merged[n]);
        double L_probe = L_ours - 0.001;
        double half = 0.5 * L_probe;
        double t0 = omp_get_wtime();

        Grid4C grid; grid.init(L_probe, 0.0);

        struct Cand {
            int score;
            std::string tag;
            std::vector<Square> tsq;
        };
        std::vector<std::vector<Cand>> thread_top(7);
        std::vector<std::vector<PreSq>> T_buf(7);
        std::vector<std::vector<uint8_t>> alive_buf(7), blocked_buf(7);

        auto push_cand = [&](int tid, int fast_sc, const std::string& tag, std::vector<Square>&& tsq) {
            auto& v = thread_top[tid];
            if ((int)v.size() < 16 || fast_sc > v.back().score) {
                v.push_back({fast_sc, tag, std::move(tsq)});
                std::sort(v.begin(), v.end(), [](const Cand& a, const Cand& b) { return a.score > b.score; });
                if (v.size() > 16) v.pop_back();
            }
        };

        auto eval_fast = [&](const std::vector<Square>& tsq) -> int {
            int tid = omp_get_thread_num();
            auto& T = T_buf[tid]; T.clear();
            for (const auto& q : tsq) T.push_back(make_pre(q.x, q.y, q.th));
            int acnt;
            grid.compute_alive(T, 0.0, alive_buf[tid], acnt);
            return (int)tsq.size() + grid.fast_greedy_lb(alive_buf[tid], blocked_buf[tid]);
        };

        // 7B jobs
        std::vector<std::tuple<int, int, int, double>> j7b;
        for (int a1 = std::max(6, k - 12); a1 <= k; ++a1) {
            for (int a2 = std::max(4, a1 - 5); a2 <= std::min(k, a1 + 2); ++a2) {
                if (a1 + a2 > 2 * k) continue;
                double W45 = (2.0 * L_probe - a1 - a2 - 2.0) / std::sqrt(2.0);
                if (W45 < 0.7 || W45 > 8.5) continue;
                int wa = std::max(1, (int)std::round(W45));
                for (int w = std::max(1, wa - 1); w <= wa + 1; ++w) {
                    for (double th_d : {18.435, 22.0, 24.5, 26.565, 30.0, 33.88, 36.87, 45.0, 56.12}) {
                        j7b.emplace_back(a1, a2, w, th_d);
                    }
                }
            }
        }

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();

            #pragma omp for schedule(dynamic, 4)
            for (size_t idx = 0; idx < j7b.size(); ++idx) {
                auto [a1, a2, w, th_d] = j7b[idx];
                std::vector<PreSq> nwse_p;
                build_nw_se(L_probe, a1, a2, nwse_p);
                double vm_c = (a2 - a1) / (2.0 * std::sqrt(2.0));
                for (double sh : {0.0, -0.04}) {
                    for (int vm : {0, 2}) {
                        auto band = pack_jogging_band(L_probe, nwse_p, w, th_d * PI / 180.0, sh, 1.5, std::sqrt(2.0) * L_probe - 0.8, vm_c - 1.5, vm_c + 1.5, vm, true, 6.0);
                        if (band.empty()) continue;
                        int sc = eval_fast(band);
                        std::ostringstream os;
                        os << "7B a=(" << a1 << "," << a2 << ") w=" << w << " th=" << th_d << " sh=" << sh << " nt=" << band.size();
                        push_cand(tid, sc, os.str(), std::move(band));
                    }
                }
            }

            // 8A: Wall & L-pocket stepped sec(th)
            std::vector<double> th8 = {14.0, 16.5, 18.435, 22.0, 24.5, 26.565, 30.0, 33.88, 36.87};
            #pragma omp for schedule(dynamic, 1)
            for (size_t ti_th = 0; ti_th < th8.size(); ++ti_th) {
                double th_d = th8[ti_th], th = th_d * PI / 180.0;
                double c = std::cos(th), s = std::sin(th), sec = 1.0 / c, hx = 0.5 * (c + s);
                for (int w : {2, 3, 4, 6, 8, 10}) {
                    for (double xoff : {0.0, 1.0, 2.0, 4.0}) {
                        for (double ybot : {0.0, 4.0, 8.0, 11.0}) {
                            for (double slope : {0.0, 0.35, 0.5, 1.0}) {
                                for (int mode : {0, 1, 2}) {
                                    if (mode == 2 && (xoff > 1.5 || ybot < 4.0)) continue;
                                    std::vector<Square> tsq;
                                    for (int col = 0; col < w; ++col) {
                                        double cx = xoff + hx + col * c;
                                        if (cx + hx > L_probe) break;
                                        double yb = std::max(hx, ybot + slope * col * c);
                                        double yt = std::min(L_probe - hx, L_probe - hx - 0.15 * slope * col * c);
                                        if (yb > yt) continue;
                                        int j0 = (int)std::ceil((yb - col * s) * c - 1e-9);
                                        int j1 = (int)std::floor((yt - col * s) * c + 1e-9);
                                        for (int j = j0; j <= j1; ++j) {
                                            double cy = j * sec + col * s;
                                            if (yb - 1e-6 <= cy && cy <= yt + 1e-6) tsq.push_back({cx, cy, th});
                                        }
                                    }
                                    if (tsq.empty() || (int)tsq.size() >= n) continue;
                                    int base_n = (int)tsq.size();
                                    if (mode == 1) for (int i = 0; i < base_n; ++i) tsq.push_back({L_probe - tsq[i].x, L_probe - tsq[i].y, th});
                                    else if (mode == 2) for (int i = 0; i < base_n; ++i) tsq.push_back({tsq[i].y, tsq[i].x, 0.5 * PI - th});
                                    if (mode > 0) {
                                        auto [p0, l0] = sat_max_overlap_and_bbox(tsq);
                                        if (p0 > 1e-6 || l0 > L_probe + 1e-6) continue;
                                    }
                                    int sc = eval_fast(tsq);
                                    std::ostringstream os;
                                    os << "8A th=" << th_d << " w=" << w << " x=" << xoff << " yb=" << ybot << " sl=" << slope << " m=" << mode << " nt=" << tsq.size();
                                    push_cand(tid, sc, os.str(), std::move(tsq));
                                }
                            }
                        }
                    }
                }
            }

            // 8B (Expanded C4 + Pinched Core) & 8C (908 Octagon NW/SE)
            std::vector<double> th8b = {16.5, 18.435, 22.0, 24.5, 26.565, 30.0, 33.69, 36.87, 41.0, 45.0};
            #pragma omp for schedule(dynamic, 1)
            for (size_t idx = 0; idx < th8b.size(); ++idx) {
                double th_d = th8b[idx], th = th_d * PI / 180.0;
                double c = std::cos(th), s = std::sin(th), hx = 0.5 * (c + s), sec = 1.0 / c;
                for (int m1 = 1; m1 <= 8; ++m1) {
                    for (int m2 = 1; m2 <= std::min(5, m1); ++m2) {
                        double max_reach_x = 0.5 * (m1 - 1) * c + 0.5 * (m2 - 1) * s + hx;
                        double max_reach_y = 0.5 * (m1 - 1) * s + 0.5 * (m2 - 1) * c + hx;
                        for (double rx = 1.5; rx <= half - max_reach_x + 0.1; rx += 1.0) {
                            for (double ry = 0.5; ry <= half - max_reach_y + 0.1; ry += 1.0) {
                                if (rx - max_reach_x < -0.25 && ry - max_reach_y < -0.25) continue;
                                std::vector<Square> tsq;
                                bool in_box = true;
                                for (int i = 0; i < m1 && in_box; ++i) {
                                    double ui = i - 0.5 * (m1 - 1);
                                    for (int j = 0; j < m2 && in_box; ++j) {
                                        double vj = j - 0.5 * (m2 - 1);
                                        double dx0 = rx + ui * c - vj * s, dy0 = ry + ui * s + vj * c;
                                        double pts[4][3] = {
                                            {half + dx0, half + dy0, th}, {half - dy0, half + dx0, th + 0.5 * PI},
                                            {half - dx0, half - dy0, th + PI}, {half + dy0, half - dx0, th + 1.5 * PI}
                                        };
                                        for (int r = 0; r < 4; ++r) {
                                            if (pts[r][0] - hx < -1e-6 || pts[r][0] + hx > L_probe + 1e-6 ||
                                                pts[r][1] - hx < -1e-6 || pts[r][1] + hx > L_probe + 1e-6) { in_box = false; break; }
                                            tsq.push_back({pts[r][0], pts[r][1], pts[r][2]});
                                        }
                                    }
                                }
                                if (!in_box || tsq.empty()) continue;
                                auto [p0, l0] = sat_max_overlap_and_bbox(tsq);
                                if (p0 > 1e-6 || l0 > L_probe + 1e-6) continue;
                                int sc = eval_fast(tsq);
                                std::ostringstream os;
                                os << "8B_C4 m=" << m1 << "x" << m2 << " th=" << th_d << " r=(" << rx << "," << ry << ") nt=" << tsq.size();
                                push_cand(tid, sc, os.str(), std::move(tsq));
                            }
                        }
                    }
                }
                // 8B_Pinched: centered (c_up x c_up) upright core + 4 C4 tilted wall strips (fast O(N_tilt) check)
                for (int c_up = std::max(6, k - 8); c_up <= k - 2; ++c_up) {
                    double core_lo = half - 0.5 * c_up, core_hi = half + 0.5 * c_up;
                    for (int m1 : {1, 2, 3, 4}) {
                        for (int m2 = 3; m2 <= c_up + 1; m2 += 2) {
                            for (double rx : {0.5 * c_up + 0.5 * m1 + 0.15, half - 0.5 * m1 - 0.15}) {
                                if (rx <= 0.5 * c_up) continue;
                                for (double ry : {-1.5, 0.0, 1.5}) {
                                    std::vector<Square> tsq;
                                    bool in_box = true;
                                    for (int i = 0; i < m1 && in_box; ++i) {
                                        double ui = i - 0.5 * (m1 - 1);
                                        for (int j = 0; j < m2 && in_box; ++j) {
                                            double vj = j - 0.5 * (m2 - 1);
                                            double dx0 = rx + ui * c - vj * s, dy0 = ry + ui * s + vj * c;
                                            double pts[4][3] = {
                                                {half + dx0, half + dy0, th}, {half - dy0, half + dx0, th + 0.5 * PI},
                                                {half - dx0, half - dy0, th + PI}, {half + dy0, half - dx0, th + 1.5 * PI}
                                            };
                                            for (int r = 0; r < 4; ++r) {
                                                double px = pts[r][0], py = pts[r][1];
                                                if (px - hx < -1e-6 || px + hx > L_probe + 1e-6 ||
                                                    py - hx < -1e-6 || py + hx > L_probe + 1e-6) { in_box = false; break; }
                                                if (px + hx > core_lo + 1e-6 && px - hx < core_hi - 1e-6 &&
                                                    py + hx > core_lo + 1e-6 && py - hx < core_hi - 1e-6) {
                                                    PreSq tp = make_pre(px, py, pts[r][2]);
                                                    int ix0 = std::max(0, (int)std::floor(px - hx - core_lo));
                                                    int ix1 = std::min(c_up - 1, (int)std::floor(px + hx - core_lo));
                                                    int iy0 = std::max(0, (int)std::floor(py - hx - core_lo));
                                                    int iy1 = std::min(c_up - 1, (int)std::floor(py + hx - core_lo));
                                                    for (int ix = ix0; ix <= ix1 && in_box; ++ix)
                                                        for (int iy = iy0; iy <= iy1; ++iy)
                                                            if (!sat_clear_pair(tp, {core_lo + 0.5 + ix, core_lo + 0.5 + iy, 1.0, 0.0, 0.5}, 0.0)) { in_box = false; break; }
                                                }
                                                if (!in_box) break;
                                                tsq.push_back({px, py, pts[r][2]});
                                            }
                                        }
                                    }
                                    if (!in_box || tsq.empty()) continue;
                                    auto [p0, l0] = sat_max_overlap_and_bbox(tsq);
                                    if (p0 > 1e-6 || l0 > L_probe + 1e-6) continue;
                                    for (int i = 0; i < c_up; ++i)
                                        for (int j = 0; j < c_up; ++j)
                                            tsq.push_back({core_lo + 0.5 + i, core_lo + 0.5 + j, 0.0});
                                    int sc = eval_fast(tsq);
                                    std::ostringstream os;
                                    os << "8B_pinch cup=" << c_up << " m=" << m1 << "x" << m2 << " th=" << th_d << " r=(" << rx << "," << ry << ") nt=" << tsq.size();
                                    push_cand(tid, sc, os.str(), std::move(tsq));
                                }
                            }
                        }
                    }
                }
                // 8C: 908 NW/SE octagon pockets
                for (double rx = 3.0; rx <= half - 2.0; rx += 1.5) {
                    for (double ry = half + 2.0; ry <= L_probe - 3.0; ry += 1.5) {
                        for (int ur : {3, 4, 5}) {
                            for (int vr : {3, 4, 5}) {
                                std::vector<Square> tsq;
                                double d_cut = 0.75 * (ur + vr);
                                bool in_box = true;
                                for (int col = -ur; col <= ur && in_box; ++col) {
                                    double cx = rx + col * c;
                                    for (int row = -vr - 3; row <= vr + 3; ++row) {
                                        double cy = ry + row * sec + col * s;
                                        if (std::abs(col) + std::abs((cy - ry) * c) <= d_cut + 1e-6 && std::abs(cy - ry) <= vr + 0.5) {
                                            if (cx - hx < -1e-6 || cx + hx > L_probe + 1e-6 || cy - hx < -1e-6 || cy + hx > L_probe + 1e-6) { in_box = false; break; }
                                            tsq.push_back({cx, cy, th});
                                            tsq.push_back({L_probe - cx, L_probe - cy, th});
                                        }
                                    }
                                }
                                if (!in_box || tsq.empty() || (int)tsq.size() >= n) continue;
                                auto [p0, l0] = sat_max_overlap_and_bbox(tsq);
                                if (p0 > 1e-6 || l0 > L_probe + 1e-6) continue;
                                int sc = eval_fast(tsq);
                                std::ostringstream os;
                                os << "8C_908oct th=" << th_d << " r=(" << rx << "," << ry << ") uv=(" << ur << "," << vr << ") nt=" << tsq.size();
                                push_cand(tid, sc, os.str(), std::move(tsq));
                            }
                        }
                    }
                }
            }
        }

        // Collect merged top ~35 candidates and evaluate with EXACT Degree-0/1 + B&B MIS
        std::vector<Cand> merged_top;
        for (int t = 0; t < 7; ++t) merged_top.insert(merged_top.end(), thread_top[t].begin(), thread_top[t].end());
        std::sort(merged_top.begin(), merged_top.end(), [](const Cand& a, const Cand& b) { return a.score > b.score; });
        if (merged_top.size() > 35) merged_top.resize(35);

        int best_cap = 0;
        std::string best_tag;
        std::vector<Square> best_full;
        for (const auto& cnd : merged_top) {
            std::vector<PreSq> T;
            for (const auto& q : cnd.tsq) T.push_back(make_pre(q.x, q.y, q.th));
            std::vector<uint8_t> alive; int acnt;
            grid.compute_alive(T, 0.0, alive, acnt);
            auto up = grid.solve_mis_exact(alive);
            int cap = (int)cnd.tsq.size() + (int)up.size();
            if (cap > best_cap) {
                best_cap = cap;
                best_tag = cnd.tag + " exact_up=" + std::to_string(up.size());
                best_full = cnd.tsq;
                best_full.insert(best_full.end(), up.begin(), up.end());
            }
        }

        double polished_L = -1.0;
        if (best_cap >= n) {
            std::vector<Square> cand(best_full.begin(), best_full.begin() + n);
            polished_L = polish_and_shrink(cand, L_probe, 400);
            if (polished_L < L_ours - 1e-7) {
                std::string cpath = "poc/swarm_angles/data/cand_n" + std::to_string(n) + ".csv";
                save_csv(cpath, n, cand);
                std::cout << "  *** CANDIDATE WIN SAVED " << cpath << ": L=" << std::setprecision(9) << polished_L << " vs L_ours=" << L_ours << "\n";
            }
        }

        double dt = omp_get_wtime() - t0;
        std::cout << "n=" << n << " (L_probe=" << std::setprecision(6) << L_probe << "): best_cap=" << best_cap
                  << " (def=" << (n - best_cap) << ") [" << best_tag << "] in " << std::setprecision(2) << dt << "s\n" << std::flush;

        jout << "  {\"n\": " << n << ", \"L_ours\": " << std::setprecision(12) << L_ours
             << ", \"L_probe\": " << L_probe << ", \"best_cap\": " << best_cap
             << ", \"deficit\": " << (n - best_cap) << ", \"tag\": \"" << best_tag
             << "\", \"polished_L\": " << polished_L << ", \"seconds\": " << std::setprecision(2) << dt << "}"
             << (ti + 1 < targets.size() ? "," : "") << "\n";
    }
    jout << "]\n";
    return 0;
}
