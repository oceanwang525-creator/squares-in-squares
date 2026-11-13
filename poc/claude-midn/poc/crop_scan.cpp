// Exhaustive axis-aligned crop scan.
// For every donor packing n' and every window anchored at (some xmin, some ymin)
// under the 4 reflections, the m squares with the smallest Chebyshev extent from
// the anchor form a valid m-packing of side ext_(m).  Report, for each target m
// in [lo, hi], the smallest such side over all donors n' > m.
// usage: crop_scan ext.bin lo hi threads > best.csv
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>
struct Sq { double x0, x1, y0, y1; };
struct Best { double s = 1e300; int donor = 0, refl = 0; double ax = 0, ay = 0; };
int main(int argc, char** argv) {
  int lo = atoi(argv[2]), hi = atoi(argv[3]), T = atoi(argv[4]);
  std::vector<std::vector<Sq>> P(1001);
  FILE* f = fopen(argv[1], "rb");
  for (int n = 1; n <= 1000; n++) { P[n].resize(n); if (fread(P[n].data(), sizeof(Sq), n, f) != (size_t)n) return 1; }
  fclose(f);
  std::vector<std::vector<Best>> B(T, std::vector<Best>(hi + 1));
  auto work = [&](int t) {
    std::vector<double> xs, ys, ext; std::vector<Sq> q, col;
    for (int d = lo + 1 + t; d <= 1000; d += T) {
      double L = 0; for (auto& s : P[d]) L = std::max({L, s.x1, s.y1});
      for (int r = 0; r < 4; r++) {
        q = P[d];
        for (auto& s : q) {
          if (r & 1) { double a = L - s.x1, b = L - s.x0; s.x0 = a; s.x1 = b; }
          if (r & 2) { double a = L - s.y1, b = L - s.y0; s.y0 = a; s.y1 = b; }
        }
        xs.clear(); ys.clear();
        for (auto& s : q) { xs.push_back(s.x0); ys.push_back(s.y0); }
        auto uniq = [](std::vector<double>& v) {
          std::sort(v.begin(), v.end()); std::vector<double> o;
          for (double x : v) if (o.empty() || x - o.back() > 1e-9) o.push_back(x);
          v.swap(o); };
        uniq(xs); uniq(ys);
        int mmax = std::min(hi, d - 1);
        for (double ax : xs) {
          col.clear(); for (auto& s : q) if (s.x0 >= ax - 1e-12) col.push_back(s);
          if ((int)col.size() < lo) break;
          for (double ay : ys) {
            ext.clear();
            for (auto& s : col) if (s.y0 >= ay - 1e-12) ext.push_back(std::max(s.x1 - ax, s.y1 - ay));
            if ((int)ext.size() < lo) break;
            std::sort(ext.begin(), ext.end());
            int top = std::min<int>(mmax, ext.size());
            for (int m = lo; m <= top; m++)
              if (ext[m - 1] < B[t][m].s) B[t][m] = {ext[m - 1], d, r, ax, ay};
          }
        }
      }
    }
  };
  std::vector<std::thread> th; for (int t = 0; t < T; t++) th.emplace_back(work, t);
  for (auto& x : th) x.join();
  printf("m,side,donor,refl,ax,ay\n");
  for (int m = lo; m <= hi; m++) {
    Best b; for (int t = 0; t < T; t++) if (B[t][m].s < b.s) b = B[t][m];
    printf("%d,%.15f,%d,%d,%.15f,%.15f\n", m, b.s, b.donor, b.refl, b.ax, b.ay);
  }
}
