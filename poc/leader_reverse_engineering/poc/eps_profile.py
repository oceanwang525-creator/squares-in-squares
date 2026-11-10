"""Print eps(k,j)=L(n)-k for n=k^2+j from an official-score JSON (default: incumbent)."""
import json, sys, math
p = sys.argv[1] if len(sys.argv) > 1 else "poc/pell_corrugation/data/border-scan/merged-literal-official.json"
L = {r["n"]: float(r["L"]) for r in json.load(open(p))["results"]}
ks = range(8, 32, 1)
print("j\\k " + " ".join(f"{k:6d}" for k in ks))
for j in range(1, 16):
    print(f"{j:3d} " + " ".join(f"{L[k*k+j]-k:6.3f}" if k*k+j in L else "     -" for k in ks))
# waste fraction W/L^2 where W=L^2-n, excess score per n
print("\nscore mass by j-bucket (k>=20, j<=k):")
for lo, hi in [(1, 1), (2, 5), (6, 10), (11, 15), (16, 31)]:
    s = sum(L[k*k+j]/math.sqrt(k*k+j)-1 for k in range(20, 32) for j in range(lo, hi+1) if j <= k and k*k+j <= 1000)
    c = sum(1 for k in range(20, 32) for j in range(lo, hi+1) if j <= k and k*k+j <= 1000)
    print(lo, hi, c, round(s, 4), round(s/c, 5))
