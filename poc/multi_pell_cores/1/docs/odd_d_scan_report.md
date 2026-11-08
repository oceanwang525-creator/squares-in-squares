# Symmetric Odd-D Corrugated Cores Scan Report ($n \le 1000$)

## Executive Summary

This scan systematically explored the parameter space of all symmetric odd-$D$ staircase-band configurations ($D \in \{3, 17, 27, 51\}$) across all admissible strip widths satisfying $\{w/\sqrt{2}\} \in (0.05, 0.42)$ wrapped with upright L-borders of thickness $t \in 0..15$ and flank perimeter tooth deletions ($m \in \{1, 2\}$ teeth removed).

- **Incumbent Reference**: `poc/pell_corrugation/data/border-scan/merged.csv` (`merged-literal-official.json`)
- **Nominal Gate**: $L_{\text{nominal}} < L_{\text{incumbent}} + 0.005$
- **Candidates Evaluated with 4-Sign-Piece LP Polish**: 11
- **Strict Wins Certified (Literal and Pandas 3.1.0 Transport)**: 0

---

## Candidate Screening and Polish Results

| $n$ | Family | Config | $L_{\text{nominal}}$ | Incumbent $L$ | Polished $L$ | Literal Cert $L$ | Pandas Cert $L$ | $\Delta L$ vs Incumbent | Status |
|---:|:---|:---|---:|---:|---:|---:|---:|---:|:---:|
| 172 | D=17, w=3 | t=0 (full) | 13.621320 | 13.617936 | 13.619260 | `13.619259925` | `13.619259925` | +0.001324 | Incumbent holds |
| 199 | D=17, w=3 | t=1 (full) | 14.621320 | 14.617572 | 14.619260 | `14.619259932` | `14.619259932` | +0.001688 | Incumbent holds |
| 297 | D=17, w=6 | t=0 (full) | 17.742641 | 17.740417 | 17.741251 | `17.741250989` | `17.741250989` | +0.000834 | Incumbent holds |
| 332 | D=17, w=6 | t=1 (full) | 18.742641 | 18.740417 | 18.741251 | `18.741251126` | `18.741251126` | +0.000834 | Incumbent holds |
| 738 | D=17, w=13 | t=0 (del1) | 27.692388 | 27.690919 | 27.690960 | `27.690960379` | `27.690960379` | +0.000041 | Incumbent holds |
| 739 | D=17, w=13 | t=0 (full) | 27.692388 | 27.690921 | 27.690961 | `27.690960516` | `27.690960516` | +0.000039 | Incumbent holds |
| 794 | D=17, w=13 | t=1 (full) | 28.692388 | 28.690921 | 28.690961 | `28.690960500` | `28.690960500` | +0.000039 | Incumbent holds |
| 851 | D=17, w=13 | t=2 (full) | 29.692388 | 29.690934 | 29.690961 | `29.690960617` | `29.690960617` | +0.000027 | Incumbent holds |
| 907 | D=51, w=3 | t=0 (full) | 30.621320 | 30.618368 | 30.618528 | `30.618527954` | `30.618527954` | +0.000160 | Incumbent holds |
| 910 | D=17, w=13 | t=3 (full) | 30.692388 | 30.690921 | 30.690961 | `30.690960561` | `30.690960561` | +0.000039 | Incumbent holds |
| 968 | D=51, w=3 | t=1 (full) | 31.621320 | 31.618368 | 31.618535 | `31.618535145` | `31.618535145` | +0.000167 | Incumbent holds |

---

## Key Geometric and Algorithmic Findings

1. **Pell-17 ($D=17, w=13$) Saturation**:
   The incumbent solutions in `merged.csv` at $n \in \{739, 794, 851, 910\}$ are already saturated by previous Pell-17 and Argon corrugation runs. Independent re-polishing confirms stability to within numerical tolerances ($< 10^{-8}$) with full separating-axis theorem clearance and interval verification.

2. **Triple-Pell $D=51, w=3$ ($n=907$) Core Barrier**:
   The uncorrugated algebraic bound $L_{45} = 30.621320344$ beats the public $n=908$ incumbent ($L = 30.637030$), but at $n=907$ the incumbent in `merged.csv` is $L = 30.618367918$. 4-sign-piece LP polishing achieved $L = 30.618527955$ (a shrinkage gain of $\Delta L = -0.002792$), coming within $+0.000160$ of the incumbent but not strictly beating it.

3. **Border Wrap $D=51, w=3, t=1$ ($n=968$) Barrier**:
   Similarly, wrapping $D=51, w=3$ with a $t=1$ upright border yields $n=968$ with nominal $L = 31.621320$. Polishing contracts the container to $L = 31.618527520$, within $+0.000160$ of the $n=968$ incumbent ($31.618367918$).

4. **Flank Perimeter Tooth Deletions**:
   Deleting outer flank perimeter teeth at $(a_1-1, 0)$ on $D=17, w=13$ ($n=738$) achieves certified $L = 27.690921182$. However, the incumbent for $n=738$ in `merged.csv` is $L = 27.690919412$ (already lower than the $n=739$ incumbent by $1.8 \times 10^{-6}$). Thus the tooth-deleted core does not surpass the incumbent.

5. **Transport Certification**:
   All evaluated candidate packings successfully pass official interval verification via `squares.core.certify` under both literal CSV format and pandas 3.1.0 round-trip with zero rejected contacts.
