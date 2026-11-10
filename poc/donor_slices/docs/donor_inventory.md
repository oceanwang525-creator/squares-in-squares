# Donor inventory

Source: `poc/pell_corrugation/data/border-scan/merged.csv`.

| Item | Value |
|---|---:|
| Groups | 1000 |
| Rows | 500500 |
| Largest donor `n` | 1000 |
| Donor `n=1000` side | 32 |
| Donor `n=1000` squares | 1000 |

The `n=1000` donor is a 32x32 upright grid: every angle is `0.0`, every center
is a half-integer in `[0.5, 31.5]`, and all 1000 centers are distinct. No local
gallery or data file has `n` above 1000, so the donor set stops there.

## Donors scanned per target

| Target `n` | Donor range | Donors scanned |
|---:|---|---:|
| 645 | 646 .. 1000 | 355 |
| 925 | 926 .. 1000 | 75 |
| 987 | 988 .. 1000 | 13 |

The search is deterministic and bounded. It does not mutate donors; it only
selects a subset of already certified squares.

## Selected donors

| Target | Role | Donor `n'` | Crop capacity | Tight crop box |
|---:|---|---:|---:|---|
| 645 | best below incumbent | 697 | 644 | `25.869769561 x 25.869769561` |
| 645 | capacity match | 698 | 645 | `25.881109226 x 25.881109226` |
| 925 | best below incumbent | 986 | 923 | `30.859361960 x 30.859361960` |
| 925 | capacity match | 988 | 925 | `30.875854187 x 30.875854187` |
| 987 | best below incumbent | 999 | 484 | `22 x 22` |
| 987 | capacity match | 988 | 956 | `30.875854187 x 31.875854188` |

The `n=987` below-incumbent row is a crop-plus-grid composite; the other rows
are crops with an empty L-shape grid. The `n=987` capacity match uses a 31-square
grid in the L-shape.
