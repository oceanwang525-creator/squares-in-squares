# Results against upstream 2c1e6a6

The incumbent is `poc/pell_corrugation/data/border-scan/merged.csv` with SHA-256
`d08670d4e517548f72ac10dc61fba5274028efacc9b305bffaf1cebc4f49557f`.
Both official validation paths scored it independently at 22.17371545003316.

The search tried decimal shrink and centering on 52 non-grid sizes. It found
35 strict side wins in both transports. All gains come from removing part of
the original rounding clearance, and the largest side change is about
1.09e-13 (n=199). The rounded official score changes by about 6.75e-14.

| Official validation | Baseline | Candidate | Accepted | Regressions |
| --- | ---: | ---: | ---: | ---: |
| Literal CSV | 22.17371545003316 | 22.173715450033093 | 1000/1000 | 0 |
| pandas 3.1.0 CSV round trip | 22.17371545003316 | 22.173715450033093 | 1000/1000 | 0 |

Improved n: 101, 102, 103, 104, 105, 106, 107, 109, 110, 122, 125, 126, 128,
129, 130, 131, 132, 145, 146, 147, 148, 154, 156, 157, 170, 172, 175, 176,
179, 180, 181, 182, 197, 199, 200.

The full candidate SHA-256 is
`7fde6cadc3c38f32cb3205137858982f219adfd7004dbf6a3c3768a918380b44`.
Only the changed rows are checked in. Both literal and transported inputs
for all 965 other puzzles were checked for exact equality against the
certified incumbent. [Per-size verification](../data/certification.json)
records old and new side lengths for both transports.

The separate whole-layout rotation, diamond pocket refill, and local and
donor relaxation searches produced no structural win. These finite searches
do not establish that any packing is optimal.
