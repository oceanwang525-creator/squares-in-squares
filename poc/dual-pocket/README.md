# Dual-pocket certified union

This POC joins the aligned local pool with the upstream team portfolio at
`22.173995906839`. It keeps exactly the rows where the local packing has a
strict official-metric improvement.

Measured result:

- literal official score: `22.173891123046`
- pandas round-trip official score: `22.173891123046`
- accepted puzzles: `1000/1000`
- strict wins: **14**
- strict regressions: **0**
- net score gain: **0.000104783792896**

| n | this `L` | team baseline `L` |
| ---: | ---: | ---: |
| 502 | 22.877637090605 | 22.877638427742 |
| 505 | 22.930439920239 | 22.930439981526 |
| 595 | 24.861685491745 | 24.861925350012 |
| 687 | 26.723271497397 | 26.725053394441 |
| 689 | 26.745616353856 | 26.745616649926 |
| 739 | 27.690919374254 | 27.690921053175 |
| 740 | 27.706455038657 | 27.706646517490 |
| 747 | 27.810663622416 | 27.811200399348 |
| 809 | 28.889722927522 | 28.889724257606 |
| 814 | 28.944429709280 | 28.944430208858 |
| 816 | 28.978163041812 | 28.978163053489 |
| 873 | 29.957160939918 | 29.957161114122 |
| 934 | 30.964143168755 | 30.964145685768 |
| 969 | 31.633982501359 | 31.633982502102 |

The full comparison is in [docs/WINS.md](docs/WINS.md). Both literal-decimal
and actual pandas `read_csv`/`to_csv` reports are checked in under `data/`.

Run from the repository root after following the shared baseline setup:

```sh
python poc/dual-pocket/poc/certify_union.py
```

The dual-pocket construction search itself found no new certified packing.
Its tested placement bounds and negatives are recorded in
[docs/NOTES.md](docs/NOTES.md). No Kaggle submission is performed.
