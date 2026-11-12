# Dual-pocket certified union

This POC joins the aligned local pool with the upstream team portfolio at
`22.173824282607`. It keeps exactly the rows where the local packing has a
strict official-metric improvement.

Measured result:

- literal official score: `22.173718865391`
- pandas round-trip official score: `22.173718865391`
- accepted puzzles: `1000/1000`
- strict wins: **23**
- strict regressions: **0**
- net score gain: **0.000105417215710**

| n | this `L` | team baseline `L` |
| ---: | ---: | ---: |
| 502 | 22.877637090605 | 22.877638427742 |
| 505 | 22.930439920239 | 22.930439981526 |
| 595 | 24.861685491745 | 24.861925350012 |
| 687 | 26.723271497397 | 26.725053394441 |
| 689 | 26.745616353856 | 26.745616649926 |
| 738 | 27.678647835553 | 27.678650146381 |
| 739 | 27.690919374254 | 27.690921053175 |
| 740 | 27.706455038657 | 27.706646517490 |
| 747 | 27.810663622416 | 27.811200399348 |
| 809 | 28.889722927522 | 28.889724257606 |
| 814 | 28.944429709280 | 28.944430208858 |
| 816 | 28.978163041812 | 28.978163053489 |
| 864 | 29.851918337290 | 29.851920062700 |
| 873 | 29.957160939918 | 29.957161114122 |
| 875 | 29.988731548874 | 29.988732796344 |
| 934 | 30.964143168755 | 30.964145685768 |
| 936 | 30.997240112561 | 30.997253040140 |
| 963 | 31.535533905933 | 31.535533905933 |
| 969 | 31.633982501359 | 31.633982502102 |
| 978 | 31.759851006258 | 31.759851270820 |
| 981 | 31.796031613665 | 31.796031627220 |
| 989 | 31.884737915963 | 31.884738373902 |
| 997 | 31.973617043822 | 31.973617098349 |

The full comparison is in [docs/WINS.md](docs/WINS.md). Both literal-decimal
and actual pandas `read_csv`/`to_csv` reports are checked in under `data/`.

Run from the repository root after following the shared baseline setup:

```sh
python poc/dual-pocket/poc/certify_union.py
```

The dual-pocket construction search itself found no new certified packing.
Its tested placement bounds and negatives are recorded in
[docs/NOTES.md](docs/NOTES.md). No Kaggle submission is performed.
