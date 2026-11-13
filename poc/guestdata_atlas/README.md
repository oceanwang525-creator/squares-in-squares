# GuestData atlas

## Source

This POC imports the complete atlas-based packing portfolio supplied by
GuestData in `final_atlas copy.csv` (SHA-256
`8692ba13635efda4eeae936e9e7ea8daaf9971fd55301042e02f9467d6200570`).

## Result

Compared with upstream `merge` artifact
`merged-22.17370485852.csv`, 163 complete `n` groups are strictly smaller in
both the unchanged official literal check and the pandas `read_csv`/`to_csv`
transport. `final.csv` contains only those groups.

The certified per-`n` union score is `22.159890710933333668...` in literal mode
and `22.159890710933333781...` after pandas transport, a gain of
`0.013814147591580...` from the merge incumbent. Two raw float wins
(`n=852` and `n=920`) were excluded because they did not survive the unchanged
official certification.
