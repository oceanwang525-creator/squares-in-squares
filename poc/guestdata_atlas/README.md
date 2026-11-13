# GuestData atlas

## Source

This POC imports the complete atlas-based packing portfolio supplied by
GuestData in `final_atlas copy.csv` (SHA-256
`8692ba13635efda4eeae936e9e7ea8daaf9971fd55301042e02f9467d6200570`).

## Result

Compared with upstream `merge` artifact
`merged-22.17370485852.csv`, 165 complete `n` groups are strictly smaller in
both the unchanged official literal check and the pandas `read_csv`/`to_csv`
transport. `final.csv` contains only those groups. For `n=852` and `n=920`,
angles below `1e-18` degrees were snapped to zero so the canonical file stayed
inside the official parser's literal-length limit.

The certified per-`n` union score is `22.159779301349195347...` in literal mode
and `22.159779301349195460...` after pandas transport, a gain of
`0.013925557175718...` from the merge incumbent.
