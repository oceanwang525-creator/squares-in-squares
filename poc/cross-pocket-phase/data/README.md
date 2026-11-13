# Data

The `scan-*.json` and `refine-*.json` files contain compact per-task records
without square coordinates. They preserve the source components, dihedral
indices, independent phases, upright offset and exact refill count.

The `probe-*.csv` files are best negative candidates, not submissions:

- `971 x 908`: 959/971
- `908 x 908` at target 971: 955/971
- `973 x 971`: 964/971
- `972 x 971`: 964/971
- `908 x 908` at target 791: 780/791
- `973 x 973` at target 972: 965/972
- `972 x 972` at target 972: 965/972

No `final.csv` exists because none of these reaches its target group.
