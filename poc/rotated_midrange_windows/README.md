# Rotated windows for 100 to 200 squares

Rotate each nearby incumbent donor around its origin and translate a square window across it. Try a fixed angular grid and align outer hull edges with the container axes. An event sweep finds the maximum number of whole squares inside the window. For each target n, the window side is 1e-8 below its incumbent. Unlike upright prefix truncation, the window can move in both coordinates after rotation.

See [results](docs/results.md) and [scan data](data/scan.json).
