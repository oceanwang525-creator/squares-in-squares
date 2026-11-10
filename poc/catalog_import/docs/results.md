# Source and certification record

jlevy source commit f54ad1048080ce14151282dd1861a29b03b51714.
evand source commit bb4cc78e5b35103c1f177db55e9cf08974e66563.

jlevy known-best witnesses cover n=1..324. Evand supplies 26 additional higher sizes, yielding 350 coordinate-bearing sizes. Algebraic conversion errors at n=70 and n=207 are recorded rather than hidden; Evand supplies alternative coordinates for these sizes. Friedman redirects to Kingbird. Direct Kingbird requests were rate limited; Evand JSON supplied the relevant layouts instead.

Four strict improvements against the contact-polish portfolio score 22.20434258797071. Independently checked C++ merged score is 22.20388872691816. Official literal and pandas both pass 1000 accepted, zero missing/rejected. Official literal score 22.203888726918134; tiny last-digit differences reflect summation. Reports are included. Catalog contributions reduce the total score by about 0.00045386105.

The coordinate catalog alone does not establish the leaderboard-leading 22.054 portfolio. Direct coordinates are missing for 650 sizes. A rough nearest-larger/integer-grid envelope produced 23.92129, not a certified full-coordinate portfolio and not a matching 22.054 construction. Further progress needs a per-size target benchmark or new high-n optimization rather than claiming all published bounds have been reconstructed.

The unchanged baseline test suite passes 65 tests. Importer byte-compiles. Independent whole-file checks require all 1,000 sizes accepted and zero missing/rejected, for literal decimal and pandas 3.1.0 transport, using both official Python and independent C++ geometry. C++ also checks floating-point round-trip transport separately.
