# Leader reverse engineering

Outcome: negative. The per-n data of the top teams could not be obtained, so no family was identified or built. No packing, no measured L, no score gain.

## Sources tried
- Kaggle discussion 751633, the Kaggle leaderboard (WebFetch and the mcp fetch tool): both return only the page title. The pages are JS-rendered or need a login.
- Erich Friedman's Packing Center: the page is only a redirect to David Ellsworth's `kingbird.myphotos.cc/packing/squares_in_squares.html`. That host reset every connection (curl, WebFetch, mcp fetch).
- WebSearch found only Ellsworth catalog subpages (`__n^2-n-1`, `__Göbel_strips`, `__compared`) and a mirror. All were unfetchable. It found no Kaggle writeup.

## What the repo's own data says
- `poc/gap_reachability/README.md`: the incumbent is at or below the `baseline-official` per-n table at all 1000 n, so the "public table" has nothing left to copy. The leaders' 22.054 envelope is not in the repo per n.
- The only public-leader packings in the repo are n=908, 910 and 972 (`poc/boundary_zipper/data/baseline-targets.csv`). Current merged.csv already beats or matches them: 908 is 30.63703 (equal), 971 is 31.67878 (equal), 972 is 31.68741 (better than 31.69239).
- Score mass is spread across every k band: 3.6 to 3.7 per five k values for k in 6..30 (`docs/eps_profile.txt`). The mean excess per n is about 0.0235 for n with (k²-n)/k in 1..2 and 0.0056 for 0..0.5. Beating the leaders by 0.15 needs a small relative gain on hundreds of n. It does not need a few rescued n.

## Hypothesis (unverified): the k²+j, small-j rows
`poc/eps_profile.py` prints eps(k,j) = L(k²+j) - k from the incumbent (`docs/eps_profile.txt`).
- j=1 is stuck at 0.5355 for k=9..24 and 0.5208 for k=25..31. j=2..3 sit on the same 0.5355 plateau.
- A flat eps while k grows is suspicious. The Roth-Vaughan lower bound on waste, W of order sqrt(L*eps), allows eps to shrink with k. The Erdős-Graham trapezoid constructions waste O(L^(7/11)). Our 45° band at k=30 wastes about 2*k*eps, roughly 32 area units.
- If the leaders' gap is real, the likeliest place is a non-45° construction (Erdős-Graham type, or a shallow-angle strip) for k²+1..k²+5 at k >= 20. That is 60 n with an average r of 0.0198 and total mass 1.2.
- Rough scale: lowering eps by 0.02 at n around 800 is worth 0.0007 per n, or about 0.04 over those 60 n. This is arithmetic on assumed numbers, not evidence.

## Honest negatives
- I did not check the hypothesis against any leader's packing. It is consistent with the `eps_profile.txt` plateau and nothing else.
- No cheap new family was built. A shallow-angle strip or Erdős-Graham block for k²+1 needs an LP or relaxation pipeline beyond a 30 minute POC.
- Realistic gain from this directory alone: 0. The hypothesis needs a separate construction effort.

## Next step if someone has Kaggle access
Export the top teams' submissions or the discussion's per-n table, subtract the incumbent L per n, and test whether the positive gaps cluster at k²+1..k²+5 (the hypothesis) or spread over (k²-n)/k in 1..2.

Files: `poc/eps_profile.py` (run from the repo root, `python3 -I`), `docs/eps_profile.txt`.
