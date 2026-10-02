---
name: 2026-10-02-destination-approaches-in-travel
date: 2026-10-02
description: The travel line includes the covered approaches from bedding timber to limited water and meadows (travel = max(line, approach)), as pre-registered; measured better on the human picks and neutral elsewhere
tags: [model, travel]
---

# Destination approaches are part of the travel line

`factors.compute_approach` folds the destination approaches into `travel` (travel = max(terrain line, approach);
`factors.destination_approach`, settings `config.Approach`): least-cost routes from bedding cover to limited water
and openings, scored where they converge, under cover, near the destination and downwind of it. It came from the
owner's method (animals travel from beds to water and meadows; a hunting lion takes the covered approach), every
value was fixed before it was measured, and it improved the human picks with nothing else moving
(docs/experiments/14-water-approaches.md). Don't re-tune its numbers on the human picks, and don't add the
corridor to the water factor instead: that form was tried and hurt.
