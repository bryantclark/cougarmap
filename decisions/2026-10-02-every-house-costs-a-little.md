---
name: 2026-10-02-every-house-costs-a-little
date: 2026-10-02
description: Every house within 500 m lowers a spot's score a little, on top of the populated-area cut; single homesteads are no longer free
tags: [model, site-penalties]
---

# Every house within 500 m costs a little

`site_penalty` multiplies the score by (1 + houses within houses_radius_m)^-houses_exponent (0.34) on top of the
populated-area ramp, so one house costs ~21% and 15 ~61%. This reverses the earlier rule that only populated areas
cost anything (homesteads were free because lions use them): within-array SNAPSHOT USA cameras out of region, collar
day beds in all 12 groups and the pin-free human picks all lean the other way (docs/experiments/10-house-cost.md).
Rejected in the same round: a stronger winter exponent (failed its a-priori test on the BC collars nearest NE WA),
refitted combine/site-penalty parameters, a patch CNN, and bench/ruggedness/edge layers.
