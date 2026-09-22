# Deployed coverage: what the model scores on the path a user takes

A conformal band is a claim about a population. `models/calibration.json` says
0.90 coverage, measured on 600 held-out samples — but those samples arrived
through the sweep, carrying boundary conditions the sweep itself tagged. A user
arrives with a STEP file and no tags at all.

Nothing measured that second path until 2026-09-21, and it was not close.

    python -m tools.deployed_coverage 100 6

Draws fresh parts, solves each one, exports it to STEP, screens the STEP through
the loaded server, and compares. Roughly twenty minutes for 600 parts on six
workers. It is not a unit test and does not live in `tests/`: it needs a trained
model, a solver, and real time. It is the only thing that measures what a user
actually gets.

**Run it after any change to boundary-condition selection, the feature builder,
the mesher, or the model.** Every family near 0.90 with median pred/true near
1.0, or the change is not done.

---

## 2026-09-21 — the first measurement

| family | coverage | median pred/true |
| --- | --- | --- |
| bracket | 0.46 | 0.72 |
| flange | **0.00** | **4.97** |
| pulley | 0.59 | 0.79 |
| spring | **0.00** | **0.00** |
| stepped_shaft | 0.77 | 0.83 |
| welded_t | 0.67 | 0.81 |

**Overall 0.415 on n=600 against a calibrated 0.90**, standard error 0.012.

An earlier 12-sample run had given bracket 0.92 and flange 0.83. It disagreed
with this, its harness no longer existed, and 600 samples with the method on
disk beat 12 without. The disagreement was never explained and is worth a look
if it ever recurs.

### Cause 1 — the deployed path asked a different question

Every family tags boundary conditions as **faces**, by a predicate on the face
centroid. The deployed path used **node-coordinate** selectors. Measured:

| family | the preset held | the tag held |
| --- | --- | --- |
| spring | 20 of 6144 nodes (0.3%), a tangent sliver of the helical surface | every node of the end-coil face |
| flange | 2946 nodes (26.5%), a slab through the whole plate | only faces whose centroid lies inside the plate thickness |
| bracket | 880 nodes (15.9%) on the x-min face | roughly the same, which is why it drifted rather than broke |

Nothing compared the two selections, so nothing noticed. `face_selectors.py`
restates each preset as the face predicate its tag uses;
`tests/test_face_selectors.py` now holds the two paths to each other.

### Cause 2 — the spring was never a part real CAD could produce

Fixing the selectors moved five families and left spring exactly where it was.
Chasing it found something else entirely.

`gmsh`'s `addPipe` does not consume its profile. The original plane surface
stayed in the model, 0.044 mm from the swept solid's bottom cap, and gmsh meshed
both: 24 nodes on the cap and 25 more on the orphan, **sharing none**. Every
swept spring carried twice the held nodes at its bottom end that it did at its
top. A STEP export drops the orphan.

The model is not at fault, and the evidence is unambiguous — fed sweep-path
spring features it scores **1.02**; fed deployed features on the same part,
**0.006**:

| | truth τ | via sweep mesh | via imported STEP |
| --- | --- | --- | --- |
| spring | 16137 | 16450 (1.02) | 100.6 (0.006) |
| bracket | 3421 | 3846 (1.12) | 3977 (1.16) |

The geometry is fixed in `generate/families/spring.py`. **The shipped weights
are not**, because they were fitted on the broken version, so `spring` is listed
in `predict.UNTRANSFERABLE_FAMILIES` and refused at screening time. Clearing
that entry needs a spring re-sweep and a retrain, not a code change.

---

## 2026-09-22 — after the selector repair

| family | before | after |
| --- | --- | --- |
| bracket | 0.46 | 0.83 |
| flange | 0.00 | 0.96 |
| pulley | 0.59 | 0.94 |
| stepped_shaft | 0.77 | 0.93 |
| welded_t | 0.67 | 0.90 |
| spring | 0.00 | 0.00 → now refused |
| **overall** | **0.415** | **0.760** |

Five of six families sit at or near nominal. The overall figure is held down by
spring, which is refused rather than reported.

Bracket at 0.83 is the one still meaningfully below its 0.90, and it is not
obviously noise — the standard error at n=100 is 0.030, so 0.83 is about two
standard errors low. Worth a look before anyone calls this finished.

---

## What this cost, and the cheap check that would have caught it

Both defects are the same shape: **two code paths that were supposed to agree,
and quietly did not.** Neither showed up as a crash, a failed assertion or an
implausible number. A spring screened cleanly and returned a confident, narrow,
calibrated-looking interval that was a hundred times too low.

The check that would have caught both is small and now exists: assert that the
deployed selection and the training tag pick the same faces, by count as well as
identity. The first version of that test compared faces as a set, which collapsed
the two coincident spring faces into one entry and let the second defect through
anyway — a reminder that a test comparing the wrong thing is worth roughly
nothing.
