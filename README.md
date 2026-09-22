# fea-surrogate

Surrogate screening server for CAD Agent. Emits `SurrogateScreening/v1`
documents consumed by `design_screening_record`.

A screening estimate is triage, not evidence. Every prediction is a
calibrated interval with declared coverage, refused outside the fitted
training domain, and gates nothing in the consuming design session.

## The flow

1. **`surrogate_inspect_cad`** reads the part and lists its faces — area,
   centroid, normal, planar or curved — with a proposed load case and what
   still has to be answered. BREP, STEP and IGES; Parasolid and native CAD
   formats need a licensed kernel OpenCASCADE does not have, and are refused
   by name rather than failing inside the mesher.
2. **State the load case**: which faces are held, what pushes where and how
   hard, the material. A proposal is recorded as an assumption until it is
   confirmed, because holding the wrong face screens a different question
   and the answer looks perfectly healthy.
3. **`surrogate_screen`** returns the interval, the parts it resembles with
   their solved stresses rescaled to this load, and everything it assumed.

Gravity is arithmetic, not screening: density times volume times g, weighed
against the applied load. Under 5% it proceeds as a stated assumption; above
it is refused, because the training set contains no body loads at all. Two
simultaneous load sets are refused too — fields superpose in linear
elasticity, but the screened quantity is a critical-distance maximum, and the
maximum of a sum is not the sum of the maxima.

## Coverage on the path a user takes

A band measured on held-out samples is a claim about the sweep, not about a
STEP file someone hands you. [docs/DEPLOYED_COVERAGE.md](docs/DEPLOYED_COVERAGE.md)
measures the second thing, and the first time it ran the answer was 0.415
against a calibrated 0.90. Run `python -m tools.deployed_coverage 100 6` after
any change to boundary-condition selection, the features, the mesher or the
model.

`spring` is currently refused: the shipped weights were fitted on spring
geometry carrying a duplicated end face that real CAD does not have.

## Licences

- Training data: `cmudrc/SFEM` (MIT) — Ezemba, McComb & Tucker,
  *Neural Network Surrogate Modeling for Stochastic FEM using 3D Graph
  Representations*, J. Mech. Des., 2025.
- FEniCSx/DOLFINx (LGPL) and Gmsh (GPL) are invoked as external tools.
- SimJEB/DeepJEB are benchmarks only; their CAD is non-commercial and is
  never redistributed here.
