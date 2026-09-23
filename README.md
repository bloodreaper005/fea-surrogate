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
4. **`surrogate_solve`** runs the real solver on the part and returns the true
   field as a contour — about 3.5 s for an 11,000-node bracket against 60 ms to
   screen it. Screening is triage that aims an expensive check at the right
   candidate; this is the check. Not an estimate, so it is not dressed as one:
   its own schema, its own attestation, and a stamp reading SOLVED rather than
   ESTIMATE.

Every solve also screens the same part and records whether the interval
contained the solved answer, so each one is a coverage datapoint on a part
someone actually cared about. No synthetic sweep can produce those.

Use `critical_distance_mpa`, never the peak. An elastic peak at a re-entrant
corner is singular and rises without bound as the mesh is refined, so it states
the mesh rather than the part. The contour's second panel maps each value
against the converged one, so the regions not to trust are visible.

Naming a family earns that family's calibration band. It does not make an
arbitrary load case belong to that family, and a document says so in its
limitations when the two differ.

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
