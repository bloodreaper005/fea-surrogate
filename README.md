# fea-surrogate

Surrogate screening server for CAD Agent. Emits `SurrogateScreening/v1`
documents consumed by `design_screening_record`.

A screening estimate is triage, not evidence. Every prediction is a
calibrated interval with declared coverage, refused outside the fitted
training domain, and gates nothing in the consuming design session.

## Licences

- Training data: `cmudrc/SFEM` (MIT) — Ezemba, McComb & Tucker,
  *Neural Network Surrogate Modeling for Stochastic FEM using 3D Graph
  Representations*, J. Mech. Des., 2025.
- FEniCSx/DOLFINx (LGPL) and Gmsh (GPL) are invoked as external tools.
- SimJEB/DeepJEB are benchmarks only; their CAD is non-commercial and is
  never redistributed here.
