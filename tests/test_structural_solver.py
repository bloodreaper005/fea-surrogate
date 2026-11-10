"""The static structural solver, checked against things known independently.

A solver that is wrong still draws a convincing contour plot. So every claim is
pinned to something it cannot fake: beam theory for bending, the training
solver for the 4-node element, exact resultants for every load type, linearity
for superposition and E, and equilibrium for reactions. The 10-node element is
the reason this module exists -- 4-node tetrahedra are 16-54% low in bending at
the surrogate's mesh -- so the beam checks are on it, at coarse meshes.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

gmsh = pytest.importorskip("gmsh", reason="meshing requires the inference extra")
pytest.importorskip("scipy", reason="solving requires the inference extra")

from fea_surrogate import structural  # noqa: E402
from fea_surrogate.structural import (  # noqa: E402
    LARGE_DOF_THRESHOLD,
    StructuralSolveError,
    elasticity_matrix,
    estimate_solve_seconds,
    solve_static,
    von_mises,
)
from fea_surrogate.structural_case import (  # noqa: E402
    CaseError,
    FixedSupport,
    ForceLoad,
    Gravity,
    Material,
    MeshSettings,
    PressureLoad,
    StaticStructuralCase,
    custom_material,
    material_from_key,
)
from fea_surrogate.structural_mesh import AnalysisMesh, mesh_for_analysis  # noqa: E402

# The closed-form cantilever: L = 100, b = h = 10 mm, P = 1000 N, E = 200 GPa.
# Euler-Bernoulli tip deflection P L^3 / (3 E I) = 2.000 mm; the fully clamped
# 3D root restrains warping, which is why the converged answer sits on EB
# rather than on Timoshenko's 2.0156 (measured: 2.0006).
LENGTH, WIDTH, HEIGHT, FORCE = 100.0, 10.0, 10.0, 1000.0
STEEL = material_from_key("structural_steel")
EB_TIP = FORCE * LENGTH**3 / (3 * STEEL.youngs_modulus_mpa * WIDTH * HEIGHT**3 / 12)


def _write(path: Path, build) -> Path:  # noqa: ANN001
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add(path.stem)
        build(gmsh.model.occ)
        gmsh.model.occ.synchronize()
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
    return path


@pytest.fixture(scope="module")
def beam_path(tmp_path_factory) -> Path:
    return _write(tmp_path_factory.mktemp("beam") / "beam.step",
                  lambda occ: occ.addBox(0.0, 0.0, 0.0, LENGTH, WIDTH, HEIGHT))


@pytest.fixture(scope="module")
def beam(beam_path) -> AnalysisMesh:
    return mesh_for_analysis(beam_path, refinement="coarse")


@pytest.fixture(scope="module")
def beam_p1(beam_path) -> AnalysisMesh:
    return mesh_for_analysis(beam_path, order=1, refinement="medium")


@pytest.fixture(scope="module")
def plate_path(tmp_path_factory) -> Path:
    def build(occ):  # noqa: ANN001, ANN202
        plate = occ.addBox(0.0, 0.0, 0.0, 100.0, 40.0, 10.0)
        bore = occ.addCylinder(50.0, 20.0, 0.0, 0.0, 0.0, 10.0, 6.0)
        occ.cut([(3, plate)], [(3, bore)])
    return _write(tmp_path_factory.mktemp("plate") / "plate.step", build)


def _face(mesh: AnalysisMesh, axis: int, sign: float) -> int:
    """The planar face whose outward normal is +/- this axis, farthest that way."""
    facing = [f for f in mesh.faces if f.planar and f.normal[axis] * sign > 0.99]
    return max(facing, key=lambda f: sign * f.centroid[axis]).tag


def _cantilever(mesh: AnalysisMesh, material: Material = STEEL, **extra) -> StaticStructuralCase:
    return StaticStructuralCase(
        supports=(FixedSupport((_face(mesh, 0, -1),)),),
        loads=(ForceLoad((_face(mesh, 0, +1),), (0.0, 0.0, -FORCE)),),
        material=material,
        mesh=MeshSettings(element_order=mesh.order),
        **extra,
    )


@pytest.fixture(scope="module")
def solved(beam) -> structural.StaticSolution:
    return solve_static(beam, _cantilever(beam))


# --------------------------------------------------------------------------
# Beam theory
# --------------------------------------------------------------------------


@pytest.mark.parametrize("refinement", ["coarse", "medium"])
def test_tip_deflection_matches_beam_theory(beam_path, refinement) -> None:
    """One 10-node element through the thickness is already within 2%."""
    mesh = mesh_for_analysis(beam_path, refinement=refinement)
    result = solve_static(mesh, _cantilever(mesh))

    tip = result.displacement_mm[mesh.face_nodes[_face(mesh, 0, +1)], 2].mean()
    assert EB_TIP == pytest.approx(2.000, rel=1e-12)
    assert -tip == pytest.approx(EB_TIP, rel=0.02)
    assert result.max_deformation.value == pytest.approx(EB_TIP, rel=0.02)


def test_mid_span_bending_stress_matches_beam_theory(solved, beam) -> None:
    """sigma_xx at the top fibre = 6 P (L - x) / (b h^2): 300 MPa at x = 50.

    Read away from the clamp on purpose: at the root the clamp is singular,
    which is how a 4-node solver's root peak can 'agree' with 6PL/bh^2 while
    the rest of its beam is 25% low.
    """
    top = np.flatnonzero(np.isclose(beam.nodes[:, 2], HEIGHT) &
                         (beam.nodes[:, 0] >= 25.0) & (beam.nodes[:, 0] <= 75.0))
    slope, intercept = np.polyfit(beam.nodes[top, 0], solved.stress_mpa[top, 0], 1)
    at_mid_span = slope * 50.0 + intercept

    assert at_mid_span == pytest.approx(6 * FORCE * (LENGTH - 50.0) / (WIDTH * HEIGHT**2), rel=0.02)
    assert at_mid_span == pytest.approx(300.0, rel=0.02)
    # And the bottom fibre is in equal compression.
    bottom = np.flatnonzero(np.isclose(beam.nodes[:, 2], 0.0) & (np.abs(beam.nodes[:, 0] - 50.0) <= 25.0))
    slope_b, intercept_b = np.polyfit(beam.nodes[bottom, 0], solved.stress_mpa[bottom, 0], 1)
    assert slope_b * 50.0 + intercept_b == pytest.approx(-300.0, rel=0.02)


def test_the_first_order_element_is_the_training_solvers(beam_p1) -> None:
    """On the same 4-node mesh with the same single force, this solver and
    `generate/solver.solve` (which made the labels) must agree to rounding --
    after scaling theirs from E = 1 to the real E."""
    from fea_surrogate.generate.family import Sample, Tagging
    from fea_surrogate.generate.harness import MeshedSample
    from fea_surrogate.generate.solver import solve

    mesh = beam_p1
    case = _cantilever(mesh)
    ours = solve_static(mesh, case)

    n = mesh.node_count
    fixed = np.zeros(n, dtype=bool)
    fixed[mesh.face_nodes[_face(mesh, 0, -1)]] = True
    loaded = np.zeros(n, dtype=bool)
    loaded[mesh.face_nodes[_face(mesh, 0, +1)]] = True
    sample = Sample("beam", {}, Tagging((1,), (2,), (0.0, 0.0, -1.0)), STEEL.poisson_ratio, FORCE)
    theirs = solve(MeshedSample(sample, mesh.nodes, mesh.elements, mesh.boundary,
                                np.zeros((n, 13)), fixed, loaded))

    u_theirs = theirs.displacement / STEEL.youngs_modulus_mpa
    scale_u = np.abs(u_theirs).max()
    assert np.abs(ours.displacement_mm - u_theirs).max() / scale_u < 1e-9
    assert np.abs(ours.von_mises_mpa - theirs.von_mises).max() / theirs.von_mises.max() < 1e-9


def test_the_first_order_element_is_much_too_stiff_in_bending(beam_p1, solved) -> None:
    """Why order 2 is the default: P1 at one-ish element through is far off."""
    p1 = solve_static(beam_p1, _cantilever(beam_p1))
    assert p1.max_deformation.value < 0.8 * solved.max_deformation.value


# --------------------------------------------------------------------------
# Linearity
# --------------------------------------------------------------------------


def test_deformation_scales_as_one_over_e_and_stress_does_not_depend_on_it(beam, solved) -> None:
    aluminium = material_from_key("aluminium")
    soft = custom_material(aluminium.youngs_modulus_mpa, STEEL.poisson_ratio)
    other = solve_static(beam, _cantilever(beam, soft))

    ratio = STEEL.youngs_modulus_mpa / soft.youngs_modulus_mpa
    scale = np.abs(solved.displacement_mm).max()
    assert np.abs(other.displacement_mm - ratio * solved.displacement_mm).max() / (ratio * scale) < 1e-9
    assert np.abs(other.stress_mpa - solved.stress_mpa).max() / np.abs(solved.stress_mpa).max() < 1e-9


def test_two_loads_are_the_sum_of_each_alone(beam) -> None:
    """Superposition: one solve of A + B equals solve(A) + solve(B)."""
    held = FixedSupport((_face(beam, 0, -1),))
    end = ForceLoad((_face(beam, 0, +1),), (0.0, 300.0, -800.0), label="End force")
    top = PressureLoad((_face(beam, 2, +1),), 1.5, label="Top pressure")

    def run(*loads) -> structural.StaticSolution:  # noqa: ANN002
        return solve_static(beam, StaticStructuralCase((held,), loads, STEEL))

    both, first, second = run(end, top), run(end), run(top)
    expected = first.displacement_mm + second.displacement_mm
    assert np.abs(both.displacement_mm - expected).max() / np.abs(expected).max() < 1e-9
    stress = first.stress_mpa + second.stress_mpa
    assert np.abs(both.stress_mpa - stress).max() / np.abs(stress).max() < 1e-9
    assert [a.label for a in both.applied] == ["End force", "Top pressure"]


# --------------------------------------------------------------------------
# Loads and reactions
# --------------------------------------------------------------------------


def test_reactions_balance_the_applied_load(solved) -> None:
    applied = np.array(solved.applied_total_n)
    reaction = np.array(solved.reaction_total_n)

    assert applied == pytest.approx([0.0, 0.0, -FORCE], abs=1e-9)
    assert np.abs(applied + reaction).max() < 1e-6 * FORCE
    assert solved.equilibrium_error < 1e-6
    (support,) = solved.reactions
    assert support.label == "Fixed Support"
    assert support.force_n == pytest.approx(tuple(reaction), abs=1e-9)
    assert solved.residual < 1e-9


def test_a_force_is_spread_over_its_faces_true_area(solved) -> None:
    (load,) = solved.applied
    assert load.kind == "force"
    assert load.area_mm2 == pytest.approx(WIDTH * HEIGHT, rel=1e-12)
    assert load.resultant_n == pytest.approx((0.0, 0.0, -FORCE), abs=1e-9)


def test_two_supports_share_the_reaction_without_double_counting(beam) -> None:
    """A node on both supports counts once, so the parts add up to the total."""
    case = StaticStructuralCase(
        supports=(FixedSupport((_face(beam, 0, -1),), label="Left"),
                  FixedSupport((_face(beam, 0, +1), _face(beam, 1, -1)), label="Right")),
        loads=(PressureLoad((_face(beam, 2, +1),), 2.0),),
        material=STEEL,
    )
    result = solve_static(beam, case)

    parts = np.array([r.force_n for r in result.reactions])
    assert parts.sum(axis=0) == pytest.approx(result.reaction_total_n, abs=1e-9)
    assert np.array(result.reaction_total_n) == pytest.approx([0.0, 0.0, 2.0 * LENGTH * WIDTH], rel=1e-9)
    assert [r.label for r in result.reactions] == ["Left", "Right"]
    assert all(r.force_n[2] > 0 for r in result.reactions)


def test_pressure_on_a_flat_face_is_minus_p_a_n(beam) -> None:
    """Positive pressure pushes INTO the surface (Ansys): on the top face, whose
    outward normal is +z, 2 MPa over 1000 mm^2 is 2000 N downward."""
    top = _face(beam, 2, +1)
    case = StaticStructuralCase((FixedSupport((_face(beam, 0, -1),)),),
                                (PressureLoad((top,), 2.0),), STEEL)
    result = solve_static(beam, case)

    (load,) = result.applied
    area = LENGTH * WIDTH
    assert load.kind == "pressure" and load.area_mm2 == pytest.approx(area, rel=1e-12)
    assert load.resultant_n == pytest.approx((0.0, 0.0, -2.0 * area), rel=1e-12, abs=1e-9)
    assert result.max_deformation.location_mm[0] == pytest.approx(LENGTH)
    assert result.displacement_mm[:, 2].min() < 0.0 < -result.displacement_mm[:, 2].min()

    suction = solve_static(beam, replace(case, loads=(PressureLoad((top,), -2.0),)))
    assert np.allclose(suction.displacement_mm, -result.displacement_mm, rtol=1e-9, atol=1e-15)


def test_pressure_in_a_bore_cancels_and_pushes_the_wall_out(plate_path) -> None:
    """A pressure follows a curved face: the net force on a full bore is ~0,
    where a 'pressure times area along one direction' would be p*A."""
    mesh = mesh_for_analysis(plate_path, refinement="coarse")
    (bore,) = [f.tag for f in mesh.faces if not f.planar]
    pressure = 10.0
    case = StaticStructuralCase((FixedSupport((_face(mesh, 0, -1),)),),
                                (PressureLoad((bore,), pressure),), STEEL)
    result = solve_static(mesh, case)

    (load,) = result.applied
    gross = pressure * load.area_mm2
    assert load.area_mm2 == pytest.approx(2 * math.pi * 6.0 * 10.0, rel=1e-4)
    assert np.linalg.norm(load.resultant_n) < 1e-6 * gross
    nodes = mesh.face_nodes[bore]
    radial = mesh.nodes[nodes, :2] - np.array([50.0, 20.0])
    radial /= np.linalg.norm(radial, axis=1, keepdims=True)
    outward = np.einsum("ij,ij->i", result.displacement_mm[nodes, :2], radial)
    assert outward.mean() > 0.0
    assert result.equilibrium_error < 1e-6


def test_gravity_is_rho_v_g(beam) -> None:
    """Self-weight as a consistent body load: resultant exactly rho * V * g."""
    g = (0.0, 0.0, -9.80665)
    case = StaticStructuralCase((FixedSupport((_face(beam, 0, -1),)),), (), STEEL,
                                gravity=Gravity(g))
    result = solve_static(beam, case)

    (load,) = result.applied
    weight = STEEL.density_kg_m3 * 1e-9 * LENGTH * WIDTH * HEIGHT     # kg (mm^3 = 1e-9 m^3)
    assert load.kind == "gravity" and load.faces == () and load.area_mm2 is None
    assert load.resultant_n == pytest.approx(tuple(weight * c for c in g), rel=1e-9, abs=1e-15)
    assert np.array(result.reaction_total_n) == pytest.approx(
        [0.0, 0.0, weight * 9.80665], rel=1e-9, abs=1e-9 * weight * 9.80665)
    # A steel beam this size sags about 1 micron under its own weight: w L^4 / 8 E I.
    line_load = weight * 9.80665 / LENGTH
    expected = line_load * LENGTH**4 / (8 * STEEL.youngs_modulus_mpa * WIDTH * HEIGHT**3 / 12)
    assert result.max_deformation.value == pytest.approx(expected, rel=0.03)


def test_a_straight_ten_node_element_has_the_textbook_body_load_shares() -> None:
    """integral(N_k dV): -V/20 at each corner and V/5 at each mid-edge."""
    corners = np.array([[0.0, 0, 0], [2.0, 0, 0], [0.0, 3, 0], [0.0, 0, 4]])
    mids = [0.5 * (corners[i] + corners[j]) for i, j in structural.TET10_EDGES]
    nodes = np.vstack([corners, mids])
    element = np.arange(10)[None, :]
    stiffness, shares, volume = structural._assemble(nodes, element, elasticity_matrix(1.0, 0.3))

    assert volume == pytest.approx(4.0)                                # 2*3*4/6
    assert shares[:4] == pytest.approx([-volume / 20] * 4)
    assert shares[4:] == pytest.approx([volume / 5] * 6)
    # A rigid translation stores no strain energy.
    for axis in range(3):
        rigid = np.zeros((10, 3))
        rigid[:, axis] = 1.0
        assert np.abs(stiffness @ rigid.ravel()).max() < 1e-12 * np.abs(stiffness).max()


def test_a_flat_six_node_triangle_puts_a_uniform_load_on_its_mid_edges() -> None:
    corners = np.array([[0.0, 0, 0], [4.0, 0, 0], [0.0, 3, 0]])
    nodes = np.vstack([corners] + [0.5 * (corners[i] + corners[j]) for i, j in ((0, 1), (1, 2), (0, 2))])
    forces, area = structural._surface_load(nodes, np.arange(6)[None, :], total_force=np.array([0, 0, -6.0]))

    assert area == pytest.approx(6.0)
    assert forces[:3] == pytest.approx(np.zeros((3, 3)), abs=1e-14)
    assert forces[3:, 2] == pytest.approx([-2.0] * 3)


def test_a_uniform_strain_field_recovers_uniform_stress(beam) -> None:
    """Patch test on the real mesh: u = eps . x gives sigma = D eps at every node."""
    strain = np.array([[1e-3, 2e-4, 0.0], [2e-4, -3e-4, 1e-4], [0.0, 1e-4, 5e-4]])
    u = beam.nodes @ strain.T
    d = elasticity_matrix(STEEL.youngs_modulus_mpa, STEEL.poisson_ratio)
    voigt = np.array([strain[0, 0], strain[1, 1], strain[2, 2],
                      2 * strain[0, 1], 2 * strain[1, 2], 2 * strain[0, 2]])

    stress = structural._recover_stress(beam.nodes, beam.elements, d, u)

    assert np.abs(stress - d @ voigt).max() < 1e-9 * np.abs(d @ voigt).max()


def test_chunked_assembly_is_the_same_system(beam, solved, monkeypatch) -> None:
    """Large meshes are assembled in chunks to bound memory; the pieces must
    add up to exactly the stiffness a single pass builds."""
    monkeypatch.setattr(structural, "CHUNK_ELEMENTS", {4: 7, 10: 13})
    chunked = solve_static(beam, _cantilever(beam))

    # Equal up to the order floating-point sums are taken in (~1e-11 measured).
    scale = np.abs(solved.displacement_mm).max()
    assert np.abs(chunked.displacement_mm - solved.displacement_mm).max() < 1e-9 * scale
    assert np.abs(chunked.stress_mpa - solved.stress_mpa).max() < 1e-9 * np.abs(solved.stress_mpa).max()


def test_von_mises_of_uniaxial_and_pure_shear() -> None:
    assert von_mises(np.array([100.0, 0, 0, 0, 0, 0])) == pytest.approx(100.0)
    assert von_mises(np.array([0.0, 0, 0, 50.0, 0, 0])) == pytest.approx(50.0 * math.sqrt(3))
    assert von_mises(np.array([7.0, 7.0, 7.0, 0, 0, 0])) == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------
# Honest maximum
# --------------------------------------------------------------------------


def test_the_max_at_a_clamp_edge_is_flagged_and_the_max_away_is_lower(solved, beam) -> None:
    """The fully clamped root of a cantilever is a 3D singularity: its nodal
    max rises with every refinement (591 -> 654 MPa measured), while stress one
    wall-thickness away converges."""
    peak = solved.max_stress
    assert peak.singular
    assert "Fixed Support" in peak.reason and f"face {_face(beam, 0, -1)}" in peak.reason
    assert peak.location_mm[0] == pytest.approx(0.0, abs=beam.element_size_mm)

    away = solved.max_stress_away
    assert away is not None and not away.singular
    assert away.value < peak.value
    assert solved.exclusion_radius_mm == pytest.approx(beam.thinnest)
    flagged = beam.nodes[solved.singular_nodes]
    assert np.min(np.linalg.norm(flagged - np.array(away.location_mm), axis=1)) > beam.thinnest
    # Away from the clamp, beam theory: at most 6P(L - r)/(b h^2), r = the radius.
    assert away.value == pytest.approx(6 * FORCE * (LENGTH - beam.thinnest) / (WIDTH * HEIGHT**2), rel=0.1)


def _check_one_radius(result: structural.StaticSolution) -> None:
    """The singular flag and the max-away exclusion use one radius, so a point
    is never both "singular" and "farther than r from flagged features"."""
    mesh = result.mesh
    flagged = mesh.nodes[result.singular_nodes]
    radius = result.exclusion_radius_mm

    def gap(node: int) -> float:
        return float(np.min(np.linalg.norm(flagged - mesh.nodes[node], axis=1)))

    peak, away = result.max_stress, result.max_stress_away
    assert peak.singular == (gap(peak.node) <= radius)
    if away is not None:
        assert gap(away.node) > radius and not away.singular
        if not peak.singular:
            assert away.node == peak.node and away.value == peak.value
        else:
            assert away.node != peak.node


def test_singular_and_max_away_never_describe_the_same_point(solved, beam, ell_path) -> None:
    """Coarse meshes have elements larger than the wall (7.6 mm here against
    4.8 mm); the flag once used the element size and the exclusion the wall,
    so one point could be both. Checked on coarse and fine meshes."""
    assert beam.element_size_mm > beam.thinnest
    _check_one_radius(solved)
    for refinement in ("coarse", "fine"):
        mesh = mesh_for_analysis(ell_path, refinement=refinement)
        _check_one_radius(solve_static(mesh, _ell_case(mesh)))


def test_the_hot_spot_stress_reads_the_clamped_root(beam_path) -> None:
    """The max one wall-thickness from a clamp leaves out the root section's
    own stress; extrapolating 1-2 thicknesses back to the edge recovers
    6PL/(bh^2) = 600 MPa (measured -2 to -4 % on medium meshes)."""
    mesh = mesh_for_analysis(beam_path, refinement="medium")
    result = solve_static(mesh, _cantilever(mesh))
    root = 6 * FORCE * LENGTH / (WIDTH * HEIGHT**2)
    assert result.max_stress.singular
    assert result.max_stress_extrapolated == pytest.approx(root, rel=0.05)
    assert result.max_stress_away.value < result.max_stress_extrapolated <= result.max_stress.value
    assert result.summary()["equivalent_stress"]["max_extrapolated_to_singularity"] == pytest.approx(
        result.max_stress_extrapolated)


def test_no_hot_spot_from_a_mesh_too_coarse_to_hold_one(solved, beam) -> None:
    """Elements larger than 1.25 wall thicknesses leave a few stray nodes in
    the band (a 20 mm mesh once 'extrapolated' 66.8 MPa beside a 312 MPa max),
    so no value is offered rather than a wrong one."""
    assert beam.element_size_mm > structural.HOT_SPOT_MAX_ELEMENT_RATIO * beam.thinnest
    assert solved.max_stress.singular
    assert solved.max_stress_extrapolated is None


def test_only_the_clamp_perimeter_is_flagged_on_a_convex_part(solved, beam) -> None:
    """A box has only outside corners, which are not singular."""
    held = set(beam.face_nodes[_face(beam, 0, -1)].tolist())
    flagged = set(solved.singular_nodes.tolist())
    assert flagged and flagged <= held
    assert np.allclose(beam.nodes[solved.singular_nodes, 0], 0.0)
    on_perimeter = np.isclose(beam.nodes[solved.singular_nodes, 1:], 0.0) | np.isclose(
        beam.nodes[solved.singular_nodes, 1:], [WIDTH, HEIGHT])
    assert on_perimeter.any(axis=1).all()


@pytest.fixture(scope="module")
def ell_path(tmp_path_factory) -> Path:
    """An L: upright 10 mm thick, leg 10 mm thick, inside corner at x = z = 10."""
    def build(occ):  # noqa: ANN001, ANN202
        leg = occ.addBox(0.0, 0.0, 0.0, 60.0, 20.0, 10.0)
        upright = occ.addBox(0.0, 0.0, 0.0, 10.0, 20.0, 60.0)
        occ.fuse([(3, leg)], [(3, upright)])
    return _write(tmp_path_factory.mktemp("ell") / "ell.step", build)


def _ell_case(mesh: AnalysisMesh) -> StaticStructuralCase:
    """Held at the top of the upright, pushed down at the end of the leg."""
    return StaticStructuralCase((FixedSupport((_face(mesh, 2, +1),)),),
                                (ForceLoad((_face(mesh, 0, +1),), (0.0, 0.0, -500.0)),), STEEL)


def test_a_reentrant_edge_is_flagged_and_an_outside_corner_is_not(ell_path) -> None:
    mesh = mesh_for_analysis(ell_path, refinement="coarse")
    case = _ell_case(mesh)
    result = solve_static(mesh, case)

    x = mesh.nodes[result.singular_nodes]
    inside = np.isclose(x[:, 0], 10.0) & np.isclose(x[:, 2], 10.0)
    outside = np.isclose(x[:, 0], 0.0) & np.isclose(x[:, 2], 0.0)
    assert inside.sum() >= 3 and not outside.any()
    reasons = structural._singular_features(mesh, case)
    for node in result.singular_nodes[inside].tolist():
        assert reasons[node].startswith("re-entrant edge between faces")
    # Mid-edge nodes along the corner are flagged too, not just the corners.
    assert (~mesh.corner_mask[result.singular_nodes[inside]]).any()

    peak = result.max_stress
    assert peak.singular and peak.reason.startswith("re-entrant edge between faces")
    assert peak.location_mm[0] == pytest.approx(10.0) and peak.location_mm[2] >= 10.0


def test_a_flagged_max_grows_with_refinement_and_the_max_away_does_not(ell_path) -> None:
    """Why the flag exists: at the inside corner the elastic stress is infinite,
    so the nodal max is a statement about the mesh (measured 88 -> 121 MPa from
    coarse to fine here) while the max one wall-thickness away holds (85.7 ->
    85.1)."""
    coarse, fine = (solve_static(m, _ell_case(m)) for m in
                    (mesh_for_analysis(ell_path, refinement=r) for r in ("coarse", "fine")))

    assert fine.max_stress.value > 1.2 * coarse.max_stress.value
    assert fine.max_stress_away.value == pytest.approx(coarse.max_stress_away.value, rel=0.05)
    assert fine.max_stress_away.value < fine.max_stress.value


# --------------------------------------------------------------------------
# Results bookkeeping
# --------------------------------------------------------------------------


def test_hotspots_and_auto_scale(solved, beam) -> None:
    tip_face = _face(beam, 0, +1)
    assert tip_face in solved.max_deformation.faces
    assert solved.min_deformation.value == 0.0
    assert solved.total_deformation_mm == pytest.approx(np.linalg.norm(solved.displacement_mm, axis=1))
    assert solved.von_mises_mpa == pytest.approx(von_mises(solved.stress_mpa))
    # MAPDL /DSCALE auto: the biggest displacement drawn at 5% of the largest extent.
    assert solved.deformation_scale_auto == pytest.approx(0.05 * LENGTH / solved.max_deformation.value)
    assert solved.dof_count == 3 * (beam.node_count - beam.face_nodes[_face(beam, 0, -1)].size)


def test_the_summary_is_json_safe_and_finite(solved) -> None:
    summary = solved.summary()
    text = json.dumps(summary, allow_nan=False)

    assert summary["element_order"] == 2
    assert summary["equivalent_stress"]["max_singular"] is True
    assert summary["equivalent_stress"]["max_away_from_singularities"] < summary["equivalent_stress"]["max"]
    assert summary["total_deformation"]["units"] == "mm"
    assert set(summary["timings_s"]) == {"assemble", "loads", "solve", "recover", "total"}
    assert "nan" not in text.lower()


def test_progress_reports_each_stage(beam) -> None:
    stages = []
    solve_static(beam, _cantilever(beam), progress=lambda stage, message: stages.append(stage))
    assert stages == ["assemble", "loads", "solve", "recover"]


def test_the_time_estimate_follows_the_measurements() -> None:
    # Bracket sample on an M4, assembly + factorisation + recovery.
    assert estimate_solve_seconds(36_324) == pytest.approx(1.3, rel=0.5)
    assert estimate_solve_seconds(188_265) == pytest.approx(23.0, rel=0.3)
    assert estimate_solve_seconds(LARGE_DOF_THRESHOLD) > estimate_solve_seconds(188_265)
    assert estimate_solve_seconds(LARGE_DOF_THRESHOLD) < 60.0


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------


def test_a_case_without_a_support_is_refused(beam) -> None:
    with pytest.raises(CaseError, match="fixed support"):
        StaticStructuralCase((), (ForceLoad((1,), (0.0, 0.0, -1.0)),), STEEL)
    # And the solver does not rely on the case having checked.
    unsupported = _cantilever(beam)
    object.__setattr__(unsupported, "supports", ())
    with pytest.raises(StructuralSolveError, match="no fixed support"):
        solve_static(beam, unsupported)


def test_a_floating_second_body_is_refused(tmp_path) -> None:
    def build(occ):  # noqa: ANN001, ANN202
        occ.addBox(0.0, 0.0, 0.0, 60.0, 20.0, 10.0)
        occ.addBox(100.0, 0.0, 0.0, 60.0, 20.0, 10.0)

    mesh = mesh_for_analysis(_write(tmp_path / "pair.step", build), refinement="coarse")
    held = min(mesh.faces, key=lambda f: f.centroid[0]).tag
    pushed = max(mesh.faces, key=lambda f: f.centroid[0]).tag
    case = StaticStructuralCase((FixedSupport((held,)),),
                                (ForceLoad((pushed,), (0.0, 0.0, -100.0)),), STEEL)

    with pytest.raises(StructuralSolveError) as caught:
        solve_static(mesh, case)
    message = str(caught.value)
    assert "2 disconnected bodies" in message and "1 of them" in message
    assert "6 rigid-body modes" in message and "noise" in message


def test_poissons_ratio_out_of_range_is_refused(beam) -> None:
    with pytest.raises(CaseError, match="Poisson"):
        custom_material(200000.0, 0.5)
    bad = custom_material(200000.0, 0.3)
    object.__setattr__(bad, "poisson_ratio", 0.5)
    with pytest.raises(StructuralSolveError, match="Poisson"):
        solve_static(beam, _cantilever(beam, bad))


def test_a_large_mesh_needs_permission(beam, monkeypatch) -> None:
    """Minutes of silence look like a hang; the caller must opt in."""
    monkeypatch.setattr(structural, "LARGE_DOF_THRESHOLD", 100)
    with pytest.raises(StructuralSolveError, match="allow_large"):
        solve_static(beam, _cantilever(beam))
    assert solve_static(beam, _cantilever(beam), allow_large=True).dof_count > 100


def test_a_mesh_of_the_wrong_order_is_refused(beam, beam_p1) -> None:
    with pytest.raises(StructuralSolveError, match="element order 2"):
        solve_static(beam_p1, replace(_cantilever(beam_p1), mesh=MeshSettings(element_order=2)))


def test_a_face_the_mesh_does_not_have_is_refused(beam) -> None:
    case = StaticStructuralCase((FixedSupport((_face(beam, 0, -1),)),),
                                (ForceLoad((42,), (0.0, 0.0, -1.0)),), STEEL)
    with pytest.raises(StructuralSolveError, match="face table"):
        solve_static(beam, case)


def test_a_singular_system_is_refused_not_reported(beam, monkeypatch) -> None:
    """If the factorisation hands back garbage, nothing is reported from it."""
    import scipy.sparse.linalg as linalg

    monkeypatch.setattr(structural, "_direct_solve", lambda matrix, rhs: np.full(rhs.shape, np.nan))
    with pytest.raises(StructuralSolveError, match="non-finite"):
        solve_static(beam, _cantilever(beam))

    monkeypatch.setattr(structural, "_direct_solve", lambda matrix, rhs: np.ones(rhs.shape))
    with pytest.raises(StructuralSolveError, match="residual"):
        solve_static(beam, _cantilever(beam))

    def warns(matrix, rhs):  # noqa: ANN001, ANN202
        import warnings
        warnings.warn("Matrix is exactly singular", linalg.MatrixRankWarning, stacklevel=1)
        return np.full(rhs.shape, np.nan)

    monkeypatch.setattr(structural, "_direct_solve", warns)
    with pytest.raises(StructuralSolveError, match="singular"):
        solve_static(beam, _cantilever(beam))

    def raises(matrix, rhs):  # noqa: ANN001, ANN202 - what SuperLU does on a zero pivot
        raise RuntimeError("Factor is exactly singular")

    monkeypatch.setattr(structural, "_direct_solve", raises)
    with pytest.raises(StructuralSolveError, match="rigid body"):
        solve_static(beam, _cantilever(beam))


def test_the_symmetric_factorisation_matches_a_general_lu(beam) -> None:
    """`_direct_solve` uses diagonal pivots, which only an SPD matrix allows;
    on the real stiffness it must give spsolve's answer."""
    from scipy.sparse.linalg import spsolve

    from fea_surrogate.structural import _assemble, _direct_solve, elasticity_matrix

    stiffness, _, _ = _assemble(beam.nodes, beam.elements, elasticity_matrix(200000.0, 0.3))
    held = np.zeros(beam.node_count, dtype=bool)
    held[beam.face_nodes[_face(beam, 0, -1)]] = True
    free = np.flatnonzero(np.repeat(~held, 3))
    reduced = stiffness[free][:, free].tocsc()
    rhs = np.random.default_rng(0).normal(size=free.size)
    fast, general = _direct_solve(reduced, rhs), spsolve(reduced, rhs)
    assert np.abs(fast - general).max() <= 1e-8 * np.abs(general).max()
