"""`fea-structural`: the interview, the flags, the exit codes and the results table.

Most of this is light: the mesher, solver and renderer are replaced by stand-ins
on `structural_run`, so the prompt logic -- re-asking on bad input, defaults,
several loads, a custom material, declining at the confirmation, EOF -- runs in
the contract job without gmsh. The last section is one real end-to-end run on a
generated beam (and the demo command), and skips only when gmsh, scipy or
matplotlib are missing.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import replace as dataclasses_replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fea_surrogate import cli
from fea_surrogate import structural_run as sr
from fea_surrogate.structural_case import (
    MATERIAL_PRESETS,
    FixedSupport,
    ForceLoad,
    Gravity,
    MeshSettings,
    PressureLoad,
    StaticStructuralCase,
)

SRC = str(Path(__file__).resolve().parents[1] / "src")
HEAVY = ("torch", "torch_geometric", "gmsh", "scipy", "matplotlib")


# --------------------------------------------------------------------------
# stand-ins
# --------------------------------------------------------------------------


def box_mesh(length=100.0, width=20.0, height=10.0, *, nodes=1234, settings=None):
    """A box as an AnalysisMesh looks to the CLI: faces 1 x-min, 2 x-max,
    3 y-min, 4 y-max, 5 z-min, 6 z-max."""
    L, W, H = length, width, height
    corners = np.array([[x, y, z] for x in (0, L) for y in (0, W) for z in (0, H)], float)
    spec = [
        (1, (0, W / 2, H / 2), (-1, 0, 0), W * H, 0, 0.0),
        (2, (L, W / 2, H / 2), (1, 0, 0), W * H, 0, L),
        (3, (L / 2, 0, H / 2), (0, -1, 0), L * H, 1, 0.0),
        (4, (L / 2, W, H / 2), (0, 1, 0), L * H, 1, W),
        (5, (L / 2, W / 2, 0), (0, 0, -1), L * W, 2, 0.0),
        (6, (L / 2, W / 2, H), (0, 0, 1), L * W, 2, H),
    ]
    faces, centroids, face_nodes = [], {}, {}
    for tag, centre, normal, area, axis, value in spec:
        idx = np.flatnonzero(np.isclose(corners[:, axis], value))
        pts = corners[idx]
        faces.append(SimpleNamespace(
            tag=tag, area=float(area), centroid=centre, normal=normal, planar=True,
            kind="plane", bbox_min=tuple(pts.min(axis=0)), bbox_max=tuple(pts.max(axis=0)),
        ))
        centroids[tag] = centre
        face_nodes[tag] = idx
    faces.sort(key=lambda f: -f.area)
    settings = settings or MeshSettings()
    return SimpleNamespace(
        faces=tuple(faces), face_centroids=centroids, face_nodes=face_nodes,
        vertices=corners, nodes=corners, thinnest=min(L, W, H),
        diagonal_mm=float(math.sqrt(L * L + W * W + H * H)), solid_count=1,
        volume_mm3=L * W * H, node_count=nodes, element_count=nodes // 2,
        dof_count=3 * nodes, order=settings.element_order, element_size_mm=5.0,
        min_scaled_jacobian=0.4, straight_sided=False, healed=False, settings=settings,
    )


class Harness:
    """Patches structural_run with stand-ins and records what the CLI asked for."""

    def __init__(self, monkeypatch, tmp_path: Path, *, run_error: Exception | None = None):
        self.tmp = tmp_path
        self.part = tmp_path / "part.step"
        self.part.write_text("not really a STEP; check_readable only checks the name\n")
        self.out = tmp_path / "out"
        self.meshed: list[MeshSettings] = []
        self.runs: list[dict] = []
        self.printed: list[str] = []
        self.prompts: list[str] = []

        def mesh_part(path, settings):
            self.meshed.append(settings)
            return box_mesh(nodes=1000 * len(self.meshed), settings=settings)

        def run(path, case, **kwargs):
            if run_error is not None:
                raise run_error
            self.runs.append({"path": path, "case": case, **kwargs})
            return SimpleNamespace(
                files={}, as_document=lambda: {"schema_version": sr.SCHEMA, "ok": True},
            )

        def faces_image(mesh, out_path):
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b"png")
            return out_path

        solid = SimpleNamespace(
            faces=box_mesh().faces, bbox_min=(0, 0, 0), bbox_max=(100, 20, 10),
            volume=20000.0, solid_count=1, healed=False,
        )
        monkeypatch.setattr(sr, "inspect_part", lambda path: solid)
        monkeypatch.setattr(sr, "mesh_part", mesh_part)
        monkeypatch.setattr(sr, "render_faces_image", faces_image)
        monkeypatch.setattr(sr, "estimate_seconds", lambda mesh: 1.5)
        monkeypatch.setattr(sr, "large_dof_threshold", lambda: 150_000)
        monkeypatch.setattr(sr, "find_surrogate_files", lambda *a, **k: (None, None, None))
        monkeypatch.setattr(sr, "run_static_structural", run)
        monkeypatch.setattr(sr, "format_summary", lambda run: "RESULTS TABLE")
        monkeypatch.setattr(cli, "_missing_extras", lambda modules: [])

    def main(self, argv, answers=()):
        queue = list(answers)

        def input_fn(prompt):
            self.prompts.append(prompt)
            if not queue:
                raise EOFError
            answer = queue.pop(0)
            if isinstance(answer, BaseException):
                raise answer
            self.printed.append(f"{prompt}{answer}")
            return answer

        return cli.main(
            [str(a) for a in argv], input_fn=input_fn,
            print_fn=lambda *a, **k: self.printed.append(" ".join(str(x) for x in a)),
        )

    @property
    def text(self) -> str:
        return "\n".join(self.printed)

    @property
    def case(self) -> StaticStructuralCase:
        assert self.runs, "the solve was never started"
        return self.runs[-1]["case"]


@pytest.fixture
def harness(monkeypatch, tmp_path):
    return Harness(monkeypatch, tmp_path)


# --------------------------------------------------------------------------
# import weight and --help
# --------------------------------------------------------------------------


def _heavy_after(script: str) -> tuple[str, list[str]]:
    full = script + f"\nprint('HEAVY:' + ' '.join(m for m in {HEAVY!r} if m in sys.modules))\n"
    result = subprocess.run([sys.executable, "-c", "import sys\n" + full],
                            capture_output=True, text=True, check=True, cwd=SRC)
    heavy_line = [l for l in result.stdout.splitlines() if l.startswith("HEAVY:")][-1]
    return result.stdout, heavy_line[len("HEAVY:"):].split()


def test_help_works_without_the_heavy_modules() -> None:
    out, heavy = _heavy_after(
        "from fea_surrogate.cli import main\nprint('RC', main(['--help']))"
    )
    assert "RC 0" in out
    assert "fea-structural" in out and "--fixed" in out and "inspect PART.step" in out
    assert heavy == []


def test_the_run_modules_import_light() -> None:
    _, heavy = _heavy_after(
        "import fea_surrogate.cli, fea_surrogate.structural_run, "
        "fea_surrogate.screenability, fea_surrogate.structural_case"
    )
    assert heavy == []


def test_subcommand_help() -> None:
    assert cli.main(["inspect", "--help"], print_fn=lambda *a: None) == 0
    assert cli.main(["demo", "--help"], print_fn=lambda *a: None) == 0


# --------------------------------------------------------------------------
# the interview
# --------------------------------------------------------------------------


def test_interview_with_custom_material_two_loads_and_re_asks(harness) -> None:
    code = harness.main([harness.part, "--out", harness.out, "--no-open"], [
        "c",              # custom material
        "70 GPa",         # E
        "0.33",           # nu
        "",               # density: none
        "270",            # yield
        "Alu 6061",       # name
        "9",              # fixed: no such face -> re-ask
        "1",
        "",               # load B: force (default)
        "1",              # faces: all fixed -> re-ask
        "2",
        "lots",           # not a force -> re-ask
        "500 N -z",
        "y",              # another load
        "p",
        "6",
        "0",              # zero pressure -> re-ask
        "1.5",
        "",               # no more loads
        "",               # gravity: no
        "",               # family: none (two loads match nothing)
        "",               # mesh: keep medium
        "y",              # proceed
    ])

    assert code == 0, harness.text
    case = harness.case
    assert case.material.youngs_modulus_mpa == pytest.approx(70000.0)
    assert case.material.poisson_ratio == pytest.approx(0.33)
    assert case.material.tensile_yield_mpa == pytest.approx(270.0)
    assert case.material.name == "Alu 6061"
    assert case.material.density_kg_m3 is None
    assert case.fixed_faces() == (1,)
    force, pressure = case.loads
    assert isinstance(force, ForceLoad) and force.vector_n == pytest.approx((0, 0, -500.0))
    assert isinstance(pressure, PressureLoad) and pressure.pressure_mpa == pytest.approx(1.5)
    assert case.gravity is None
    assert harness.runs[-1]["family"] is None
    assert harness.runs[-1]["out_dir"] == harness.out

    text = harness.text
    assert "does not have" in text                     # unknown face, one line why
    assert "fixed, so a load there goes straight into the support" in text
    assert "'lots' is not a force" in text
    assert "pressure must not be zero" in text
    # re-asked the same question after each bad answer
    assert sum("A  Fixed Support faces" in p for p in harness.prompts) == 2
    # echoes: resolved vector, area and traction; pressure equivalent force
    assert "500 N along (0, 0, -1) -Z on face 2: 200 mm², traction 2.5 MPa" in text
    assert "1.5 MPa normal to face 6 (into the surface)" in text
    assert "equivalent 3000 N along (0, 0, -1)" in text
    # the confirmation box with letters and the screening verdict
    assert "┌─ A: Static Structural" in text
    assert "A  Fixed Support — face 1" in text
    assert "B  Force" in text and "C  Pressure" in text
    assert "not screenable" in text and "2 loads" in text


def test_defaults_in_brackets_and_question_mark_help(harness) -> None:
    code = harness.main([harness.part, "--out", harness.out, "--no-open"], [
        "?", "",          # help, then default material 1 = structural steel
        "1", "", "2", "500 N -z", "", "", "", "", "",
    ])
    assert code == 0, harness.text
    assert harness.prompts[0] == "Material [1]: "
    assert "supplier's data sheet" in harness.text      # the help text
    assert harness.case.material == MATERIAL_PRESETS["structural_steel"]
    assert any(p.startswith("Load B: [f]orce or [p]ressure [f]") for p in harness.prompts)
    assert any("Proceed? [Y/n]" in p for p in harness.prompts)


def test_the_matching_family_is_offered_as_the_default(harness) -> None:
    code = harness.main([harness.part, "--out", harness.out, "--no-open"], [
        "1", "1", "", "2", "500 N -z", "", "", "", "", "",
    ])
    assert code == 0, harness.text
    assert "'bracket' family's trained load case" in harness.text
    family_prompt = [p for p in harness.prompts if p.startswith("Surrogate family")][0]
    assert family_prompt.endswith("[bracket]: ")
    assert harness.runs[-1]["family"] == "bracket"


def test_gravity_asks_for_its_direction_and_a_density_when_missing(harness) -> None:
    code = harness.main([harness.part, "--out", harness.out, "--no-open", "--no-screen"], [
        "c", "200 GPa", "", "", "", "",      # custom steel, no density
        "1", "", "2", "500 N -z", "",
        "y",                                  # gravity
        "normal",                             # not a gravity direction -> re-ask
        "",                                   # -z
        "7.85 g/cm3",                         # density now required
        "", "",                               # mesh, proceed
    ])
    assert code == 0, harness.text
    case = harness.case
    assert case.gravity.acceleration_mps2 == pytest.approx((0, 0, -9.80665))
    assert case.material.density_kg_m3 == pytest.approx(7850.0)
    assert not any(p.startswith("Surrogate family") for p in harness.prompts)   # --no-screen


def test_changing_the_mesh_re_meshes(harness) -> None:
    code = harness.main([harness.part, "--out", harness.out, "--no-open", "--no-screen"], [
        "1", "1", "", "2", "500 N -z", "", "", "fine", "",
    ])
    assert code == 0, harness.text
    assert [s.refinement for s in harness.meshed] == ["medium", "fine"]
    assert harness.case.mesh.refinement == "fine"
    assert harness.runs[-1]["mesh"].node_count == 2000          # the new mesh is reused


def test_declining_at_the_confirmation_solves_nothing(harness) -> None:
    code = harness.main([harness.part, "--out", harness.out, "--no-open"], [
        "1", "1", "", "2", "500 N -z", "", "", "", "", "n",
    ])
    assert code == 0
    assert harness.runs == []
    assert "nothing was solved" in harness.text
    # the answers are kept, so declining does not cost the interview
    saved = harness.out / "case.json"
    assert saved.is_file() and f"rerun with --case {saved}" in harness.text
    assert json.loads(saved.read_text())["loads"][0]["faces"] == [2]


def test_end_of_input_exits_130_without_a_traceback(harness) -> None:
    code = harness.main([harness.part, "--out", harness.out, "--no-open"], ["1", "1"])
    assert code == 130
    assert "Interrupted" in harness.text
    assert "Traceback" not in harness.text
    assert harness.runs == []


def test_ctrl_c_exits_130(harness) -> None:
    code = harness.main([harness.part, "--out", harness.out, "--no-open"], [KeyboardInterrupt()])
    assert code == 130


def test_back_switch_and_undo_in_the_load_questions(harness) -> None:
    """T7: a wrong turn in a load costs one word, not the interview."""
    code = harness.main([harness.part, "--out", harness.out, "--no-open", "--no-screen"], [
        "1", "1",
        "", "back",               # B force -> faces: back to force/pressure
        "", "2", "p",             # force prompt: p switches to a pressure
        "back",                   # pressure prompt: back
        "f", "2", "500 N -z",
        "undo",                   # remove B; a case needs a load, so B is asked again
        "p", "6", "1.5", "",
        "", "", "",               # gravity, mesh, proceed
    ])
    assert code == 0, harness.text
    (load,) = harness.case.loads
    assert isinstance(load, PressureLoad) and load.faces == (6,) and load.pressure_mpa == 1.5
    assert "removed B (force on face 2)" in harness.text
    assert sum(p.startswith("Load B: [f]orce or [p]ressure") for p in harness.prompts) == 4


def test_no_density_at_the_gravity_question_drops_gravity(harness) -> None:
    code = harness.main([harness.part, "--out", harness.out, "--no-open", "--no-screen"], [
        "c", "200 GPa", "", "", "", "",      # custom, no density
        "1", "", "2", "500 N -z", "",
        "y", "",                              # gravity along -z
        "none",                               # no density after all
        "", "",
    ])
    assert code == 0, harness.text
    assert harness.case.gravity is None
    assert "no gravity (no density given)" in harness.text


def test_a_mesh_no_solver_could_factor_is_refused_at_the_mesh_question(harness) -> None:
    """T5: checked where it is asked, not after the confirmation box."""
    code = harness.main([harness.part, "--out", harness.out, "--no-open", "--no-screen"], [
        "1", "1", "", "2", "500 N -z", "", "",
        "0.001 mm",               # trillions of elements -> asked again
        "0 mm",                   # not a size -> asked again
        "", "",                   # keep medium, proceed
    ])
    assert code == 0, harness.text
    assert "far more than a direct solver can factor" in harness.text
    assert "the element size must be a positive number of mm" in harness.text
    assert [s.refinement for s in harness.meshed] == ["medium"]      # never re-meshed
    assert len(harness.runs) == 1


def test_a_large_mesh_is_priced_and_confirmed_at_the_mesh_question(harness, monkeypatch) -> None:
    monkeypatch.setattr(sr, "large_dof_threshold", lambda: 5000)
    code = harness.main([harness.part, "--out", harness.out, "--no-open", "--no-screen"], [
        "1", "1", "", "2", "500 N -z", "", "",
        "fine", "n",              # priced before meshing; no -> pick again
        "fine", "y", "",          # accepted once, then proceed
    ])
    assert code == 0, harness.text
    priced = [p for p in harness.prompts if "Use it?" in p]
    assert len(priced) == 2 and "above 5,000" in priced[0] and "several GB" in priced[0]
    assert not any("Solve anyway?" in p for p in harness.prompts)    # not asked twice
    assert harness.runs[-1]["allow_large"] is True
    assert harness.case.mesh.refinement == "fine"


def test_declining_a_large_solve_says_how_to_rerun(harness, monkeypatch) -> None:
    monkeypatch.setattr(sr, "large_dof_threshold", lambda: 10)
    code = harness.main([harness.part, "--fixed", "1", "--force", "2:0,0,-5", "--no-open",
                         "--out", harness.out], ["", "n"])
    assert code == 0 and harness.runs == []
    assert "Not solved: 3,000 DOF is above 10" in harness.text and "--save-case" in harness.text
    assert "refused" not in harness.text


def test_the_face_table_and_picture_come_before_the_questions(harness) -> None:
    harness.main([harness.part, "--out", harness.out, "--no-open"], [])
    text = harness.text
    assert "[1/6] Reading geometry" in text and "[2/6] Meshing" in text
    assert "3,000 DOF" in text and "about 1.5 s" in text
    assert "outward normal" in text and "(-1.00, +0.00, +0.00) -X" in text
    assert (harness.out / "faces.png").is_file()
    assert text.index("outward normal") < text.index("Material")


# --------------------------------------------------------------------------
# flags and case files
# --------------------------------------------------------------------------


def test_flags_build_the_case_without_questions(harness) -> None:
    code = harness.main([
        harness.part, "--fixed", "1", "--force", "2:500:-z", "--force", "4:0,100,0",
        "--pressure", "6:200 kPa", "--material", "aluminium", "--gravity",
        "--family", "Bracket", "--mesh", "coarse", "--yes", "--no-open", "--out", harness.out,
    ])
    assert code == 0, harness.text
    assert harness.prompts == []
    case = harness.case
    assert case.fixed_faces() == (1,)
    assert [type(l).__name__ for l in case.loads] == ["ForceLoad", "ForceLoad", "PressureLoad"]
    assert case.loads[0].vector_n == pytest.approx((0, 0, -500.0))
    assert case.loads[1].vector_n == pytest.approx((0, 100.0, 0))
    assert case.loads[2].pressure_mpa == pytest.approx(0.2)
    assert case.material == MATERIAL_PRESETS["aluminum_alloy"]
    assert case.gravity == Gravity()
    assert case.mesh.refinement == "coarse"
    assert harness.meshed[0].refinement == "coarse"
    assert harness.runs[-1]["family"] == "bracket"


def test_flags_without_material_default_to_structural_steel_and_say_so(harness) -> None:
    code = harness.main([harness.part, "--fixed", "1", "--force", "2:0,0,-500", "--yes",
                         "--no-open", "--out", harness.out])
    assert code == 0
    assert harness.case.material == MATERIAL_PRESETS["structural_steel"]
    assert "Structural Steel (the Ansys default)" in harness.text


def test_custom_material_flag(harness) -> None:
    harness.main([harness.part, "--fixed", "1", "--force", "2:0,0,-500", "--material",
                  "70 GPa,0.33,2700", "--yes", "--no-open", "--out", harness.out])
    material = harness.case.material
    assert (material.youngs_modulus_mpa, material.poisson_ratio, material.density_kg_m3) == (
        pytest.approx(70000.0), pytest.approx(0.33), pytest.approx(2700.0))


@pytest.mark.parametrize("argv, fragment", [
    (["--fixed", "1", "--force", "2:500", "--yes"], "has a size but no direction"),
    (["--fixed", "1", "--force", "nope", "--yes"], "--force 'nope'"),
    (["--fixed", "9", "--force", "2:0,0,-5", "--yes"], "does not have"),
    (["--fixed", "1", "--pressure", "6:0", "--yes"], "must not be zero"),
    (["--fixed", "1", "--force", "2:0,0,-5", "--material", "unobtainium", "--yes"], "no material preset"),
    (["--fixed", "1", "--yes"], "--yes needs a complete case"),
    (["--yes"], "--fixed FACES"),
    (["--fixed", "1", "--force", "1:0,0,-5", "--yes"], "acts only on fixed faces"),
])
def test_bad_flags_exit_2_with_the_reason(harness, argv, fragment) -> None:
    code = harness.main([harness.part, *argv, "--no-open", "--out", harness.out])
    assert code == 2, harness.text
    assert fragment in harness.text
    assert harness.runs == []


@pytest.mark.parametrize("argv, fragment", [
    (["--fixed", "1", "--force", "2:0,0,-5", "--material", "unobtainium", "--yes"],
     "--material 70000,0.33"),
    (["--fixed", "1", "--yes"], "--yes needs a complete case"),
    (["--fixed", "1", "--force", "2:0,0,-5", "--material", "70 GPa,0.33", "--gravity", "--yes"],
     "--material '70 GPa,0.33,2700'"),
    (["--fixed", "1", "--force", "2:0,0,-5", "--family", "spring", "--yes"], "cannot be screened"),
    (["--fixed", "1", "--force", "2:0,0,-5", "--element-size", "0", "--yes"],
     "--element-size must be a positive number of mm"),
])
def test_mistakes_that_need_no_faces_fail_before_the_part_is_read(harness, argv, fragment) -> None:
    """T14: a typo costs a moment, not a mesh."""
    code = harness.main([harness.part, *argv, "--no-open", "--out", harness.out])
    assert code == 2, harness.text
    assert fragment in harness.text
    assert harness.meshed == [] and "Reading geometry" not in harness.text


def test_a_broken_case_file_fails_before_the_part_is_read(harness) -> None:
    file = harness.tmp / "case.json"
    file.write_text(json.dumps({"supports": [{"type": "fixed", "faces": [1]}],
                                "loads": [{"type": "force", "faces": [2], "magnitude_n": 5,
                                           "direction": "normal"}],
                                "material": {"preset": "steel"}, "colour": "red"}))
    code = harness.main([harness.part, "--case", file, "--yes", "--no-open", "--out", harness.out])
    assert code == 2 and "unexpected ['colour']" in harness.text
    assert harness.meshed == []
    # "normal" is resolved against the faces later, so it is not refused early
    file.write_text(json.dumps({"supports": [{"type": "fixed", "faces": [1]}],
                                "loads": [{"type": "force", "faces": [2], "magnitude_n": 5,
                                           "direction": "normal"}],
                                "material": {"preset": "steel"}}))
    code = harness.main([harness.part, "--case", file, "--yes", "--no-open", "--out", harness.out])
    assert code == 0, harness.text
    assert harness.case.loads[0].vector_n == pytest.approx((-5.0, 0.0, 0.0))


def test_outputs_that_cannot_be_written_fail_before_any_work(harness) -> None:
    """T8: exit 2 with the path, before reading or meshing; not 'solve failed'."""
    blocker = harness.tmp / "a_file"
    blocker.write_text("x")
    code = harness.main([harness.part, "--fixed", "1", "--force", "2:0,0,-5", "--yes",
                         "--no-open", "--out", blocker / "out"])
    assert code == 2 and "cannot write the results to" in harness.text
    assert harness.meshed == [] and "solve failed" not in harness.text

    code = harness.main([harness.part, "--fixed", "1", "--force", "2:0,0,-5", "--yes",
                         "--no-open", "--out", harness.out, "--save-case", harness.tmp])
    assert code == 2 and "is a folder" in harness.text and harness.meshed == []

    target = harness.tmp / "new" / "folder" / "case.json"     # parents are made
    code = harness.main([harness.part, "--fixed", "1", "--force", "2:0,0,-5", "--yes",
                         "--no-open", "--out", harness.out, "--save-case", target])
    assert code == 0 and target.is_file()


def test_a_write_failure_during_the_run_is_an_input_error(monkeypatch, tmp_path) -> None:
    h = Harness(monkeypatch, tmp_path, run_error=PermissionError(13, "Permission denied", "/x/y"))
    code = h.main([h.part, "--fixed", "1", "--force", "2:0,0,-5", "--yes", "--no-open", "--out", h.out])
    assert code == 2 and "error: cannot write /x/y: Permission denied" in h.text


def test_messages_speak_the_language_of_where_they_are_asked(harness) -> None:
    """T11: the interview says 'type c', a yield is not a pressure."""
    code = harness.main([harness.part, "--out", harness.out, "--no-open", "--no-screen"], [
        "abc",                                # not a material
        "c", "70 GPa", "0.33", "", "NaN", "250", "",
        "1", "", "2", "500 N -z", "", "", "", "",
    ])
    assert code == 0, harness.text
    text = harness.text
    assert "c to type in a custom material" in text and "youngs_modulus_mpa" not in text
    assert "tensile yield: 'NaN' is not a finite number" in text and "pressure:" not in text


def test_the_family_list_is_the_screenable_families(harness) -> None:
    """T13: one list for the prompt and the error; the spring does not transfer."""
    names = cli.screenable_families()
    assert "bracket" in names and "spring" not in names
    code = harness.main([harness.part, "--out", harness.out, "--no-open"], [
        "1", "1", "", "2", "0,100,0", "", "",
        "foo", "", "", "",
    ])
    assert code == 0, harness.text
    prompt = [p for p in harness.prompts if p.startswith("Surrogate family")][0]
    assert f"({', '.join(names)}, none)" in prompt
    assert f"choose one of {', '.join(names)}, or none" in harness.text


def test_crosscheck_is_asked_for(harness) -> None:
    """T4: the P1 crosscheck costs more than screening and solving; opt in."""
    base = [harness.part, "--fixed", "1", "--force", "2:0,0,-5", "--yes", "--no-open",
            "--out", harness.out]
    assert harness.main(base) == 0 and harness.runs[-1]["crosscheck"] is False
    assert harness.main([*base, "--crosscheck"]) == 0 and harness.runs[-1]["crosscheck"] is True


def test_a_console_that_cannot_encode_the_tables_gets_ascii(harness, monkeypatch) -> None:
    """R1: Windows redirects stdout through cp1252, which has no box drawing."""
    import io

    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252", newline="\n")
    monkeypatch.setattr(sys, "stdout", stream)
    code = cli.main([str(harness.part), "--fixed", "1", "--force", "2:0,0,-5", "--yes",
                     "--no-open", "--out", str(harness.out)])
    stream.flush()
    text = raw.getvalue().decode("cp1252")
    assert code == 0, text
    assert "+- A: Static Structural - check before solving" in text and "| Part" in text
    assert "area mm^2" in text


def test_the_face_table_drops_round_off(tmp_path) -> None:
    """T10: '-3.913e-05' and '-5.546e-14' on a 100 mm part are zeros."""
    mesh = box_mesh()
    face = next(f for f in mesh.faces if f.tag == 1)
    face.centroid = (-3.913e-05, 10.0, -1.295e-05)
    face.bbox_min = (-5.546e-14, 0.0, -6.562e-08)
    row = next(line for line in cli.face_table(mesh) if line.split()[0] == "1")
    assert "e-" not in row
    assert "(0, 10, 0)" in row and "x 0..0" in row and "z 0..10" in row


def test_a_case_file_round_trips(harness) -> None:
    case = StaticStructuralCase(
        supports=(FixedSupport(faces=(1,)),),
        loads=(ForceLoad(faces=(2,), vector_n=(0, 0, -750.0)),
               PressureLoad(faces=(6,), pressure_mpa=-0.1)),
        material=MATERIAL_PRESETS["titanium_alloy"],
        mesh=MeshSettings(refinement="fine"),
    )
    file = harness.tmp / "case.json"
    file.write_text(json.dumps(case.as_document()))
    code = harness.main([harness.part, "--case", file, "--yes", "--no-open", "--out", harness.out])
    assert code == 0, harness.text
    assert harness.case == case
    assert harness.meshed[0].refinement == "fine"


def test_a_case_file_and_case_flags_together_are_refused(harness) -> None:
    file = harness.tmp / "case.json"
    file.write_text("{}")
    code = harness.main([harness.part, "--case", file, "--fixed", "1", "--yes", "--out", harness.out])
    assert code == 2 and "not both" in harness.text


def test_a_broken_case_file_exits_2(harness) -> None:
    file = harness.tmp / "case.json"
    file.write_text("{not json")
    assert harness.main([harness.part, "--case", file, "--yes", "--out", harness.out]) == 2
    assert "not valid JSON" in harness.text


def test_save_case_writes_a_replayable_file(harness) -> None:
    target = harness.tmp / "saved.json"
    harness.main([harness.part, "--fixed", "1", "--force", "2:0,0,-500", "--yes", "--no-open",
                  "--out", harness.out, "--save-case", target])
    saved = json.loads(target.read_text())
    assert saved["schema_version"] == "StaticStructuralCase/v1"
    assert saved["loads"][0]["vector_n"] == [0, 0, -500.0]


def test_json_mode_prints_only_the_results_document(harness, capsys) -> None:
    code = harness.main([harness.part, "--fixed", "1", "--force", "2:0,0,-500", "--yes",
                         "--no-open", "--out", harness.out, "--json"])
    assert code == 0
    assert len(harness.printed) == 1
    assert json.loads(harness.printed[0])["schema_version"] == "FeaStaticStructural/v1"
    assert "Reading geometry" in capsys.readouterr().err


# --------------------------------------------------------------------------
# exit codes
# --------------------------------------------------------------------------


def test_an_unreadable_format_exits_2(harness) -> None:
    part = harness.tmp / "part.sldprt"
    part.write_text("x")
    assert harness.main([part, "--yes"]) == 2
    assert "Export the part as STEP" in harness.text


def test_missing_extras_exit_2_with_the_install_hint(harness, monkeypatch) -> None:
    monkeypatch.setattr(cli, "_missing_extras", lambda modules: ["gmsh"])
    code = harness.main([harness.part, "--fixed", "1", "--force", "2:0,0,-5", "--yes"])
    assert code == 2
    assert "missing gmsh" in harness.text and "uv pip install" in harness.text


def test_a_refusal_exits_3(monkeypatch, tmp_path) -> None:
    h = Harness(monkeypatch, tmp_path, run_error=sr.RunRefused("too large; pass --allow-large"))
    code = h.main([h.part, "--fixed", "1", "--force", "2:0,0,-5", "--yes", "--no-open", "--out", h.out])
    assert code == 3 and "refused: too large" in h.text


def test_a_solver_failure_exits_4_without_a_traceback(monkeypatch, tmp_path) -> None:
    h = Harness(monkeypatch, tmp_path, run_error=RuntimeError("singular stiffness"))
    code = h.main([h.part, "--fixed", "1", "--force", "2:0,0,-5", "--yes", "--no-open", "--out", h.out])
    assert code == 4
    assert "solve failed: RuntimeError: singular stiffness" in h.text
    assert "Traceback" not in h.text


def test_a_large_mesh_is_asked_about_interactively(harness, monkeypatch) -> None:
    monkeypatch.setattr(sr, "large_dof_threshold", lambda: 10)
    code = harness.main([harness.part, "--fixed", "1", "--force", "2:0,0,-5", "--no-open",
                         "--out", harness.out], ["", "y"])
    assert code == 0
    assert any("Solve anyway?" in p for p in harness.prompts)
    assert harness.runs[-1]["allow_large"] is True


# --------------------------------------------------------------------------
# pieces
# --------------------------------------------------------------------------


def test_default_output_is_under_the_working_directory_not_beside_the_step(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    out = sr.default_out_dir(Path("/somewhere/else/bracket.step"), "20261028-120000")
    assert out == tmp_path / "fea_results" / "bracket-20261028-120000"


def test_surrogate_files_flags_then_env_then_repo(tmp_path, monkeypatch) -> None:
    models = tmp_path / "models"
    models.mkdir()
    for name in ("sweep_gnn.pt", "calibration.json"):
        (models / name).write_text("x")
    monkeypatch.setattr(sr, "_REPO_MODELS", models)
    for env in (sr.WEIGHTS_ENV, sr.CALIBRATION_ENV, sr.NEIGHBOURS_ENV):
        monkeypatch.delenv(env, raising=False)

    weights, calibration, neighbours = sr.find_surrogate_files()
    assert weights == models / "sweep_gnn.pt" and calibration == models / "calibration.json"
    assert neighbours is None                       # absent, so not offered

    monkeypatch.setenv(sr.WEIGHTS_ENV, str(tmp_path / "env.pt"))
    assert sr.find_surrogate_files()[0] == tmp_path / "env.pt"
    assert sr.find_surrogate_files(weights=tmp_path / "flag.pt")[0] == tmp_path / "flag.pt"


def test_the_calibration_supplies_the_gate_domain_without_torch(tmp_path) -> None:
    file = tmp_path / "calibration.json"
    file.write_text(json.dumps({"domain": {"poisson_ratio": {"min": 0.27, "max": 0.42}},
                                "calibration": {"set_sha256": "abc"}}))
    domain, sha = sr.read_calibration(file)
    assert domain["poisson_ratio"]["max"] == 0.42 and sha == "abc"
    assert sr.read_calibration(None) == (None, None)


def test_the_box_is_rectangular() -> None:
    lines = cli.box("Title", [("Part", "x" * 200), ("", "short")], width=60)
    assert {len(line) for line in lines} == {60}


def test_face_table_lists_every_face_with_axis_labels() -> None:
    lines = cli.face_table(box_mesh())
    assert len(lines) == 2 + 6
    assert any(line.split()[0] == "5" and "-Z" in line for line in lines[2:])


# --------------------------------------------------------------------------
# results table and document, from a stand-in solution
# --------------------------------------------------------------------------


def _hotspot(value, *, singular=False, reason=None, faces=(2,)):
    return SimpleNamespace(value=value, node=7, location_mm=(100.0, 10.0, 10.0), faces=faces,
                           singular=singular, reason=reason)


def _run(screening: sr.ScreeningOutcome, *, tmp_path: Path, yield_mpa=250.0,
         extrapolated=None, singular=True, order=2) -> sr.StructuralRun:
    material = MATERIAL_PRESETS["structural_steel"]
    if yield_mpa is None:
        material = MATERIAL_PRESETS["gray_cast_iron"]
    case = StaticStructuralCase(
        supports=(FixedSupport(faces=(1,)),),
        loads=(ForceLoad(faces=(2,), vector_n=(0, 0, -1000.0)),), material=material,
    )
    solution = SimpleNamespace(
        max_stress=_hotspot(700.0, singular=singular,
                            reason="edge of A Fixed Support (face 1)" if singular else None,
                            faces=(1, 5)),
        min_stress=_hotspot(0.01),
        max_stress_away=_hotspot(np.float64(310.0)),
        max_stress_extrapolated=extrapolated,
        exclusion_radius_mm=10.0,
        max_deformation=_hotspot(2.01), min_deformation=_hotspot(0.0, faces=(1,)),
        deformation_scale_auto=2.5,
        reactions=(SimpleNamespace(label="Fixed Support", faces=(1,), force_n=(0.0, 0.0, 1000.0)),),
        applied_total_n=(0.0, 0.0, -1000.0), reaction_total_n=(0.0, 0.0, 1000.0),
        equilibrium_error=1e-12,
    )
    mesh = box_mesh(settings=MeshSettings(element_order=order))
    files = {"case": tmp_path / "case.json", "results": tmp_path / "results.json"}
    return sr.StructuralRun(
        path=Path("/parts/beam.step"), out_dir=tmp_path, case=case, mesh=mesh, solution=solution,
        screening=screening, files=files, timings={"mesh": 0.5, "solve": 1.0, "total": 2.0},
        convergence=None, part={"path": "/parts/beam.step", "healed": False},
    )


def test_the_results_table_has_every_group_and_never_says_estimate(tmp_path) -> None:
    outcome = sr.ScreeningOutcome("not_screenable", ("no surrogate family named",), None)
    text = " ".join(sr.format_summary(_run(outcome, tmp_path=tmp_path)).split())
    for heading in ("EQUIVALENT (VON-MISES) STRESS", "TOTAL DEFORMATION", "REACTIONS",
                    "YIELD CHECK", "SURROGATE SCREENING (estimate, not an analysis)",
                    "MESH & TIME", "FILES"):
        assert heading in text
    assert "ESTIMATE" not in text and "NOT AN ANALYSIS" not in text
    assert "700 MPa" in text and "singular" in text and "edge of A Fixed Support" in text
    assert "Max away from singularities" in text and "310 MPa" in text
    assert "A Fixed Support" in text and "|R| 1000 N" in text
    # No hot-spot value (too coarse to extrapolate): the singular max is the
    # conservative headline, the value away from it the optimistic end.
    assert "Stress used 700 MPa singular max" in text and "Safety factor 0.357" in text
    assert "Away from singularities 0.806 optimistic bound" in text
    assert "- no surrogate family named" in text


def test_the_yield_check_never_uses_less_than_a_stress_the_part_carries(tmp_path) -> None:
    """mech-1: a singular max is replaced by the hot-spot value at the edge, but
    never by anything below the max away from singularities."""
    outcome = sr.ScreeningOutcome("skipped", (), None)
    check = _run(outcome, tmp_path=tmp_path, extrapolated=330.0).yield_check()
    assert check["stress_used_mpa"] == 330.0 and "hot-spot" in check["basis"]
    assert check["safety_factor"] == pytest.approx(250.0 / 330.0)
    assert check["safety_factor_at_singular_max"] == pytest.approx(250.0 / 700.0)
    assert check["safety_factor_away_from_singularities"] is None

    check = _run(outcome, tmp_path=tmp_path, extrapolated=290.0).yield_check()
    assert check["stress_used_mpa"] == 310.0 and "max away" in check["basis"]

    check = _run(outcome, tmp_path=tmp_path, singular=False).yield_check()
    assert check["stress_used_mpa"] == 700.0 and check["safety_factor_at_singular_max"] is None


def test_a_four_node_run_gets_no_safety_factor_and_says_why(tmp_path) -> None:
    outcome = sr.ScreeningOutcome("skipped", (), None)
    run = _run(outcome, tmp_path=tmp_path, order=1)
    check = run.yield_check()
    assert check["safety_factor"] is None and check["stress_used_mpa"] is None
    text = " ".join(sr.format_summary(run).split())
    assert "WARNING: 4-node tetrahedra are stiff in bending" in text
    assert "Safety factor - not given" in text


def test_no_yield_means_no_safety_factor(tmp_path) -> None:
    outcome = sr.ScreeningOutcome("skipped", ("screening switched off (--no-screen)",), None)
    run = _run(outcome, tmp_path=tmp_path, yield_mpa=None)
    assert "no safety factor" in sr.format_summary(run)
    assert run.as_document()["results"]["yield_ratio"] is None


def test_a_screened_run_nests_the_document_and_shows_the_p1_crosscheck(tmp_path) -> None:
    document = {
        "schema_version": "SurrogateScreening/v1", "coverage": 0.9,
        "predictions": [{"quantity": "von_mises_critical_distance", "lower": 165.2,
                         "upper": 313.3, "units": "MPa"}],
        "limitations": ["screening estimate, not a strength certification"],
    }
    crosscheck = {"available": True, "inside": True, "solved": 208.6,
                  "p1_critical_distance_mpa": 208.6}
    outcome = sr.ScreeningOutcome("screened", (), "bracket", document, crosscheck, 1.2)
    run = _run(outcome, tmp_path=tmp_path, extrapolated=330.0)
    text = " ".join(sr.format_summary(run).split())       # prose wraps; compare words
    assert "165.2 - 313.3 MPa" in text and "critical-distance" in text
    assert "208.6 MPa" in text and "inside the interval" in text
    assert "not the design stress" in text

    doc = run.as_document()
    json.dumps(doc, allow_nan=False)
    assert doc["schema_version"] == "FeaStaticStructural/v1"
    assert doc["attestation"] == "computed_analysis"
    assert doc["screening"]["document"] == document            # untouched, nested
    assert "predictions" not in doc["results"]
    stress = doc["results"]["equivalent_stress"]
    assert stress["max"] == 700.0 and stress["max_singular"] is True
    assert stress["max_away_from_singularities"] == 310.0
    assert stress["max_extrapolated_to_singularity"] == 330.0
    assert doc["results"]["yield_ratio"]["stress_used_mpa"] == 330.0
    assert doc["files"]["results"] == "results.json"
    assert doc["idealisation"][1] == "10-node quadratic tetrahedra"


def test_the_document_never_carries_a_non_finite_number(tmp_path) -> None:
    outcome = sr.ScreeningOutcome("skipped", (), None)
    run = _run(outcome, tmp_path=tmp_path)
    run.solution.min_stress.value = float("nan")
    doc = run.as_document()
    assert doc["results"]["equivalent_stress"]["min"] is None
    json.dumps(doc, allow_nan=False)


def _bracket_case() -> StaticStructuralCase:
    return StaticStructuralCase(
        supports=(FixedSupport(faces=(1,)),),
        loads=(ForceLoad(faces=(2,), vector_n=(0, 0, -500.0)),),
        material=MATERIAL_PRESETS["structural_steel"],
    )


def test_screening_reports_every_gate_reason_and_never_loads_the_model(tmp_path) -> None:
    outcome, log = sr._screen(
        tmp_path / "part.step", _bracket_case(), box_mesh(), None, tmp_path,
        family=None, weights=None, calibration=None, neighbours=None,
    )
    assert outcome.status == "not_screenable"
    assert any("no surrogate family" in r for r in outcome.reasons)
    assert outcome.document is None and log is None


def test_screening_without_weights_is_unavailable_and_says_how_to_fix_it(tmp_path) -> None:
    outcome, _ = sr._screen(
        tmp_path / "part.step", _bracket_case(), box_mesh(), None, tmp_path,
        family="bracket", weights=None, calibration=None, neighbours=None,
    )
    assert outcome.status == "unavailable"
    assert "--weights" in outcome.reasons[0] and sr.WEIGHTS_ENV in outcome.reasons[0]

    outcome, _ = sr._screen(
        tmp_path / "part.step", _bracket_case(), box_mesh(), None, tmp_path,
        family="bracket", weights=tmp_path / "missing.pt",
        calibration=tmp_path / "missing.json", neighbours=None,
    )
    assert outcome.status == "unavailable" and "not found" in outcome.reasons[0]


# --------------------------------------------------------------------------
# heavy: one real run, and the demo
# --------------------------------------------------------------------------


def test_gmsh_leaves_ctrl_c_to_python() -> None:
    """T9/R5: gmsh's initialize() sets SIGINT to SIG_DFL and finalize() never
    restores it, so Ctrl-C killed the CLI with no 'Interrupted' and no exit 130."""
    import signal

    gmsh = pytest.importorskip("gmsh", reason="meshing requires the inference extra")
    from fea_surrogate.cad_intake import keep_sigint, start_gmsh

    before = signal.getsignal(signal.SIGINT)
    start_gmsh()
    gmsh.finalize()
    assert signal.getsignal(signal.SIGINT) is before
    with keep_sigint():
        gmsh.initialize()           # the frozen training mesher still does this
        gmsh.finalize()
        assert signal.getsignal(signal.SIGINT) is signal.SIG_DFL
    assert signal.getsignal(signal.SIGINT) is before


@pytest.fixture(scope="module")
def solved_beam(tmp_path_factory):
    gmsh = pytest.importorskip("gmsh", reason="meshing requires the inference extra")
    pytest.importorskip("scipy", reason="solving requires the inference extra")
    pytest.importorskip("matplotlib", reason="images require the render extra")
    from fea_surrogate.cad_intake import inspect

    root = tmp_path_factory.mktemp("cli")
    step = root / "beam.step"
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("beam")
        gmsh.model.occ.addBox(0.0, 0.0, 0.0, 100.0, 10.0, 10.0)
        gmsh.model.occ.synchronize()
        gmsh.write(str(step))
    finally:
        gmsh.finalize()
    solid = inspect(step)
    held = [f.tag for f in solid.faces if f.normal[0] < -0.9]
    pushed = [f.tag for f in solid.faces if f.normal[0] > 0.9]
    out = root / "out"
    printed: list[str] = []
    code = cli.main(
        [str(step), "--fixed", ",".join(map(str, held)),
         "--force", f"{pushed[0]}:0,0,-1000", "--material", "steel",
         "--no-screen", "--no-open", "--yes", "--json", "--out", str(out)],
        input_fn=lambda prompt: (_ for _ in ()).throw(AssertionError(f"asked {prompt!r}")),
        print_fn=lambda *a, **k: printed.append(" ".join(map(str, a))),
    )
    return {"code": code, "out": out, "printed": printed}


def test_a_non_interactive_run_writes_every_file(solved_beam) -> None:
    assert solved_beam["code"] == 0, solved_beam["printed"]
    out = solved_beam["out"]
    for name in ("case.json", "faces.png", "setup.png", "equivalent_stress.png",
                 "total_deformation.png", "board.png", "results.vtu", "results.json",
                 "summary.txt"):
        assert (out / name).is_file(), name
    summary = (out / "summary.txt").read_text()
    assert "EQUIVALENT (VON-MISES) STRESS" in summary and "ESTIMATE" not in summary


def test_the_numbers_are_those_of_a_cantilever(solved_beam) -> None:
    document = json.loads(solved_beam["printed"][-1])
    assert document == json.loads((solved_beam["out"] / "results.json").read_text())
    results = document["results"]
    # 100 x 10 x 10 steel cantilever, 1000 N: Euler-Bernoulli tip 2.000 mm,
    # root bending stress 600 MPa (the clamp edge itself is singular).
    assert results["total_deformation"]["max"] == pytest.approx(2.0, rel=0.08)
    assert results["equivalent_stress"]["max"] > 400.0
    assert results["equilibrium_error"] < 1e-6
    applied = results["applied_total_n"]
    reaction = results["reaction_total_n"]
    assert applied == pytest.approx([0.0, 0.0, -1000.0], abs=1e-6)
    assert reaction == pytest.approx([0.0, 0.0, 1000.0], rel=1e-6, abs=1e-6)
    assert document["screening"]["status"] == "skipped"
    assert document["mesh"]["order"] == 2


def test_the_demo_writes_a_bracket_and_solves_it(tmp_path) -> None:
    pytest.importorskip("gmsh", reason="meshing requires the inference extra")
    pytest.importorskip("scipy", reason="solving requires the inference extra")
    pytest.importorskip("matplotlib", reason="images require the render extra")
    out = tmp_path / "demo"
    printed: list[str] = []
    code = cli.main(["demo", "--out", str(out), "--no-open", "--no-screen"],
                    print_fn=lambda *a, **k: printed.append(" ".join(map(str, a))))
    assert code == 0, "\n".join(printed)
    assert (out / "demo_bracket.step").is_file()
    document = json.loads((out / "results.json").read_text())
    assert document["case"]["loads"][0]["vector_n"] == pytest.approx([0, 0, -1000.0])
    assert document["case"]["material"]["preset"] == "structural_steel"
    assert (out / "equivalent_stress.png").is_file()


def test_a_screenable_case_is_screened_and_crosschecked_with_the_p1_solver(tmp_path) -> None:
    """The whole screening branch with an untrained model: the numbers mean
    nothing, the plumbing -- gate, screen_step on the family preset, the P1
    label-solver crosscheck, the nested untouched document, the log -- is what
    is checked."""
    pytest.importorskip("gmsh", reason="meshing requires the inference extra")
    pytest.importorskip("scipy", reason="solving requires the inference extra")
    torch = pytest.importorskip("torch", reason="screening requires the inference extra")
    pytest.importorskip("torch_geometric", reason="screening requires the inference extra")
    from fea_surrogate.model import PeakGNN
    from fea_surrogate.screenability import RANGES_MEASURED_ON_SET
    from fea_surrogate.screening import validate_document

    step = cli.write_demo_bracket(tmp_path / "bracket.step")
    torch.manual_seed(0)
    torch.save(PeakGNN(13).state_dict(), tmp_path / "weights.pt")
    (tmp_path / "calibration.json").write_text(json.dumps({
        "model": {"name": "sweep-mesh-gnn", "version": "0.4.0", "sha256": "0" * 64},
        "coverage": 0.9,
        "calibration": {
            "method": "mondrian_split_conformal_log_residual", "set_size": 600,
            "set_sha256": RANGES_MEASURED_ON_SET, "log_half_width": math.log(1.42),
            "delta": 0.0,
            "bands": {"bracket": {"log_half_width": math.log(1.377), "set_size": 100}},
        },
        "normaliser": {"log_mean": 5.91, "log_std": 2.85, "delta": 0.0},
        "target": {"quantity": "von_mises_critical_distance"},
        "features": {"version": "sweep/v2", "columns": 13},
        "domain": {"material_class": "linear_elastic",
                   "poisson_ratio": {"min": 0.27, "max": 0.42}},
        "limitations": ["screening estimate, not a strength certification"],
    }))
    mesh = sr.mesh_part(step, MeshSettings(refinement="coarse"))
    case = dataclasses_replace(cli.demo_case(mesh), mesh=MeshSettings(refinement="coarse"))
    run = sr.run_static_structural(
        step, case, out_dir=tmp_path / "out", family="bracket", mesh=mesh,
        weights=tmp_path / "weights.pt", calibration=tmp_path / "calibration.json",
        render=False, export_vtu=False, crosscheck=True,
    )

    assert run.screening.status == "screened", run.screening.reasons
    # Loading torch and the weights is timed apart from the screen itself.
    assert "load_model" in run.timings and run.timings["load_model"] > 0
    validate_document(dict(run.screening.document))
    crosscheck = run.screening.crosscheck
    assert crosscheck["available"] is True
    assert crosscheck["p1_critical_distance_mpa"] > 0
    assert isinstance(crosscheck["inside"], bool)
    # the preset was used, so no "supplied by the caller" limitation was added
    assert not any("supplied by the caller" in l for l in run.screening.document["limitations"])
    logged = (tmp_path / "out" / "crosschecks.jsonl").read_text().splitlines()
    assert len(logged) == 1 and json.loads(logged[0])["available"] is True
    document = run.as_document()
    assert document["screening"]["document"]["attestation"] == "screening_estimate"
    assert document["attestation"] == "computed_analysis"

    # Without --crosscheck the label solver is not run (it costs more than the
    # screen and the solve together), the summary says how to ask for it, and
    # the loaded model is reused.
    again = sr.run_static_structural(
        step, case, out_dir=tmp_path / "again", family="bracket", mesh=mesh,
        weights=tmp_path / "weights.pt", calibration=tmp_path / "calibration.json",
        render=False, export_vtu=False,
    )
    assert again.screening.status == "screened"
    assert again.screening.crosscheck["available"] is False
    assert "--crosscheck" in again.screening.crosscheck["skipped"]
    assert not (tmp_path / "again" / "crosschecks.jsonl").exists()
    assert again.screening.load_seconds < run.screening.load_seconds
    assert "--crosscheck" in " ".join(sr.format_summary(again).split())
