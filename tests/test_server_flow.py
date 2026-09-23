"""The guided flow over MCP: inspect, state a load case, screen.

The failures worth catching here are the quiet ones -- a load case accepted
without held faces, a family whose shipped weights do not transfer being screened
anyway, a CAD format refused three calls deep instead of at the door.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

gmsh = pytest.importorskip("gmsh", reason="meshing requires the inference extra")
torch = pytest.importorskip("torch", reason="the server loads a model")
pytest.importorskip("scipy", reason="the families require the inference extra")

from fea_surrogate.model import PeakGNN  # noqa: E402
from fea_surrogate.predict import UNTRANSFERABLE_FAMILIES, Surrogate  # noqa: E402
from fea_surrogate.screening import Calibration, file_sha256  # noqa: E402
from fea_surrogate.server import create_mcp  # noqa: E402
from fea_surrogate.predict import Normaliser  # noqa: E402


def _box(tmp_path, dx=100.0, dy=40.0, dz=12.0):
    path = tmp_path / "box.step"
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("box")
        gmsh.model.occ.addBox(0.0, 0.0, 0.0, dx, dy, dz)
        gmsh.model.occ.synchronize()
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
    return path


def _surrogate(tmp_path) -> Surrogate:
    """An untrained model. These tests are about the flow, not the numbers."""
    weights = tmp_path / "w.pt"
    torch.save(PeakGNN(13).state_dict(), weights)
    return Surrogate(
        model=PeakGNN(13).eval(),
        calibration=Calibration(
            method="mondrian_split_conformal_log_residual",
            set_size=100,
            set_sha256="c" * 64,
            coverage=0.9,
            log_half_width=0.35,
            delta=0.0,
            bands={"bracket": {"log_half_width": 0.32, "set_size": 100}},
        ),
        normaliser=Normaliser(log_mean=5.9, log_std=2.8, delta=0.0),
        weights_sha256=file_sha256(weights),
        domain={"material_class": "linear_elastic"},
        limitations=["screening estimate, not a strength certification"],
    )


def _call(mcp, name, args):
    async def run():
        result = await mcp.call_tool(name, args)
        first = result[0]
        text = first.text if hasattr(first, "text") else first[0].text
        return json.loads(text)

    return asyncio.run(run())


def _case(fixed, loaded, **over):
    document = {
        "fixed": {"faces": fixed},
        "load": {"faces": loaded, "magnitude_n": 5000.0, "direction": [0, 0, -1]},
    }
    document.update(over)
    return json.dumps(document)


def test_inspecting_a_part_lists_its_faces_and_proposes_without_deciding(tmp_path):
    mcp = create_mcp(surrogate=_surrogate(tmp_path))

    out = _call(mcp, "surrogate_inspect_cad", {"path": str(_box(tmp_path))})

    assert len(out["faces"]) == 6
    assert out["proposed"]["fixed"]["faces"]
    assert out["why"]["fixed"].startswith("face ")
    # It must never invent a load magnitude.
    assert "magnitude_n" not in out["proposed"]["load"]
    assert any("magnitude_n" in q for q in out["still_needed"])


def test_a_stated_load_case_screens_and_the_document_is_admissible(tmp_path):
    mcp = create_mcp(surrogate=_surrogate(tmp_path))
    path = str(_box(tmp_path))
    faces = _call(mcp, "surrogate_inspect_cad", {"path": path})["faces"]
    held = [f["tag"] for f in faces if f["normal"][0] < -0.9]
    pushed = [f["tag"] for f in faces if f["normal"][0] > 0.9]

    out = _call(
        mcp,
        "surrogate_screen",
        {"path": path, "load_case_json": _case(held, pushed), "family": "bracket"},
    )

    assert out["schema_version"] == "SurrogateScreening/v1"
    assert out["attestation"] == "screening_estimate"
    assert out["predictions"][0]["lower"] <= out["predictions"][0]["upper"]
    assert out["coverage"] == 0.9


def test_a_load_case_with_no_held_faces_is_refused(tmp_path):
    """An unconstrained body has rigid-body freedom and no stress solution."""
    mcp = create_mcp(surrogate=_surrogate(tmp_path))

    with pytest.raises(Exception, match="held faces"):
        _call(
            mcp,
            "surrogate_screen",
            {
                "path": str(_box(tmp_path)),
                "load_case_json": json.dumps(
                    {"load": {"faces": [1], "magnitude_n": 10.0}}
                ),
            },
        )


def test_a_family_whose_weights_do_not_transfer_is_refused(tmp_path):
    """Better to report nothing than an estimate 100x below the solved value."""
    assert "spring" in UNTRANSFERABLE_FAMILIES
    mcp = create_mcp(surrogate=_surrogate(tmp_path))

    with pytest.raises(Exception, match="duplicated end face"):
        _call(
            mcp,
            "surrogate_screen",
            {
                "path": str(_box(tmp_path)),
                "load_case_json": _case([1], [2]),
                "family": "spring",
            },
        )


def test_gravity_is_weighed_in_closed_form_and_recorded(tmp_path):
    mcp = create_mcp(surrogate=_surrogate(tmp_path))
    path = str(_box(tmp_path))
    faces = _call(mcp, "surrogate_inspect_cad", {"path": path})["faces"]
    held = [f["tag"] for f in faces if f["normal"][0] < -0.9]
    pushed = [f["tag"] for f in faces if f["normal"][0] > 0.9]

    out = _call(
        mcp,
        "surrogate_screen",
        {
            "path": path,
            "load_case_json": _case(
                held,
                pushed,
                material={"poisson_ratio": 0.3, "density_kg_m3": 7850.0},
                gravity={"enabled": True},
            ),
        },
    )

    assert any("self-weight" in a and "negligible" in a for a in out["assumptions"])


def test_an_unreadable_cad_format_is_refused_at_the_door(tmp_path):
    mcp = create_mcp(surrogate=_surrogate(tmp_path))

    with pytest.raises(Exception, match="Parasolid"):
        _call(mcp, "surrogate_inspect_cad", {"path": str(tmp_path / "part.x_t")})


def test_the_older_entry_point_still_works(tmp_path):
    """Anything already wired to surrogate_screen_step must keep working."""
    mcp = create_mcp(surrogate=_surrogate(tmp_path))

    out = _call(
        mcp,
        "surrogate_screen_step",
        {
            "step_path": str(_box(tmp_path)),
            "force_n": 5000.0,
            "fixed_json": json.dumps({"axis": "x", "at": "min"}),
            "loaded_json": json.dumps({"axis": "x", "at": "max"}),
        },
    )

    assert out["schema_version"] == "SurrogateScreening/v1"


def test_screening_works_with_no_retrieval_index_configured(tmp_path, monkeypatch):
    """Retrieval is optional context, never a dependency."""
    monkeypatch.delenv("FEA_SURROGATE_NEIGHBOURS", raising=False)
    mcp = create_mcp(surrogate=_surrogate(tmp_path))
    path = str(_box(tmp_path))
    faces = _call(mcp, "surrogate_inspect_cad", {"path": path})["faces"]

    out = _call(
        mcp,
        "surrogate_screen",
        {
            "path": path,
            "load_case_json": _case(
                [f["tag"] for f in faces if f["normal"][0] < -0.9],
                [f["tag"] for f in faces if f["normal"][0] > 0.9],
            ),
        },
    )

    assert out["predictions"]


def test_solving_returns_the_true_field_and_says_it_is_not_an_estimate(
    tmp_path, monkeypatch
) -> None:
    """The solve is a different kind of thing from the screen, and says so."""
    monkeypatch.setenv("FEA_SURROGATE_OUTPUT_DIR", str(tmp_path / "out"))
    mcp = create_mcp(surrogate=_surrogate(tmp_path))
    path = str(_box(tmp_path))
    faces = _call(mcp, "surrogate_inspect_cad", {"path": path})["faces"]
    held = [f["tag"] for f in faces if f["normal"][0] < -0.9]
    pushed = [f["tag"] for f in faces if f["normal"][0] > 0.9]

    out = _call(
        mcp,
        "surrogate_solve",
        {"path": path, "load_case_json": _case(held, pushed)},
    )

    assert out["schema_version"] == "FeaSolve/v1"
    assert out["attestation"] == "computed_analysis"
    assert out["critical_distance_mpa"] > 0.0
    # The peak is singular at a re-entrant corner; the summary must say which
    # number to use rather than leaving a reader to pick the bigger one.
    assert "critical_distance_mpa" in out["use"]
    assert out["critical_distance_mpa"] <= out["peak_von_mises_mpa"]
    assert (Path(out["output_dir"]) / out["image_relative_path"]).exists()


def test_a_solve_records_whether_the_screening_interval_held(
    tmp_path, monkeypatch
) -> None:
    """Every solve is a coverage datapoint on a part someone cared about."""
    out_dir = tmp_path / "out"
    monkeypatch.setenv("FEA_SURROGATE_OUTPUT_DIR", str(out_dir))
    mcp = create_mcp(surrogate=_surrogate(tmp_path))
    path = str(_box(tmp_path))
    faces = _call(mcp, "surrogate_inspect_cad", {"path": path})["faces"]

    out = _call(
        mcp,
        "surrogate_solve",
        {
            "path": path,
            "load_case_json": _case(
                [f["tag"] for f in faces if f["normal"][0] < -0.9],
                [f["tag"] for f in faces if f["normal"][0] > 0.9],
            ),
            "family": "bracket",
        },
    )

    crosscheck = out["crosscheck"]
    assert crosscheck["available"]
    assert crosscheck["solved"] == pytest.approx(out["critical_distance_mpa"])
    assert isinstance(crosscheck["inside"], bool)
    assert (out_dir / "crosschecks.jsonl").exists()


def test_inspecting_a_part_also_draws_its_faces(tmp_path, monkeypatch) -> None:
    """Asking which face is held is answerable by pointing, or it is guesswork."""
    monkeypatch.setenv("FEA_SURROGATE_OUTPUT_DIR", str(tmp_path / "out"))
    mcp = create_mcp(surrogate=_surrogate(tmp_path))

    out = _call(mcp, "surrogate_inspect_cad", {"path": str(_box(tmp_path))})

    assert "faces_image_error" not in out
    assert (Path(out["output_dir"]) / out["faces_image_relative_path"]).exists()
