"""The contract has to be checkable without the model stack.

`screening.py` is the boundary with CAD Agent, and its docstring claims nothing
in it imports torch or gmsh. That claim is what lets a consumer inspect a stored
document, or a CI job check the contract, without installing a CUDA-capable
wheel and a mesher. Nothing enforced it until this file, and an import added
anywhere in the chain would have broken it silently -- the modules would still
work perfectly on a developer machine that has everything installed.

Checked in a subprocess because an in-process check is worthless: by the time
the suite has imported `test_render`, torch is already in `sys.modules` and the
assertion passes no matter what the module does.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


SRC = str(Path(__file__).resolve().parents[1] / "src")

#: Everything under `inference` and `render`. A contract module that reaches for
#: any of these has moved the boundary.
HEAVY = ("torch", "torch_geometric", "gmsh", "scipy", "matplotlib")

#: The modules a consumer may import to read or check a document.
CONTRACT = ("fea_surrogate", "fea_surrogate.screening", "fea_surrogate.versions",
            "fea_surrogate.selectors")

#: The terminal analysis's front half: the command, the case model, the
#: screening gate and the core that orchestrates them. `fea-structural --help`,
#: the interview and the gate all run before anything is meshed, so none of
#: them may pull in the mesher, the solver, the renderer or the model stack at
#: import time -- that is what lets the CI contract job exercise them.
TERMINAL = ("fea_surrogate.cli", "fea_surrogate.structural_case",
            "fea_surrogate.screenability", "fea_surrogate.structural_run")


def _heavy_after_importing(modules: tuple[str, ...]) -> list[str]:
    script = (
        "import sys\n"
        f"for name in {modules!r}:\n"
        "    __import__(name)\n"
        f"print(' '.join(m for m in {HEAVY!r} if m in sys.modules))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, check=True, cwd=SRC,
    )
    return result.stdout.split()


def test_the_contract_modules_import_without_the_model_stack() -> None:
    assert _heavy_after_importing(CONTRACT) == []


def test_the_check_would_notice_a_heavy_import() -> None:
    """Guard the guard: a test that cannot fail is not protecting anything.

    Needs gmsh to exist, so it runs in the full job and skips in the contract
    job, whose base install deliberately has none of the heavy modules.
    """
    pytest.importorskip("gmsh")
    found = _heavy_after_importing(("fea_surrogate.screening", "numpy", "gmsh"))

    assert "gmsh" in found


def test_the_renderer_does_not_drag_in_the_model_stack() -> None:
    """Drawing a stored field needs no model.

    `render` is a separate extra from `inference` for this reason, so the module
    has to keep its matplotlib and torch imports inside the functions that use
    them.
    """
    assert _heavy_after_importing(("fea_surrogate.render",)) == []


@pytest.mark.parametrize("module", TERMINAL)
def test_the_terminal_front_end_imports_without_the_heavy_stack(module: str) -> None:
    """Prompts and the screening gate must not wait on gmsh, scipy or torch.

    Each module is checked on its own, so a heavy import in one of them is
    named rather than hidden behind another that happens to import it first.
    """
    assert _heavy_after_importing((module,)) == []
