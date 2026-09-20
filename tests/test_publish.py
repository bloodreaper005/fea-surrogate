from __future__ import annotations

import json

import pytest

from fea_surrogate.generate.publish import build_card, publish


MANIFEST = {
    "samples": 3000,
    "by_family": {"bracket": 500, "spring": 500},
    "set_sha256": "b" * 64,
    "seed": 0,
    "rejected": 7,
    "generated_at": "2026-09-20T08:00:00Z",
}


def test_the_card_reports_what_the_manifest_says() -> None:
    card = build_card(MANIFEST, "someone/sweep")

    assert "3000 samples" in card
    assert "b" * 64 in card
    assert "| `bracket` | 500 |" in card
    assert "7 draws rejected" in card


def test_the_card_states_the_conversion_back_to_stress() -> None:
    """A dimensionless target is useless without the way back."""
    card = build_card(MANIFEST, "someone/sweep")

    assert "sigma[MPa] = tau * F[N] / L[mm]^2" in card
    assert "independent of" in card and "Young" in card


def test_the_card_declares_its_limitations() -> None:
    card = build_card(MANIFEST, "someone/sweep")

    for fragment in ("Constant-strain", "Linear elastic only", "not a certified value"):
        assert fragment in card


def test_publishing_without_a_token_refuses(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    (tmp_path / "manifest.json").write_text(json.dumps(MANIFEST), encoding="utf-8")

    with pytest.raises(SystemExit, match="HF_TOKEN"):
        publish(tmp_path, "someone/sweep")


def test_publishing_without_a_sweep_refuses(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_fake")

    with pytest.raises(SystemExit, match="no manifest"):
        publish(tmp_path, "someone/sweep")
