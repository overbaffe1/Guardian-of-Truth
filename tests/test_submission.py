"""Submission-format guarantees.

Both behaviours below were learned from participants losing runs to opaque
platform errors, and neither is visible in a local smoke test:

* the scorer reads Parquet even when ``--output`` ends in ``.csv``;
* the archive must be a ZIP with no wrapping directory, no ``metrics/`` and
  readable permissions, or stage 08 fails.
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_csv_output_path_still_writes_parquet(tmp_path):
    """Stage 22 fails when the predictions are CSV, whatever --output says."""
    sys.path.insert(0, str(REPO / "src"))
    from guardian.io_utils import write_predictions
    from guardian.types import Prediction

    target = tmp_path / "predictions.csv"
    write_predictions(str(target), [Prediction(id="a", label=1, score=1.0)])
    assert target.read_bytes()[:4] == b"PAR1", "output must be Parquet, not CSV"

    import pandas as pd

    frame = pd.read_parquet(target)
    assert list(frame.columns) == ["id", "label"]
    assert frame["label"].tolist() == [1]


def test_output_format_can_be_overridden(tmp_path):
    sys.path.insert(0, str(REPO / "src"))
    from guardian.io_utils import write_predictions
    from guardian.types import Prediction

    target = tmp_path / "out.csv"
    write_predictions(str(target), [Prediction(id="a", label=0, score=0.0)], fmt="csv")
    assert target.read_text().splitlines()[0] == "id,label"


def _build(tmp_path):
    out = tmp_path / "submission.zip"
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "make_submission.py"), "--out", str(out)],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert result.returncode == 0, result.stderr
    return out


def test_archive_is_zip_with_required_entries(tmp_path):
    out = _build(tmp_path)
    assert zipfile.is_zipfile(out)
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
    # No wrapping directory: these must live at the archive root.
    assert "scripts/predict.py" in names
    assert "pyproject.toml" in names
    assert "submission.zip/scripts/predict.py" not in names


def test_archive_permissions_are_traversable(tmp_path):
    """A zip built on Windows can store dirs as 0600 and stage 08 then fails."""
    out = _build(tmp_path)
    with zipfile.ZipFile(out) as zf:
        for info in zf.infolist():
            if info.is_dir():
                mode = (info.external_attr >> 16) & 0o777
                assert mode == 0o755, f"{info.filename} has mode {mode:o}"
            elif info.filename.endswith(".py"):
                mode = (info.external_attr >> 16) & 0o777
                assert mode == 0o644, f"{info.filename} has mode {mode:o}"


def test_archive_excludes_forbidden_and_junk_paths(tmp_path):
    out = _build(tmp_path)
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
    assert not any(n.startswith("metrics/") for n in names)
    assert not any("__pycache__" in n for n in names)
    assert not any(n.startswith(".git/") for n in names)
    assert not any(n.endswith((".pyc", ".parquet")) for n in names)


def test_check_reports_missing_weights(tmp_path):
    """The scoring environment ships no weights, so they must be in the archive."""
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "make_submission.py"), "--check"],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert "weights" in result.stderr

def test_entries_are_written_with_zip64(tmp_path, monkeypatch):
    """The judge weights push the archive past 4 GB.

    zipfile cannot know an entry's size while it is streamed, so without
    ``force_zip64`` the build dies with "File size unexpectedly exceeded ZIP64
    limit" -- which is exactly how the first runner build failed.
    """
    sys.path.insert(0, str(REPO / "scripts"))
    import make_submission

    seen: list[bool] = []
    real_open = zipfile.ZipFile.open

    def spy(self, name, mode="r", *args, **kwargs):
        if mode == "w":
            # Directory stubs go through writestr and are tiny; only real files
            # can cross the 4 GB boundary.
            filename = getattr(name, "filename", name)
            if not str(filename).endswith("/"):
                seen.append(bool(kwargs.get("force_zip64")))
        return real_open(self, name, mode, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "open", spy)
    make_submission.build(REPO, tmp_path / "s.zip", None)
    assert seen, "nothing was written"
    assert all(seen), "every entry must be written with force_zip64"


def test_weights_from_an_external_directory_are_included(tmp_path):
    """``--model /path/to/weights`` must actually put those files in model/."""
    sys.path.insert(0, str(REPO / "scripts"))
    import make_submission

    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "w.safetensors").write_bytes(b"x" * 1024)

    out = tmp_path / "s.zip"
    make_submission.build(REPO, out, weights)
    with zipfile.ZipFile(out) as zf:
        listed = zf.namelist()
    assert len(listed) == len(set(listed)), "duplicate entries in the archive"
    assert "model/w.safetensors" in listed
    with zipfile.ZipFile(out) as zf:
        assert zf.read("model/w.safetensors") == b"x" * 1024
