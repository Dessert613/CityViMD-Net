import zipfile

from tools import package_submission


def test_source_archive_contains_public_reproducibility_files(tmp_path, monkeypatch):
    output = tmp_path / "cityvimd_source.zip"
    monkeypatch.setattr(package_submission, "OUTPUT", str(output))

    package_submission.main()

    assert output.exists()
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())

    required = {
        "README.md",
        "BENCHMARKS.md",
        "CONTRIBUTING.md",
        "LICENSE",
        "CITATION.cff",
        "docs/architecture.md",
        "docs/reproducibility.md",
        "tests/test_model.py",
    }
    assert required <= names
    assert "SUBMISSION_LOG.md" not in names
    assert not any("__pycache__" in name for name in names)
    assert not any(name.endswith((".pyc", ".pyo")) for name in names)
