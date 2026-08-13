"""Create a clean source-code submission archive."""

import os
import zipfile


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(ROOT, "runs", "submission", "cityvimd_source.zip")
INCLUDE = [
    "README.md",
    "SUBMISSION.md",
    "BENCHMARKS.md",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "CITATION.cff",
    "LICENSE",
    "requirements.txt",
    "requirements-dev.txt",
    "pytest.ini",
    "train.py",
    "test.py",
    "configs",
    "datasets",
    "docs",
    "models",
    "tests",
    "utils",
    "tools",
]
EXCLUDED_DIRS = {"__pycache__"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def iter_files(path):
    if os.path.isfile(path):
        yield path
        return
    for directory, dirnames, filenames in os.walk(path):
        dirnames[:] = [name for name in dirnames if name not in EXCLUDED_DIRS]
        for filename in filenames:
            if os.path.splitext(filename)[1].lower() not in EXCLUDED_SUFFIXES:
                yield os.path.join(directory, filename)


def main():
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in INCLUDE:
            path = os.path.join(ROOT, item)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required submission item is missing: {item}")
            for file_path in iter_files(path):
                archive.write(file_path, os.path.relpath(file_path, ROOT))
    print(f"SUBMISSION_ARCHIVE_OK {OUTPUT}")


if __name__ == "__main__":
    main()
