"""Writes the package versions installed on this computer into requirements.txt.

Run it once after training, before you deploy the app. The saved models
open reliably only when the hosted app uses the same versions.

Run:  python pin_versions.py
"""
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

APP_PACKAGES = ["pandas", "numpy", "pyarrow", "scikit-learn", "optbinning", "lightgbm", "joblib", "streamlit"]

lines = []
for name in APP_PACKAGES:
    try:
        lines.append(f"{name}=={version(name)}")
    except PackageNotFoundError:
        lines.append(name)
lines.append("snowflake-connector-python[pandas]")
Path(__file__).with_name("requirements.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
