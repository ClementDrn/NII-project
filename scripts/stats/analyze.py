"""Run DIGIT's quantitative workflows from an uninstalled source checkout."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

try:
    from stats.cli import main
except ModuleNotFoundError as exc:
    if exc.name == "stats" or exc.name.startswith("stats."):
        raise
    raise SystemExit(
        f"Missing analysis dependency {exc.name!r}. Install scripts/stats/requirements.txt; "
        "see doc/analysis.md."
    ) from exc

if __name__ == "__main__":
    raise SystemExit(main())
