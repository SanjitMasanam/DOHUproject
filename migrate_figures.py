#!/usr/bin/env python3
"""One-time migration of the legacy figures_*/ trees into the figures/ layout.

    figures_2013b/geoffroy_replicate_results/step1/png/x_50yrAvg.png
      ->  figures/2013b/01_replicate/step1/x.png

What changes, and why:

  * `figures_<variant>/`         -> `figures/<variant>/`      (one root, not seven)
  * the run-type directory       -> `01_replicate` / `02_forcing-50yravg` /
                                    `03_lrfit-50yravg`, so it sorts in run order
                                    and is named identically in both families
  * the trailing `png/` segment  -> dropped (nothing was ever written beside it)
  * run-type-root `*.csv`        -> `tables/`, matching 2013b
  * the `_50yrAvg` / `_50yrAvg_LRparamFit` stamp -> dropped (the run-type
    directory already says this; re-enable via figure_paths.INCLUDE_RUNTYPE_STAMP)
  * the `_unblinded` / `_validation` token -> dropped (ditto for the section dir)
  * budget bars missing the `4xCO2_` prefix -> given it, matching the other trees

`--dry-run` is the default: a bare `python3 migrate_figures.py` cannot mutate
anything. Idempotent -- anything already under figures/ is ignored, so a second
`--apply` is a no-op.

Stdlib-only (plus figure_paths, which is itself stdlib-only), so this runs
without matplotlib / rpy2 installed.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from figure_paths import (
   CHECKS_DIR,
   COMPARISON,
   FIGURE_ROOT,
   INCLUDE_RUNTYPE_STAMP,
   RUN_TYPE_STAMPS,
   TABLES,
   RunLayout,
)

REPO = Path(__file__).resolve().parent

# Legacy top-level directory -> variant slug. The slug is the variant token
# verbatim, so it stays trivially checkable against each script's filename.
LEGACY_ROOTS = {
   "figures_2013a": "2013a",
   "figures_2013b": "2013b",
   "figures_diffusive": "diffusive",
   "figures_diffusive-eps": "diffusive-eps",
   "figures_diffusive-eps-corner": "diffusive-eps-corner",
   "figures_diffusive-eps-joint": "diffusive-eps-joint",
   "figures_diffusive-eps-joint_fixedRadiativeParams": "diffusive-eps-joint_fixedRadiativeParams",
}

# Legacy run-type directory -> run_type. Both families' names map onto the same
# three numbers; that collapse is the point of the migration.
LEGACY_RUN_DIRS = {
   "replicate_results": 1,
   "geoffroy_replicate_results": 1,
   "50-yr_avg_forcing_results": 2,
   "50-yr_avg_LR_fit_results": 3,
   "50-yr_avg_tau_s_LR_fit_results": 3,
}

# The flat PDE-check dump and the stale copy of it.
LEGACY_CHECKS = "figures_diffusiveChecks"
STALE_CHECKS_COPY = "1lyEBM_diffusive"
LEGACY_COMPARISON = "comparison_plots"

# Longest-first so "_50yrAvg_LRparamFit" is matched before "_50yrAvg".
STAMP_SUFFIXES = tuple(
   sorted((f"_{s}" for s in RUN_TYPE_STAMPS.values() if s), key=len, reverse=True)
)

RESULT_SECTIONS = ("unblinded", "validation")
SCALE_TOKENS = ("linear", "log")
EXPTS = ("4xCO2",)

# Files that carry no information and should not survive the move.
JUNK_NAMES = {".DS_Store"}

#: Names whose legacy form does not fit the general grammar and are rewritten
#: by hand. Keyed by (section, stem).
EXPLICIT_RENAMES = {
   # `all_` was the scope field; spell it the way every other figure does.
   ("validation", "all_validation_params_GF_vs_SN"): "all_models_validation_params_GF_vs_SN",
}


class UnmappedPath(Exception):
   """A legacy path this script does not know how to classify."""


def _strip_stamp(stem: str, run_type: int) -> str:
   """Remove the trailing run-type stamp, verifying it matches the directory.

   A mismatch means a file is sitting in the wrong run-type directory, which is
   a real problem worth stopping for rather than silently normalising away.
   """
   if INCLUDE_RUNTYPE_STAMP:
      return stem
   for suffix in STAMP_SUFFIXES:
      if stem.endswith(suffix):
         expected = RUN_TYPE_STAMPS[run_type]
         if suffix != f"_{expected}":
            raise UnmappedPath(
               f"stem {stem!r} carries stamp {suffix!r} but sits in a run_type={run_type} "
               f"directory (expected {'_' + expected if expected else 'no stamp'})"
            )
         return stem[: -len(suffix)]
   return stem


def _strip_results_token(stem: str, section: str) -> str:
   """Drop a redundant `_unblinded` / `_validation` field from a filename.

   Only removed where it is genuinely the results marker: either the final
   field, or the field immediately before a trailing `linear`/`log`. That
   precision matters -- `all_validation_params_GF_vs_SN` contains "validation"
   as a field but it is part of the quantity name, not a section marker.
   """
   if section not in RESULT_SECTIONS:
      return stem
   fields = stem.split("_")
   if fields[-1] == section:
      return "_".join(fields[:-1])
   if len(fields) >= 2 and fields[-1] in SCALE_TOKENS and fields[-2] == section:
      return "_".join(fields[:-2] + fields[-1:])
   return stem


def _add_expt_prefix(stem: str) -> str:
   """Give per-model figures the `4xCO2_` prefix the other trees already carry."""
   return stem if stem.startswith(tuple(f"{e}_" for e in EXPTS)) else f"{EXPTS[0]}_{stem}"


def new_path(old: Path) -> Path | None:
   """Map one legacy path to its new home. None means "delete, don't move".

   Pure: touches no filesystem state, so it is unit-testable and safe to run
   over a frozen `git ls-files` snapshot.
   """
   parts = old.parts
   if not parts:
      raise UnmappedPath(str(old))

   # Already migrated -- this is what makes a second --apply a no-op.
   if parts[0] == FIGURE_ROOT.name:
      return None

   if old.name in JUNK_NAMES:
      return None

   # The stale byte-for-byte copy of the PDE checks.
   if parts[0] == STALE_CHECKS_COPY:
      return None

   if parts[0] == LEGACY_CHECKS:
      return CHECKS_DIR.joinpath(*parts[1:])

   if parts[0] not in LEGACY_ROOTS:
      raise UnmappedPath(f"unknown top-level directory: {old}")
   variant = LEGACY_ROOTS[parts[0]]

   if len(parts) < 3:
      raise UnmappedPath(f"too shallow to classify: {old}")

   # Cross-run-type comparison output.
   if parts[1] == LEGACY_COMPARISON:
      return FIGURE_ROOT.joinpath(variant, COMPARISON, *parts[2:])

   if parts[1] not in LEGACY_RUN_DIRS:
      raise UnmappedPath(f"unknown run-type directory: {old}")
   run_type = LEGACY_RUN_DIRS[parts[1]]
   layout = RunLayout(variant, run_type)

   rest = parts[2:]
   stem, ext = old.stem, old.suffix.lstrip(".")

   # A CSV sitting directly in the run-type root (the diffusive family and
   # 2013b's fitted_model_params) belongs in tables/ like everything else.
   if len(rest) == 1:
      if ext != "csv":
         raise UnmappedPath(f"unexpected non-CSV file at run-type root: {old}")
      return layout.path(TABLES, _strip_stamp(stem, run_type), ext="csv")

   section = rest[0]
   inner = rest[1:]

   if section == TABLES:
      if len(inner) != 1:
         raise UnmappedPath(f"unexpected nesting under tables/: {old}")
      return layout.path(TABLES, _strip_stamp(stem, run_type), ext=ext)

   # Collapse the vestigial png/ level. Anything else nested here is unknown.
   if len(inner) == 2:
      if inner[0] != "png":
         raise UnmappedPath(f"unexpected directory {inner[0]!r} under {section}/: {old}")
   elif len(inner) != 1:
      raise UnmappedPath(f"unexpected nesting under {section}/: {old}")

   stem = _strip_stamp(stem, run_type)
   stem = EXPLICIT_RENAMES.get((section, stem), _strip_results_token(stem, section))
   if section in ("budget", "corner"):
      stem = _add_expt_prefix(stem)

   return layout.path(section, stem, ext=ext)


# --------------------------------------------------------------------- scanning


def tracked_legacy_files() -> list[Path]:
   """Every git-tracked file under a legacy tree, repo-relative."""
   out = subprocess.run(
      ["git", "-C", str(REPO), "ls-files", "-z"],
      check=True, capture_output=True, text=True,
   ).stdout
   roots = set(LEGACY_ROOTS) | {LEGACY_CHECKS, STALE_CHECKS_COPY}
   return [
      Path(line) for line in out.split("\0")
      if line and Path(line).parts[0] in roots
   ]


def untracked_legacy_files() -> list[Path]:
   """Legacy files on disk that git does not track (e.g. a fresh .DS_Store)."""
   roots = set(LEGACY_ROOTS) | {LEGACY_CHECKS, STALE_CHECKS_COPY}
   found = []
   for root in sorted(roots):
      base = REPO / root
      if not base.is_dir():
         continue
      for p in base.rglob("*"):
         if p.is_file():
            found.append(p.relative_to(REPO))
   tracked = set(tracked_legacy_files())
   return [p for p in found if p not in tracked]


def build_plan(paths: list[Path]) -> tuple[dict[Path, Path], list[Path], list[str]]:
   """Return (moves, deletions, problems)."""
   moves: dict[Path, Path] = {}
   deletions: list[Path] = []
   problems: list[str] = []

   for old in sorted(paths):
      try:
         dest = new_path(old)
      except UnmappedPath as exc:
         problems.append(f"UNMAPPED  {old}  ({exc})")
         continue
      if dest is None:
         deletions.append(old)
      else:
         moves[old] = dest.relative_to(REPO)

   # Collision check: two sources landing on one destination would silently
   # destroy a file. This is the gate that catches an over-eager token strip.
   by_dest: dict[Path, list[Path]] = defaultdict(list)
   for src, dest in moves.items():
      by_dest[dest].append(src)
   for dest, srcs in sorted(by_dest.items()):
      if len(srcs) > 1:
         problems.append(
            "COLLISION " + str(dest) + "\n" + "".join(f"             <- {s}\n" for s in srcs)
         )

   return moves, deletions, problems


# -------------------------------------------------------------------- execution


def _run(cmd: list[str]) -> None:
   subprocess.run(cmd, check=True, cwd=REPO, capture_output=True, text=True)


def apply_plan(moves: dict[Path, Path], deletions: list[Path], use_git: bool) -> None:
   for src, dest in sorted(moves.items()):
      (REPO / dest).parent.mkdir(parents=True, exist_ok=True)
      if use_git:
         _run(["git", "mv", str(src), str(dest)])
      else:
         os.replace(REPO / src, REPO / dest)

   for path in deletions:
      if use_git:
         # --ignore-unmatch so untracked junk (a fresh .DS_Store) doesn't abort.
         _run(["git", "rm", "-q", "--ignore-unmatch", str(path)])
      (REPO / path).unlink(missing_ok=True)


def prune_empty_dirs() -> list[Path]:
   """Remove drained legacy directories, deepest first.

   This is what clears the vestigial png/ dirs, the six permanently-empty
   validation/ dirs under figures_diffusive-eps-joint_fixedRadiativeParams, and
   the legacy roots themselves. Disk-only: git does not track empty directories.
   """
   removed = []
   roots = set(LEGACY_ROOTS) | {LEGACY_CHECKS, STALE_CHECKS_COPY}
   for root in sorted(roots):
      base = REPO / root
      if not base.is_dir():
         continue
      for d in sorted(base.rglob("*"), key=lambda p: len(p.parts), reverse=True):
         if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
            removed.append(d.relative_to(REPO))
      if not any(base.iterdir()):
         base.rmdir()
         removed.append(base.relative_to(REPO))
   return removed


def main(argv=None) -> int:
   ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
   ap.add_argument("--apply", action="store_true",
                   help="actually move files (default is a dry run)")
   ap.add_argument("--no-git", action="store_true",
                   help="use os.replace instead of `git mv` (loses rename staging)")
   args = ap.parse_args(argv)

   paths = tracked_legacy_files() + untracked_legacy_files()
   if not paths:
      print("Nothing to migrate: no files under any legacy figures_* tree.")
      return 0

   moves, deletions, problems = build_plan(paths)

   for src, dest in sorted(moves.items()):
      print(f"  {src}\n    -> {dest}")
   for path in deletions:
      print(f"  DELETE {path}")
   print(f"\n{len(moves)} to move, {len(deletions)} to delete, {len(problems)} problem(s).")

   if problems:
      print("\nAborting -- resolve these first:\n", file=sys.stderr)
      for p in problems:
         print(f"  {p}", file=sys.stderr)
      return 1

   if not args.apply:
      print("\nDry run. Re-run with --apply to perform the migration.")
      return 0

   use_git = not args.no_git
   if use_git:
      dirty = subprocess.run(
         ["git", "-C", str(REPO), "status", "--porcelain"],
         check=True, capture_output=True, text=True,
      ).stdout.strip()
      if dirty:
         print("Aborting: working tree is not clean. Commit or stash first.", file=sys.stderr)
         return 1

   apply_plan(moves, deletions, use_git)
   removed = prune_empty_dirs()
   print(f"\nMigrated. Removed {len(removed)} drained director{'y' if len(removed) == 1 else 'ies'}.")
   return 0


if __name__ == "__main__":
   raise SystemExit(main())
