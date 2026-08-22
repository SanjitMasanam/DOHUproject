#!/usr/bin/env python3
"""Checks for figure_paths.RunLayout and the migrate_figures path mapping.

Plain asserts, no pytest dependency:  python3 test_figure_paths.py

Two layers:

  1. A golden table pinning one path per figure class per variant, so a change
     to the layout has to be made deliberately rather than by accident.
  2. A round-trip over the real legacy file list, asserting that every path
     migrate_figures produces is collision-free and is also reachable from
     RunLayout. That is the check that keeps the migration and the runtime path
     construction from drifting apart -- they are two independent
     implementations of the same layout, and this pins them together.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import figure_paths as fp
import migrate_figures as mig
from figure_paths import (
   BUDGET,
   CORNER,
   FIGURE_ROOT,
   RUN_TYPE_DIRS,
   STEP1,
   STEP2,
   VALIDATION,
   RunLayout,
   figname,
)

RUN_TYPE_BY_DIR = {v: k for k, v in RUN_TYPE_DIRS.items()}

FAILURES: list[str] = []


def check(label: str, got, want) -> None:
   if got != want:
      FAILURES.append(f"{label}\n     got:  {got}\n     want: {want}")


def rel(p: Path) -> str:
   return str(p.relative_to(FIGURE_ROOT.parent))


# --------------------------------------------------------------- figname()

def test_figname() -> None:
   check("figname plain",
         figname("4xCO2", "all_models", "T2M_vs_NETTOA"),
         "4xCO2_all_models_T2M_vs_NETTOA")
   check("figname with scale",
         figname("4xCO2", "all_models", "T2m_vs_t", "linear"),
         "4xCO2_all_models_T2m_vs_t_linear")
   check("figname drops empty fields",
         figname("4xCO2", "", "all_models", None, "ohc_ts"),
         "4xCO2_all_models_ohc_ts")
   check("figname iteration is last and zero-padded",
         figname("4xCO2", "all_models", "T2M_vs_NETTOA", iteration=7),
         "4xCO2_all_models_T2M_vs_NETTOA_iter07")
   check("figname iteration reaches two digits",
         figname("x", iteration=10), "x_iter10")

   try:
      figname()
   except ValueError:
      pass
   else:
      FAILURES.append("figname() with no fields should raise ValueError")


# ------------------------------------------------------------ golden table

def test_golden_paths() -> None:
   """One row per figure class. These strings are the contract."""
   b1 = RunLayout("2013b", 1)
   b3v = RunLayout("2013b", 3, VALIDATION)
   fx2 = RunLayout("diffusive-eps-joint_fixedRadiativeParams", 2)
   cor1 = RunLayout("diffusive-eps-corner", 1)

   cases = [
      (b1.path(STEP1, figname("4xCO2", "all_models", "T2M_vs_NETTOA", iteration=3)),
       "figures/2013b/01_replicate/step1/4xCO2_all_models_T2M_vs_NETTOA_iter03_2013b.png"),

      (b3v.path(STEP2, figname("4xCO2", "all_models", "log_Teq_minus_T_vs_t", iteration=7)),
       "figures/2013b/03_lrfit-50yravg/step2/"
       "4xCO2_all_models_log_Teq_minus_T_vs_t_iter07_2013b.png"),

      (b3v.results_path(figname("4xCO2", "all_models", "T2m_vs_t", "log")),
       "figures/2013b/03_lrfit-50yravg/validation/4xCO2_all_models_T2m_vs_t_log_2013b.png"),

      (b1.path(BUDGET, figname("4xCO2", "CCSM3", "surface_budget_bars")),
       "figures/2013b/01_replicate/budget/4xCO2_CCSM3_surface_budget_bars_2013b.png"),

      (b1.csv_path("params_final", mkdir=False),
       "figures/2013b/01_replicate/tables/params_final_2013b.csv"),

      (fx2.path(STEP1, figname("4xCO2", "all_models", "N_T_H_regression3d")),
       "figures/diffusive-eps-joint_fixedRadiativeParams/02_forcing-50yravg/step1/"
       "4xCO2_all_models_N_T_H_regression3d_diffusive-eps-joint_fixedRadiativeParams.png"),

      (fx2.results_path(figname("4xCO2", "all_models", "sensitivity_spaghetti_kappa", "log")),
       "figures/diffusive-eps-joint_fixedRadiativeParams/02_forcing-50yravg/unblinded/"
       "4xCO2_all_models_sensitivity_spaghetti_kappa_log"
       "_diffusive-eps-joint_fixedRadiativeParams.png"),

      (fx2.results_path(figname("4xCO2", "all_models", "sensitivity_heatmap_rmse")),
       "figures/diffusive-eps-joint_fixedRadiativeParams/02_forcing-50yravg/unblinded/"
       "4xCO2_all_models_sensitivity_heatmap_rmse"
       "_diffusive-eps-joint_fixedRadiativeParams.png"),

      (fx2.csv_path("fitted_model_params", mkdir=False),
       "figures/diffusive-eps-joint_fixedRadiativeParams/02_forcing-50yravg/tables/"
       "fitted_model_params_diffusive-eps-joint_fixedRadiativeParams.csv"),

      (cor1.path(CORNER, figname("4xCO2", "CESM1", "posterior_corner")),
       "figures/diffusive-eps-corner/01_replicate/corner/"
       "4xCO2_CESM1_posterior_corner_diffusive-eps-corner.png"),
   ]
   for got, want in cases:
      check(f"golden {want}", rel(got), want)

   check("run-type dirs sort in run order",
         [RunLayout("v", rt).dir.name for rt in (1, 2, 3)],
         sorted(RunLayout("v", rt).dir.name for rt in (1, 2, 3)))



def test_validation() -> None:
   for bad in (0, 4, "1"):
      try:
         RunLayout("v", bad)
      except ValueError:
         pass
      else:
         FAILURES.append(f"RunLayout should reject run_type={bad!r}")

   try:
      RunLayout("v", 1).path("stpe1", "x")
   except ValueError:
      pass
   else:
      FAILURES.append("path() should reject an unknown section (typo guard)")

   try:
      RunLayout("v", 1, "unbinded")
   except ValueError:
      pass
   else:
      FAILURES.append("RunLayout should reject an unknown results kind")


# ------------------------------------------------------- migration round-trip

def test_migration_round_trip() -> None:
   """Every legacy path maps somewhere unique that RunLayout can also produce."""
   paths = mig.tracked_legacy_files()
   if not paths:
      print("  (migration round-trip skipped: legacy trees already migrated)")
      return

   moves, deletions, problems = mig.build_plan(paths)
   check("migration reports no problems", problems, [])
   check("migration accounts for every legacy file",
         len(moves) + len(deletions), len(paths))

   by_dest = defaultdict(list)
   for src, dest in moves.items():
      by_dest[dest].append(src)
   collisions = {d: s for d, s in by_dest.items() if len(s) > 1}
   check("no two files map to one destination", collisions, {})

   # Every destination must live under figures/ and be re-derivable from
   # RunLayout -- i.e. migrate_figures cannot invent a shape the scripts can't.
   for dest in moves.values():
      parts = dest.parts
      if parts[0] != "figures":
         FAILURES.append(f"destination escapes figures/: {dest}")
         continue
      if parts[1] == "pde-checks":
         continue
      if len(parts) >= 3 and parts[2] == "comparison":
         check(f"comparison shape {dest}", len(parts), 4)
         continue
      variant, run_dir, section, name = parts[1], parts[2], parts[3], parts[4]
      run_type = RUN_TYPE_BY_DIR.get(run_dir)
      if run_type is None:
         FAILURES.append(f"destination has an unknown run-type dir: {dest}")
         continue
      stem, ext = Path(name).stem, Path(name).suffix.lstrip(".")
      rebuilt = RunLayout(variant, run_type).path(section, stem, ext=ext)
      check(f"round-trip {dest}", rel(rebuilt), str(dest))

   # The png/ layer must be gone everywhere.
   leftover = [d for d in moves.values() if "png" in d.parts[:-1]]
   check("no destination retains a png/ directory", leftover, [])

   # And no destination may keep a redundant run-type stamp.
   stamped = [d for d in moves.values()
              if any(d.stem.endswith(s) for s in mig.STAMP_SUFFIXES)]
   check("no destination retains a run-type stamp", stamped, [])


# ------------------------------------------------------------ variant stamp

def test_variant_stamp() -> None:
   """Every filename ends with the variant that produced it."""
   mt2 = RunLayout("MT2", 1, VALIDATION)
   name = figname("4xCO2", "all_models", "T2M_vs_NETTOA")

   check("stamp is the last field of a figure name",
         mt2.results_path(name).stem, "4xCO2_all_models_T2M_vs_NETTOA_MT2")
   check("stamp follows the iteration field",
         mt2.path(STEP1, figname("4xCO2", "all_models", "T2M_vs_NETTOA", iteration=3)).stem,
         "4xCO2_all_models_T2M_vs_NETTOA_iter03_MT2")
   check("stamp lands before the extension on tables",
         mt2.csv_path("fitted_model_params", mkdir=False).name,
         "fitted_model_params_MT2.csv")
   check("variant_stamp property", mt2.variant_stamp, "_MT2")

   # The same plot from two variants must not collide once the files are moved
   # out of their directories -- that is the whole point of the stamp.
   stems = {RunLayout(v, 1, VALIDATION).results_path(name).name
            for v in ("MT2", "2013a", "2013b", "diffusive-eps")}
   check("four variants give four distinct filenames", len(stems), 4)

   # And the switch has to actually switch it off.
   fp.INCLUDE_VARIANT_STAMP = False
   try:
      check("stamp off restores the bare name",
            mt2.results_path(name).stem, "4xCO2_all_models_T2M_vs_NETTOA")
   finally:
      fp.INCLUDE_VARIANT_STAMP = True


def main() -> int:
   for fn in (test_figname, test_golden_paths, test_variant_stamp, test_validation,
             test_migration_round_trip):
      print(f"- {fn.__name__}")
      fn()

   if FAILURES:
      print(f"\n{len(FAILURES)} FAILURE(S):\n")
      for f in FAILURES:
         print(f"  {f}\n")
      return 1
   print("\nAll figure-path checks passed.")
   return 0


if __name__ == "__main__":
   sys.exit(main())
