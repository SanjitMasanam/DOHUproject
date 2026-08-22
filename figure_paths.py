#!/usr/bin/env python3
"""Canonical figure/table output layout for the longrun parameter-fit scripts.

Every output in this project lands at one of four shapes:

    figures/<variant>/<NN_runtype>/<section>/<name>.png
    figures/<variant>/<NN_runtype>/tables/<name>.csv
    figures/<variant>/comparison/<name>.{png,pdf}
    figures/pde-checks/<name>.png

`variant` is the token that used to name the top-level `figures_<variant>/`
directory (e.g. "2013b", "diffusive-eps-joint"); `NN_runtype` is the numbered
run-type directory, numbered so `ls` sorts in run order rather than
alphabetically.

Every filename ends with its variant (`..._T2M_vs_NETTOA_MT2.png`), so the same
plot fit by two different models stays distinguishable once the files are out of
their directories -- side by side in a viewer, dropped into a slide, or attached
to an email. See INCLUDE_VARIANT_STAMP.

Directories are created by writes, never declared up front -- `save()`,
`reserve()` and `csv_path()` mkdir on demand, so a directory exists if and only
if something was actually written into it. (The old `ensure_dirs()` helper
created a `validation/` directory in every diffusive script even though all of
them set RESULT_KINDS = ["unblinded"], leaving permanently-empty dirs behind.)

Stdlib-only on purpose: migrate_figures.py and test_figure_paths.py import this
without needing matplotlib / rpy2 / emcee installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Root of the whole figure tree, anchored to this file rather than to the
# process CWD so the scripts can be run from anywhere.
FIGURE_ROOT = Path(__file__).resolve().parent / "figures"

# Flat dump of the PDE solver's single-panel physics checks (was
# ./figures_diffusiveChecks). Not run-type structured -- it has no run types.
CHECKS_DIR = FIGURE_ROOT / "pde-checks"

# run_type -> directory name. The numeric prefix exists so that `ls` and any
# glob sort these in run order; the old names ("replicate_results",
# "50-yr_avg_forcing_results", ...) sorted run_type 2 and 3 ahead of 1.
RUN_TYPE_DIRS = {
   1: "01_replicate",
   2: "02_forcing-50yravg",
   3: "03_lrfit-50yravg",
}

# Legacy per-run-type filename stamps. The run-type DIRECTORY already encodes
# the run type, and no filename in the tree is disambiguated by the stamp
# (verified: stripping the variant and run-type directory from the full file
# list leaves a set with no duplicates), so it is off by default. Flip
# INCLUDE_RUNTYPE_STAMP to True to restore `_50yrAvg` / `_50yrAvg_LRparamFit`
# on every filename -- useful if figures get copied out of the tree and need to
# carry their own provenance.
RUN_TYPE_STAMPS = {
   1: "",
   2: "50yrAvg",
   3: "50yrAvg_LRparamFit",
}
INCLUDE_RUNTYPE_STAMP = False

# Trailing variant stamp: `4xCO2_all_models_T2M_vs_NETTOA` fit by the MT2 model
# is written as `4xCO2_all_models_T2M_vs_NETTOA_MT2.png`. Unlike the run-type
# stamp above this is on, because the variant is the axis these figures are
# actually compared along -- the directory says which model produced a figure,
# but the directory is exactly what gets lost the moment two of them are opened
# side by side. Set to False to go back to bare names.
INCLUDE_VARIANT_STAMP = True

# Section names as constants, so a typo is an ImportError at the call site
# rather than a silently-created stray directory (lazy mkdir would happily
# honour "stpe1").
STEP1 = "step1"
STEP2 = "step2"
BUDGET = "budget"
CORNER = "corner"
UNBLINDED = "unblinded"
VALIDATION = "validation"
TABLES = "tables"
COMPARISON = "comparison"

#: Every section a RunLayout is allowed to write into.
SECTIONS = frozenset({STEP1, STEP2, BUDGET, CORNER, UNBLINDED, VALIDATION, TABLES})

# Two resolutions, deliberately not unified. The DPI_WIDE figures are already
# large-canvas single-model plots (budget bars are figsize=(13,8), corner plots
# are a 5x5 panel grid), so rendering them at 200 would inflate file size for no
# visible gain.
DPI_GRID = 200   # multi-panel model grids
DPI_WIDE = 150   # single-model wide figures: budget bars, corner posteriors


def figname(*parts, iteration: int | None = None) -> str:
   """Join filename fields with '_', dropping empty/None fields.

   The iteration tag is always emitted last so field order can't drift between
   call sites:

   >>> figname("4xCO2", "all_models", "T2M_vs_NETTOA")
   '4xCO2_all_models_T2M_vs_NETTOA'
   >>> figname("4xCO2", "all_models", "T2M_vs_NETTOA", iteration=3)
   '4xCO2_all_models_T2M_vs_NETTOA_iter03'
   >>> figname("4xCO2", "CCSM3", "surface_budget_bars")
   '4xCO2_CCSM3_surface_budget_bars'
   >>> figname("4xCO2", "all_models", "T2m_vs_t", "linear")
   '4xCO2_all_models_T2m_vs_t_linear'
   """
   stem = "_".join(str(p) for p in parts if p not in (None, ""))
   if not stem:
      raise ValueError("figname() needs at least one non-empty field")
   return f"{stem}_iter{iteration:02d}" if iteration is not None else stem


@dataclass(frozen=True)
class RunLayout:
   """One (variant, run_type, results-kind) output destination.

   Constructed once per iteration of each script's `for run_type / for results`
   loop and threaded through the body, replacing the old trio of `outdir`,
   `current_dir` and `suffix` locals.
   """

   variant: str
   run_type: int
   results: str = UNBLINDED
   root: Path = FIGURE_ROOT

   def __post_init__(self):
      if self.run_type not in RUN_TYPE_DIRS:
         raise ValueError(
            f"run_type must be one of {sorted(RUN_TYPE_DIRS)}, got {self.run_type!r}"
         )
      if self.results not in (UNBLINDED, VALIDATION):
         raise ValueError(
            f"results must be {UNBLINDED!r} or {VALIDATION!r}, got {self.results!r}"
         )

   # ------------------------------------------------------------------ identity

   @property
   def dir(self) -> Path:
      """This run type's directory: figures/<variant>/<NN_runtype>/."""
      return self.root / self.variant / RUN_TYPE_DIRS[self.run_type]

   @property
   def label(self) -> str:
      """Human-readable tag for the scripts' progress banners."""
      return f"{self.variant}/{RUN_TYPE_DIRS[self.run_type]} [{self.results}]"

   @property
   def stamp(self) -> str:
      """Trailing run-type stamp, empty unless INCLUDE_RUNTYPE_STAMP is on."""
      s = RUN_TYPE_STAMPS[self.run_type]
      return f"_{s}" if (s and INCLUDE_RUNTYPE_STAMP) else ""

   @property
   def variant_stamp(self) -> str:
      """Trailing variant stamp (`_MT2`), empty if INCLUDE_VARIANT_STAMP is off."""
      return f"_{self.variant}" if INCLUDE_VARIANT_STAMP else ""

   # -------------------------------------------------- pure path construction

   def path(self, section: str, name: str, ext: str = "png") -> Path:
      """Full path for one output. Pure -- creates nothing."""
      if section not in SECTIONS:
         raise ValueError(f"unknown section {section!r}; expected one of {sorted(SECTIONS)}")
      return self.dir / section / f"{name}{self.stamp}{self.variant_stamp}.{ext}"

   def results_path(self, name: str, ext: str = "png") -> Path:
      """Full path in whichever of unblinded/ or validation/ this pass writes."""
      return self.path(self.results, name, ext)

   def csv_path(self, name: str, *, mkdir: bool = True) -> Path:
      """Path for a parameter table under tables/, creating the dir by default."""
      p = self.path(TABLES, name, ext="csv")
      if mkdir:
         p.parent.mkdir(parents=True, exist_ok=True)
      return p

   # ------------------------------------------------------------------------ IO

   def save(
      self,
      fig,
      section: str,
      name: str,
      *,
      iteration: int | None = None,
      dpi: int = DPI_GRID,
      bbox_inches: str = "tight",
      ext: str = "png",
      close: bool = False,
      **savefig_kw,
   ) -> Path:
      """Save `fig` into <run dir>/<section>/, creating the directory on demand.

      `close=False` by default because the existing call sites already do their
      own plt.close(fig) -- pass close=True to hand that off to this method.
      """
      if iteration is not None:
         name = figname(name, iteration=iteration)
      p = self.path(section, name, ext)
      p.parent.mkdir(parents=True, exist_ok=True)
      fig.savefig(p, dpi=dpi, bbox_inches=bbox_inches, **savefig_kw)
      if close:
         import matplotlib.pyplot as plt  # local: keep this module stdlib-only to import
         plt.close(fig)
      return p

   def save_results(self, fig, name: str, **kw) -> Path:
      """save() into the results-kind section (unblinded/ or validation/)."""
      return self.save(fig, self.results, name, **kw)

   def reserve(self, section: str, name: str, ext: str = "png") -> Path:
      """mkdir + return a path, for helpers that run their own fig.savefig().

      Used by plot_surface_budget_bars() and run_mcmc_corner(), which take the
      destination as an argument.
      """
      p = self.path(section, name, ext)
      p.parent.mkdir(parents=True, exist_ok=True)
      return p


def comparison_dir(variant: str, root: Path = FIGURE_ROOT) -> Path:
   """figures/<variant>/comparison/ -- cross-run-type output, created on demand.

   Sorts after the 0N_* run-type directories, which is right: it summarises them.
   """
   d = root / variant / COMPARISON
   d.mkdir(parents=True, exist_ok=True)
   return d


def params_csv(variant: str, run_type: int, root: Path = FIGURE_ROOT) -> Path:
   """Locate a variant's fitted-parameter table for one run type.

   Prefers the converged per-iteration table (2013b) over the single-shot one
   (everything else). Because every script now writes into tables/, callers no
   longer need to branch on which variant they are reading.
   """
   layout = RunLayout(variant, run_type, root=root)
   for candidate in ("params_final", "fitted_model_params"):
      p = layout.path(TABLES, candidate, ext="csv")
      if p.exists():
         return p
   raise FileNotFoundError(
      f"no parameter table (params_final or fitted_model_params, stamped "
      f"{layout.variant_stamp!r}) under {layout.dir / TABLES}"
   )
