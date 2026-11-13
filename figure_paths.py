#!/usr/bin/env python3
"""Output layout for every figure and table in the project.

    figures/<variant>/<NN_runtype>/<section>/<name>_<variant>.png
    figures/<variant>/<NN_runtype>/tables/<name>_<variant>.csv
    figures/<variant>/comparison/<name>.{png,pdf}
    figures/<topic>/<name>.png                     (pde-checks, held, li2012)

Directories are created by the writes, so one exists only if something was
written into it. Stdlib-only, so the path tooling runs without matplotlib.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Anchored to this file, not the CWD.
FIGURE_ROOT = Path(__file__).resolve().parent / "figures"

# run_type -> directory name; the numeric prefix makes `ls` sort in run order.
RUN_TYPE_DIRS = {
   1: "01_replicate",
   2: "02_forcing-50yravg",
   3: "03_lrfit-50yravg",
}

# Append the variant to every RunLayout filename (`..._T2M_vs_NETTOA_MT2.png`),
# so the same plot from two variants stays distinguishable outside its directory.
INCLUDE_VARIANT_STAMP = True

# Sections are constants so a typo fails at import instead of creating a directory.
STEP1 = "step1"
STEP2 = "step2"
BUDGET = "budget"
UNBLINDED = "unblinded"
VALIDATION = "validation"
TABLES = "tables"
COMPARISON = "comparison"

#: Every section a RunLayout may write into.
SECTIONS = frozenset({STEP1, STEP2, BUDGET, UNBLINDED, VALIDATION, TABLES})

DPI_GRID = 200   # multi-panel model grids
DPI_WIDE = 150   # large single-model figures (budget bars, profiles)


def figname(*parts, iteration: int | None = None) -> str:
   """Join filename fields with '_', dropping empty ones; the iteration tag goes last.

   >>> figname("4xCO2", "all_models", "T2M_vs_NETTOA", iteration=3)
   '4xCO2_all_models_T2M_vs_NETTOA_iter03'
   """
   stem = "_".join(str(p) for p in parts if p not in (None, ""))
   if not stem:
      raise ValueError("figname() needs at least one non-empty field")
   return f"{stem}_iter{iteration:02d}" if iteration is not None else stem


@dataclass(frozen=True)
class RunLayout:
   """Output destination of one (variant, run_type, results-kind) pass."""

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

   @property
   def dir(self) -> Path:
      """figures/<variant>/<NN_runtype>/"""
      return self.root / self.variant / RUN_TYPE_DIRS[self.run_type]

   @property
   def label(self) -> str:
      """Tag for progress banners."""
      return f"{self.variant}/{RUN_TYPE_DIRS[self.run_type]} [{self.results}]"

   @property
   def variant_stamp(self) -> str:
      return f"_{self.variant}" if INCLUDE_VARIANT_STAMP else ""

   # ---- path construction (creates nothing unless stated) ----

   def path(self, section: str, name: str, ext: str = "png") -> Path:
      if section not in SECTIONS:
         raise ValueError(f"unknown section {section!r}; expected one of {sorted(SECTIONS)}")
      return self.dir / section / f"{name}{self.variant_stamp}.{ext}"

   def results_path(self, name: str, ext: str = "png") -> Path:
      """Path in unblinded/ or validation/, whichever this pass writes."""
      return self.path(self.results, name, ext)

   def csv_path(self, name: str, *, mkdir: bool = True) -> Path:
      """Path of a table under tables/; creates the directory by default."""
      p = self.path(TABLES, name, ext="csv")
      if mkdir:
         p.parent.mkdir(parents=True, exist_ok=True)
      return p

   def reserve(self, section: str, name: str, ext: str = "png") -> Path:
      """mkdir + path, for helpers that call fig.savefig() themselves."""
      p = self.path(section, name, ext)
      p.parent.mkdir(parents=True, exist_ok=True)
      return p

   # ---- writing ----

   def save(self, fig, section: str, name: str, *, iteration: int | None = None,
            dpi: int = DPI_GRID, bbox_inches: str = "tight", ext: str = "png",
            close: bool = False, **savefig_kw) -> Path:
      """Save `fig` into <run dir>/<section>/; close it if `close`."""
      if iteration is not None:
         name = figname(name, iteration=iteration)
      p = self.reserve(section, name, ext)
      fig.savefig(p, dpi=dpi, bbox_inches=bbox_inches, **savefig_kw)
      if close:
         import matplotlib.pyplot as plt  # local import keeps this module stdlib-only
         plt.close(fig)
      return p

   def save_results(self, fig, name: str, **kw) -> Path:
      """save() into unblinded/ or validation/."""
      return self.save(fig, self.results, name, **kw)


def comparison_dir(variant: str, root: Path = FIGURE_ROOT) -> Path:
   """figures/<variant>/comparison/ (cross-run-type output), created on demand."""
   d = root / variant / COMPARISON
   d.mkdir(parents=True, exist_ok=True)
   return d


def topic_dir(topic: str, root: Path = FIGURE_ROOT) -> Path:
   """figures/<topic>/ for scripts with no run types, created on demand."""
   d = root / topic
   d.mkdir(parents=True, exist_ok=True)
   return d


# Output of the PDE solver's verification suite. Not created here: importing
# this module must not touch the filesystem.
CHECKS_DIR = FIGURE_ROOT / "pde-checks"


def params_csv(variant: str, run_type: int, root: Path = FIGURE_ROOT) -> Path:
   """A variant's fitted-parameter table: params_final (2013b) or fitted_model_params."""
   layout = RunLayout(variant, run_type, root=root)
   for candidate in ("params_final", "fitted_model_params"):
      p = layout.path(TABLES, candidate, ext="csv")
      if p.exists():
         return p
   raise FileNotFoundError(
      f"no parameter table (params_final or fitted_model_params, stamped "
      f"{layout.variant_stamp!r}) under {layout.dir / TABLES}"
   )
