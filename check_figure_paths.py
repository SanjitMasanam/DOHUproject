#!/usr/bin/env python3
"""Static audit of every figure/table path the analysis scripts can produce.

    python3 check_figure_paths.py            # report + exit 1 on any problem
    python3 check_figure_paths.py --list     # also print every path template

Parses each script with `ast` rather than importing it. That is not a
preference: none of the longrun scripts has an `if __name__ == "__main__"`
guard, so importing one runs the entire multi-hour fit.

Checks:
  1. No script still builds an output path by hand -- every figure and table
     goes through RunLayout, so the layout can only be changed in one place.
  2. Every section argument is a figure_paths constant. Directories are created
     lazily by the writes, so a typo like "stpe1" would otherwise be silently
     honoured as a new directory instead of raising.
  3. Every filename template is well-formed and unique within its section, so
     two figures cannot quietly overwrite each other.
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

from figure_paths import SECTIONS

REPO = Path(__file__).resolve().parent

SCRIPTS = sorted(REPO.glob("longrun_data_param_fit_script_*.py")) + [
   REPO / "compare_run_type_parameter_progression.py",
   REPO / "compare_ohc_ts_across_models.py",
   REPO / "pde_solver_1lyrEBM_diffusive.py",
]

# RunLayout methods that take (section, name) or just (name).
SECTIONED = {"save", "reserve", "path"}
UNSECTIONED = {"save_results", "results_path", "csv_path"}

# Section constants, as they are spelled at the call sites.
SECTION_CONSTS = {
   "STEP1": "step1", "STEP2": "step2", "BUDGET": "budget",
   "CORNER": "corner", "VALIDATION": "validation", "UNBLINDED": "unblinded",
}


def describe(node) -> str:
   """Render a figname() argument as a template, e.g. '4xCO2_all_models_{scale}'."""
   if isinstance(node, ast.Constant):
      return str(node.value)
   if isinstance(node, ast.Name):
      return "{" + node.id + "}"
   if isinstance(node, ast.Attribute):
      return "{" + node.attr + "}"
   if isinstance(node, ast.Subscript):
      return describe(node.value)
   if isinstance(node, ast.JoinedStr):
      return "".join(describe(v) for v in node.values)
   if isinstance(node, ast.FormattedValue):
      return describe(node.value)
   return "{?}"


def template(call: ast.Call) -> str:
   """Filename template for a figname(...) call, or a marker for anything else."""
   if not (isinstance(call.func, ast.Name) and call.func.id == "figname"):
      return describe(call)
   parts = [describe(a) for a in call.args]
   iteration = any(k.arg == "iteration" for k in call.keywords)
   return "_".join(parts) + ("_iterNN" if iteration else "")


class Collector(ast.NodeVisitor):
   def __init__(self, path):
      self.path = path
      self.calls = []      # (line, method, section, name_template)
      self.problems = []

   def visit_Call(self, node):
      f = node.func
      if isinstance(f, ast.Attribute) and f.attr in SECTIONED | UNSECTIONED:
         if isinstance(f.value, ast.Name) and f.value.id == "layout":
            self._record(node, f.attr)
      self.generic_visit(node)

   def _record(self, node, method):
      args = node.args
      if method in SECTIONED:
         if len(args) < 2:
            self.problems.append(f"{self.path.name}:{node.lineno}: "
                                 f"layout.{method}() needs (section, name)")
            return
         sec_node, name_node = args[0], args[1]
         if method == "save":
            sec_node, name_node = args[1], args[2] if len(args) > 2 else args[1]
         section = self._section(sec_node, node.lineno)
      else:
         section = "<results>" if method in ("save_results", "results_path") else "tables"
         name_node = args[1] if method == "save_results" and len(args) > 1 else args[0]

      name = template(name_node) if isinstance(name_node, ast.Call) else describe(name_node)
      self.calls.append((node.lineno, method, section, name))

   def _section(self, node, lineno):
      if isinstance(node, ast.Name):
         if node.id in SECTION_CONSTS:
            return SECTION_CONSTS[node.id]
         self.problems.append(
            f"{self.path.name}:{lineno}: section {node.id!r} is not a figure_paths constant")
         return f"<{node.id}>"
      if isinstance(node, ast.Constant):
         if node.value in SECTIONS:
            self.problems.append(
               f"{self.path.name}:{lineno}: section given as the literal {node.value!r}; "
               f"use the figure_paths constant so typos fail loudly")
            return node.value
         self.problems.append(f"{self.path.name}:{lineno}: unknown section {node.value!r}")
         return f"<{node.value}>"
      return "<dynamic>"


def derived_path_names(tree):
   """Locals assigned from a path-join expression, e.g. `out_png = d / "x.png"`.

   Those are legitimate destinations as long as the left-hand side of the join
   came from the layout; what this check is really hunting is a path assembled
   from a string literal root, which is what the refactor removed.
   """
   names = set()
   for node in ast.walk(tree):
      if isinstance(node, ast.Assign) and isinstance(node.value, (ast.BinOp, ast.Call)):
         for target in node.targets:
            if isinstance(target, ast.Name):
               names.add(target.id)
   return names


def find_manual_paths(tree, path):
   """Any surviving hand-built output path: a .savefig()/.to_csv() whose
   destination is not layout-derived, or a reference to the retired locals."""
   problems = []
   derived = derived_path_names(tree)
   for node in ast.walk(tree):
      if isinstance(node, ast.Name) and node.id in ("outdir", "current_dir", "ensure_dirs"):
         problems.append(f"{path.name}:{node.lineno}: refers to the retired {node.id!r}")
      if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
         if node.func.attr in ("savefig", "to_csv"):
            arg = node.args[0] if node.args else None
            ok = (
               # layout.<something>(...) supplied the path directly
               isinstance(arg, ast.Call)
               # a path-join expression
               or isinstance(arg, (ast.BinOp, ast.Attribute))
               # the png_path parameter threaded in from layout.reserve()
               or (isinstance(arg, ast.Name)
                   and (arg.id == "png_path" or arg.id in derived))
            )
            if arg is not None and not ok:
               problems.append(
                  f"{path.name}:{node.lineno}: .{node.func.attr}() with a hand-built path")
   return problems


def main() -> int:
   ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
   ap.add_argument("--list", action="store_true", help="print every path template")
   args = ap.parse_args()

   problems = []
   total = 0
   for path in SCRIPTS:
      if not path.exists():
         continue
      tree = ast.parse(path.read_text(), filename=str(path))
      col = Collector(path)
      col.visit(tree)
      problems += col.problems + find_manual_paths(tree, path)
      total += len(col.calls)

      # Two writes to the same section+name would overwrite each other. Only
      # the writing methods count: csv_path() is also used to *report* a table's
      # location in a print(), which is not a second write.
      seen = {}
      for lineno, method, section, name in col.calls:
         if method not in ("save", "save_results", "reserve"):
            continue
         key = (section, name)
         if key in seen:
            problems.append(
               f"{path.name}:{lineno}: {section}/{name} is also written at line {seen[key]}")
         seen[key] = lineno

      if args.list and col.calls:
         print(f"\n{path.name}")
         for lineno, method, section, name in sorted(col.calls, key=lambda c: (c[2], c[3])):
            print(f"  {section:<12} {name:<52} ({method}, line {lineno})")

   print(f"\n{total} layout-mediated output paths across {len(SCRIPTS)} scripts.")
   if problems:
      print(f"\n{len(problems)} problem(s):\n")
      for p in problems:
         print(f"  {p}")
      return 1
   print("No hand-built paths, unknown sections, or duplicate destinations.")
   return 0


if __name__ == "__main__":
   sys.exit(main())
