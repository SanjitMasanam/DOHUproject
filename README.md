# DOHUproject — Deep Ocean Heat Uptake

Fits a family of energy-balance models (EBMs) to LongRunMIP AOGCM output in order to compare how
the fitted parameters change with the fitting procedure and the calibration window. For each of 8
AOGCMs, the annual-mean 2-m air temperature (`T2M`) and net top-of-atmosphere imbalance
(`NETTOA`) from an abrupt-4xCO2 run — baselined against that model's `piControl` — are fit to:
the Geoffroy et al. (2013a) analytic two-box EBM; the Geoffroy et al. (2013b) two-box EBM with a
deep-ocean heat-uptake efficacy ε; a 1-layer slab mixed layer coupled to a diffusive thermocline
after Hansen et al. (1984), solved as a PDE; and an AMOC-strength-based (MT2) framework. Each
variant lives in its own `longrun_data_param_fit_script_*.py`, writes its fitted parameters to a
CSV and its diagnostics to `figures/<variant>/`, and shares the data loading, output-path layout
and plot styling with the others.

## Repository map

| Path | Role |
| --- | --- |
| [get_int_nettoa_longrun_data.R](get_int_nettoa_longrun_data.R) | Reads LongRunMIP NetCDF and writes the `.Rdata` bundle every Python script consumes |
| [data/](data/) | `int_netToa_longrun.Rdata` — the preprocessed per-model time series |
| [longrun_data_param_fit_script_2013a_allModels.py](longrun_data_param_fit_script_2013a_allModels.py) | Geoffroy 2013a two-box fit |
| [longrun_data_param_fit_script_2013b_allModels.py](longrun_data_param_fit_script_2013b_allModels.py) | Geoffroy 2013b two-box + efficacy fit |
| [longrun_data_param_fit_script_diffusive.py](longrun_data_param_fit_script_diffusive.py) | Slab + diffusive thermocline fit (no efficacy) |
| [longrun_data_param_fit_script_diffusive-eps.py](longrun_data_param_fit_script_diffusive-eps.py) | Same, with iterative ε calibration |
| [longrun_data_param_fit_script_diffusive-eps-joint.py](longrun_data_param_fit_script_diffusive-eps-joint.py) | Same physics, one joint nonlinear fit instead of the alternating loop |
| [longrun_data_param_fit_script_diffusive-eps-joint_fixedRadiativeParams.py](longrun_data_param_fit_script_diffusive-eps-joint_fixedRadiativeParams.py) | Joint fit with the radiative parameters pinned by Step 1 |
| [longrun_data_param_fit_script_diffusive-eps-corner.py](longrun_data_param_fit_script_diffusive-eps-corner.py) | MCMC posteriors over the same parameters (`emcee` + `corner`) |
| [longrun_data_param_fit_script_MT2_allModels.py](longrun_data_param_fit_script_MT2_allModels.py) | MT2 / AMOC-strength framework + two-layer ocean ODE |
| [pde_solver_1lyrEBM_diffusive.py](pde_solver_1lyrEBM_diffusive.py) | The PDE model the diffusive scripts import, plus a standalone verification suite |
| [pde_solver_1lyrEBM_diffusive_practice.py](pde_solver_1lyrEBM_diffusive_practice.py) | Self-contained teaching copy of the solver; nothing imports it |
| [figure_paths.py](figure_paths.py) | Canonical output layout: `RunLayout`, `figname`, section constants (stdlib-only) |
| [plot_style.py](plot_style.py) | Shared rcParams, font sizes, `format_ax`, `make_model_grid*`, `plot_regression_3d` |
| [compare_run_type_parameter_progression.py](compare_run_type_parameter_progression.py) | Reads the fitted-parameter CSVs back and plots each parameter across run types |
| [check_figure_paths.py](check_figure_paths.py) | AST audit of every output path the scripts can produce |
| [test_figure_paths.py](test_figure_paths.py) | Golden-path checks for `figure_paths` and the migration mapping |
| [migrate_figures.py](migrate_figures.py) | One-time migration of the legacy `figures_*/` trees into `figures/` |
| [figures/](figures/) | All figure and table output, one subtree per variant |

## Data pipeline

[get_int_nettoa_longrun_data.R](get_int_nettoa_longrun_data.R) opens one NetCDF per
(model, experiment) from `/archive/jpd/longrunmip`, pulls `T2M`, `NETTOA` and `TAX_YEARS` for
`piControl` and `4xCO2`, and saves everything as a nested R list in
`data/int_netToa_longrun.Rdata`. It needs the `ncdf4` R package and access to the archive, so it
runs on the GFDL analysis machine; the resulting `.Rdata` is committed, so the Python side runs
anywhere.

The Python scripts load that bundle through `rpy2` (`ro.r["load"](...)`, then
`ro.globalenv["int_nettoa_longrun_data"]`) and index it per model and experiment. The R script's
model list has 10 entries; the fit scripts use the 8 with usable series for both experiments —
`CCSM3`, `CESM1`, `CNRMCM6`, `ECHAM5`, `GISSE2R`, `IPSLCM5A`, `HadGEM2`, `MPIESM11` (`ECEARTH`
and `MIROC3` are left out). The data path is relative, so **run the scripts from the repo root**.

## The fit variants

Each script sets a `VARIANT` token, which becomes its directory under `figures/`, and loops over
run types and result kinds (see below).

| Script | `VARIANT` | Fits | Run types × results |
| --- | --- | --- | --- |
| `..._2013a_allModels.py` | `2013a` | Analytic two-box EBM, staged regressions | 3, 2, 1 × unblinded, validation |
| `..._2013b_allModels.py` | `2013b` | Two-box + efficacy ε, iterated to convergence (`MIN_ITERATIONS`/`MAX_ITERATIONS`, `CONVERGENCE_RTOL`); writes a table per iteration plus `params_final.csv` | 1, 2, 3 × unblinded, validation |
| `..._diffusive.py` | `diffusive` | κ and `h_ml` of the slab + thermocline PDE via `curve_fit`, plus κ/`h_ml`/`dz` sensitivity sweeps | 3, 2, 1 × unblinded |
| `..._diffusive-eps.py` | `diffusive-eps` | Same, alternating a radiative ε regression with a κ/`h_ml` refit until the parameters settle | 1, 2, 3 × unblinded |
| `..._diffusive-eps-joint.py` | `diffusive-eps-joint` | One `least_squares` over `[log10(κ), h_ml, ε, λ]` against a stacked T and Net-TOA residual, with `T_eq` pinned from Step 1 | 3, 2, 1 × unblinded |
| `..._diffusive-eps-joint_fixedRadiativeParams.py` | `diffusive-eps-joint_fixedRadiativeParams` | Reduced version of the same fit: only `[log10(κ), h_ml]` are free, `F_ref`/λ/ε/`T_eq` all fixed by Step 1 | 3, 2, 1 × unblinded |
| `..._diffusive-eps-corner.py` | `diffusive-eps-corner` | `emcee` sampling of the 5-parameter posterior (`MCMC_N_WALKERS`, `MCMC_N_STEPS`, `MCMC_BURN`, `MCMC_THIN`) with `corner` plots | 1, 2, 3 × unblinded |
| `..._MT2_allModels.py` | `MT2` | MT2 / MT2T model of Gregory et al. (2024): decomposes `N` into `N_M = pF` (from each model's AMOC strength `M_SV`) plus `N_T` from a two-layer ocean ODE, and derives `H = H_M + H_T` and OHUE `κ = κ_M + q`. Every panel carries both the paper's ensemble-mean constants (MT2) and per-AOGCM two-layer parameters fit by the paper's Appendix C.4.2 regression (MT2T). Runs the 4 models present in the paper's Table 1 | 1 × validation, `--unblinded` |

[compare_run_type_parameter_progression.py](compare_run_type_parameter_progression.py) sits on top
of these: for `2013a` and `2013b` it reads each run type's parameter table back and plots how
every parameter moves from one fitting procedure to the next.

## Run types and result kinds

`run_type` selects the calibration procedure and window; `results` selects how much of the series
the final plots show.

| `run_type` | Directory | Meaning |
| --- | --- | --- |
| 1 | `01_replicate` | Replicate Geoffroy: years 1–150 (late fit 30–150, early fit 1–10) |
| 2 | `02_forcing-50yravg` | 50-yr-averaged forcing; `T_eq` fit over the whole run, late fit still 30–150 |
| 3 | `03_lrfit-50yravg` | 50-yr avg + long-run fit; late window runs from year 50 to the end of the run |

| `results` | Meaning |
| --- | --- |
| `unblinded` | Full available time series |
| `validation` | First 151 years only — run by the two `2013*` scripts |

Within a run-type directory, output is grouped by stage: `step1/` (the Net-TOA-vs-T regressions
that set `F_ref`, λ and `T_eq`), `step2/` (the late-time exponential fits for `tau_s`/`a_s`),
`budget/` (per-model surface energy-budget bars), `corner/` (MCMC posteriors),
`unblinded/` or `validation/` (the final result figures), and `tables/` (parameter CSVs).

## Output layout

Every path in the project has one of four shapes, defined in [figure_paths.py](figure_paths.py):

```
figures/<variant>/<NN_runtype>/<section>/<name>.png
figures/<variant>/<NN_runtype>/tables/<name>.csv
figures/<variant>/comparison/<name>.{png,pdf}
figures/pde-checks/<name>.png
```

Scripts never build these paths as strings. Each pass constructs a
`RunLayout(VARIANT, run_type, results)` and writes through it — `layout.save(fig, STEP1, name)`,
`layout.save_results(fig, name)`, `layout.csv_path(name)`, `layout.reserve(...)` for helpers that
call `savefig` themselves — with filenames assembled by `figname(...)`. Sections are module
constants, run types and result kinds are validated on construction, and directories are created
by the writes, so a directory exists only if something was written into it. Cross-run-type output
goes to `comparison_dir(variant)`, and `params_csv(variant, run_type)` locates a variant's
parameter table without the caller knowing whether it is named `params_final.csv` or
`fitted_model_params.csv`.

`figure_paths.py` is deliberately stdlib-only, so the tooling can import it without matplotlib or
rpy2 installed. The visual side lives in [plot_style.py](plot_style.py): importing it applies the
project rcParams as a side effect and provides the shared font-size constants, `format_ax`, the
`make_model_grid` / `make_model_grid_with_ratio` / `make_model_grid_3d` panel constructors, and
`plot_regression_3d`.

`figures/pde-checks/` is the flat output of the PDE solver's own verification suite, which has no
run types.

## Environment

Python: `numpy`, `pandas`, `matplotlib`, `scipy`, `sympy`, `rpy2`, `tqdm`, plus `emcee` and
`corner` for the `-corner` variant — see [requirements.txt](requirements.txt). `rpy2` needs a
working system R. R: `ncdf4`, only for the preprocessing script.

On this machine, `~/anaconda3/envs/ldmx` (Python 3.11) carries the stack. The stdlib-only tools —
`figure_paths.py`, `test_figure_paths.py`, `check_figure_paths.py`, `migrate_figures.py` — run
under any Python 3.

## Running things

```bash
# Regenerate the data bundle (GFDL machine; needs /archive access)
Rscript get_int_nettoa_longrun_data.R

# A full fit variant, from the repo root
python3 longrun_data_param_fit_script_2013b_allModels.py

# Quick iteration on a model subset: first 2, first N, or named models
python3 longrun_data_param_fit_script_diffusive-eps-joint.py --test
python3 longrun_data_param_fit_script_diffusive-eps-joint.py --test 3
python3 longrun_data_param_fit_script_diffusive-eps-joint.py --test CCSM3,HadGEM2

# Cross-run-type parameter comparison (reads the tables the fits wrote)
python3 compare_run_type_parameter_progression.py

# PDE solver verification suite: PASS/FAIL checks + figures/pde-checks/
python3 pde_solver_1lyrEBM_diffusive.py

# Output-layout checks
python3 test_figure_paths.py
python3 check_figure_paths.py          # --list to print every path template
python3 migrate_figures.py             # --dry-run is the default; --apply to move
```

Two things worth knowing before running a fit: the scripts are top-level `for run_type / for
results` loops with no `if __name__ == "__main__"` guard, so importing one executes the entire
multi-hour fit — run them as scripts. And the diffusive variants fan their PDE solves (sensitivity
sweeps, finite-difference Jacobian columns) across a `multiprocessing` fork pool, so they will use
every core available.
