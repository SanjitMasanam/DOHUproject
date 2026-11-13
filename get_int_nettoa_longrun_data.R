# Builds the .Rdata bundles the Python scripts read. Run from the repo root. Two datasets:
#
#   Rscript get_int_nettoa_longrun_data.R                LongRunMIP abrupt-4xCO2 + the GFDL runs
#                                                        -> ./data/int_netToa_longrun.Rdata
#   Rscript get_int_nettoa_longrun_data.R cmip_1pctCO2   1pctCO2 + piControl from /uda/cmip{3,5,6}
#                                                        -> ./data/int_netToa_1pctCO2.Rdata
#        --resolve-only   only find and check the input files (no reading: needs neither ncdf4 nor NCO)
#        --models A,B     only these labels (see cmip_runs below)
#
# For cmip_1pctCO2, run ./find_cmip_1pct_models.sh first: it prints the real center/model names
# and a sample path per archive, which is what cmip_runs and path_templates below must match.

args         = commandArgs(trailingOnly = TRUE)
dataset      = if (length(args) > 0 && !startsWith(args[1], "--")) args[1] else "longrunmip"
resolve_only = "--resolve-only" %in% args
i_models     = match("--models", args)
model_subset = if (is.na(i_models)) NULL else strsplit(args[i_models + 1], ",")[[1]]
if (!dataset %in% c("longrunmip", "cmip_1pctCO2"))
   stop("unknown dataset '", dataset, "': use longrunmip (default) or cmip_1pctCO2")
if (!is.null(model_subset) && anyNA(model_subset)) stop("--models needs a value, e.g. --models CESM1,CM3")
if (!(dataset == "cmip_1pctCO2" && resolve_only)) library(ncdf4)

# =================================================================================================
# CMIP 1pctCO2 + piControl (dataset = "cmip_1pctCO2")
# =================================================================================================
# Monthly tas, rsdt, rsut, rlut on regular lat/lon grids -> annual global means (cos-lat weights)
# T2M = tas and NETTOA = rsdt - rsut - rlut, stored per label under "piControl" and "1pctCO2" like
# the LongRunMIP bundle. A model that fails is reported and skipped; the rest are still saved.

# ---- Config: the part expected to need hand edits ------------------------------------------------
uda          = Sys.getenv("CMIP_UDA", "/uda")     # override to test against a copy of the tree
cmip_root    = c(cmip3 = file.path(uda, "cmip3"), cmip5 = file.path(uda, "cmip5"), cmip6 = file.path(uda, "cmip6"))
min_years    = 100                                # both experiments must have at least this many
spatial_mean = "nco"   # "nco": ncap2 does the area mean (needs `module load nco`); "r": ncdf4 in R
cmip_vars    = c("tas", "rsdt", "rsut", "rlut")
cmip_out     = "./data/int_netToa_1pctCO2.Rdata"

# label = key in the Rdata (the LongRunMIP names). member "-" = choose automatically (member_pref,
# then first in sort order); pi_member "-" = the same member as the 1pctCO2 run.
# Mapping caveats (check against find_cmip_1pct_models.sh):
#   CESM1    LongRunMIP's CESM1.0.4 has CAM4 physics, closest to CMIP5 CCSM4 (alternative: CESM1-CAM5)
#   MPIESM11 MPI-ESM1.1 sits between CMIP5 MPI-ESM-LR (1.0) and CMIP6 MPI-ESM1-2-LR
#   MIROC3   cmip3 has both miroc3_2_medres and miroc3_2_hires
#   HadGEM2  HadGEM2-ES assumed (cmip5 also has HadGEM2-CC)
cmip_runs = read.table(header = TRUE, colClasses = "character", text = "
label     era    center        model            member  pi_member
CCSM3     cmip3  -             ncar_ccsm3_0     -       -
CESM1     cmip5  NCAR          CCSM4            -       -
CNRMCM6   cmip6  CNRM-CERFACS  CNRM-CM6-1       -       -
ECEARTH   cmip5  ICHEC         EC-EARTH         -       -
ECHAM5    cmip3  -             mpi_echam5       -       -
GISSE2R   cmip5  NASA-GISS     GISS-E2-R        -       -
IPSLCM5A  cmip5  IPSL          IPSL-CM5A-LR     -       -
MIROC3    cmip3  -             miroc3_2_medres  -       -
HadGEM2   cmip5  MOHC          HadGEM2-ES       -       -
MPIESM11  cmip5  MPI-M         MPI-ESM-LR       -       -
CM3       cmip5  NOAA-GFDL     GFDL-CM3         -       -
ESM2M     cmip5  NOAA-GFDL     GFDL-ESM2M       -       -
ESM2G     cmip5  NOAA-GFDL     GFDL-ESM2G       -       -
")
member_pref = c("r1i1p1f1", "r1i1p1", "run1")

# Experiment names per archive; alternatives are tried in order (cmip3 has no plain 1pctCO2).
cmip_expts = list(
   cmip3 = list(piControl = "picntrl",   "1pctCO2" = c("1pctto4x", "1pctto2x")),
   cmip5 = list(piControl = "piControl", "1pctCO2" = "1pctCO2"),
   cmip6 = list(piControl = "piControl", "1pctCO2" = "1pctCO2"))
# Forcing of the perturbed run, read the same way as the LongRunMIP bundle's co2_multiple /
# ramp_years. 1pctCO2 ramps for the whole run (NA); the cmip3 runs hold CO2 once it is reached.
expt_forcing = list(
   "1pctCO2"  = list(co2_multiple = NA, ramp_years = NA),
   "1pctto4x" = list(co2_multiple = 4,  ramp_years = log(4)/log(1.01)),
   "1pctto2x" = list(co2_multiple = 2,  ramp_years = log(2)/log(1.01)))

# Path templates per archive, tried in order; the first with any match is used. Placeholders:
# {root} {center} {model} {expt} {member} {var} {grid} {version}; member/grid/version are
# wildcards unless fixed, and * matches within one path level. Only cmip6 follows the layout
# given for /uda; cmip5 and cmip3 are the usual ESGF/PCMDI layouts:
# VERIFY them against the sample paths printed by find_cmip_1pct_models.sh.
path_templates = list(
   cmip6 = c(
      "{root}/CMIP/{center}/{model}/{member}/Amon/{var}/{grid}/{version}/{var}_Amon_{model}_{expt}_{member}_{grid}_*.nc",
      "{root}/CMIP/{center}/{model}/{expt}/{member}/Amon/{var}/{grid}/{version}/{var}_Amon_{model}_{expt}_{member}_{grid}_*.nc"),
   cmip5 = c(
      "{root}/output1/{center}/{model}/{expt}/mon/atmos/Amon/{member}/{version}/{var}/{var}_Amon_{model}_{expt}_{member}_*.nc",
      "{root}/output1/{center}/{model}/{expt}/mon/atmos/Amon/{member}/{var}/{var}_Amon_{model}_{expt}_{member}_*.nc",
      "{root}/output1/{center}/{model}/{expt}/mon/atmos/{var}/{member}/{var}_Amon_{model}_{expt}_{member}_*.nc"),
   cmip3 = c(
      "{root}/{expt}/atm/mo/{var}/{model}/{member}/{var}_A1*.nc"))

# ---- Helpers -------------------------------------------------------------------------------------
cmip_fail = function(ctx, ...) stop(paste0("[", ctx, "] ", ...), call. = FALSE)
ym_str    = function(m) sprintf("%04d-%02d", m %/% 12, m %% 12 + 1)     # month index -> "YYYY-MM"
re_escape = function(s) gsub("([.+()^$|?{}\\[\\]\\\\])", "\\\\\\1", s, perl = TRUE)

fill_template = function(tmpl, fields){
   for (k in names(fields)) tmpl = gsub(paste0("{", k, "}"), fields[[k]], tmpl, fixed = TRUE)
   tmpl
}

# Regex equivalent of a template: fixed fields literal, wildcard fields captured (a field used
# twice, e.g. the member in the directory and the file name, must be the same both times).
template_regex = function(tmpl, fields){
   toks   = regmatches(tmpl, gregexpr("\\{[a-z]+\\}|\\*|[^{*]+", tmpl))[[1]]
   groups = character(0)
   re     = ""
   for (tok in toks){
      if (tok == "*"){ re = paste0(re, "[^/]*"); next }
      if (!startsWith(tok, "{")){ re = paste0(re, re_escape(tok)); next }
      key = substr(tok, 2, nchar(tok) - 1)
      if (fields[[key]] != "*")  re = paste0(re, re_escape(fields[[key]]))
      else if (key %in% groups)  re = paste0(re, "\\", match(key, groups))
      else { groups = c(groups, key); re = paste0(re, "([^/_]+)") }
   }
   list(re = paste0("^", re, "$"), groups = groups)
}

# For a glob that matches nothing: the deepest prefix that does exist, and what is in it.
explain_glob = function(glob){
   parts = strsplit(glob, "/", fixed = TRUE)[[1]]
   hits  = "/"
   for (i in 2:length(parts)){
      nxt = Sys.glob(paste(parts[1:i], collapse = "/"))
      if (!length(nxt)){
         dir  = hits[1]
         kids = list.files(dir)
         more = if (length(kids) > 20) sprintf(" ... (%d entries)", length(kids)) else ""
         return(sprintf("      exists up to: %s%s\n      wanted next:  %s\n      contents:     %s%s",
                        dir, if (length(hits) > 1) sprintf("  (first of %d matches)", length(hits)) else "",
                        parts[i], if (length(kids)) paste(head(kids, 20), collapse = " ") else "(empty)", more))
      }
      hits = nxt
   }
   "      (the directories exist, but no file name matches)"
}

# Start/end month index from CMIP5/6 file names (..._YYYYMM-YYYYMM.nc); NA when absent (cmip3).
file_months = function(paths){
   m = regmatches(paths, regexec("_([0-9]{4,})([0-9]{2})-([0-9]{4,})([0-9]{2})\\.nc$", paths))
   t(vapply(m, function(x) if (length(x) == 5)
      c(as.integer(x[2]) * 12L + as.integer(x[3]) - 1L, as.integer(x[4]) * 12L + as.integer(x[5]) - 1L)
      else c(NA_integer_, NA_integer_), integer(2)))
}

pick_one = function(vals, prefer = character(0), latest = FALSE){
   u = sort(unique(vals[!is.na(vals)]))
   if (length(u) <= 1) return(list(chosen = if (length(u)) u else NA, others = character(0)))
   p  = intersect(prefer, u)
   ch = if (length(p)) p[1] else if (latest) tail(u, 1) else u[1]
   list(chosen = ch, others = setdiff(u, ch))
}

# All files of one (run, experiment, variable): template match, then one member / grid / version,
# then date order and contiguity. Returns the files plus what was chosen.
resolve_one = function(run, expt, var, member){
   ctx    = paste(run$label, expt, var)
   fields = list(root = cmip_root[[run$era]], center = run$center, model = run$model, expt = expt,
                 var = var, member = member, grid = "*", version = "*")
   tmpls  = path_templates[[run$era]]
   if (is.null(tmpls)) cmip_fail(ctx, "no path_templates for era '", run$era, "'")
   df = NULL
   for (tmpl in tmpls){
      hits = Sys.glob(fill_template(tmpl, fields))
      if (!length(hits)) next
      tr = template_regex(tmpl, fields)
      m  = regmatches(hits, regexec(tr$re, hits, perl = TRUE))
      ok = lengths(m) > 0
      if (!any(ok)) next
      df = data.frame(path = hits[ok], stringsAsFactors = FALSE)
      for (g in c("member", "grid", "version"))
         df[[g]] = if (g %in% tr$groups) vapply(m[ok], `[`, "", 1 + match(g, tr$groups))
                   else if (fields[[g]] != "*") fields[[g]] else NA_character_
      break
   }
   if (is.null(df)){
      why = vapply(seq_along(tmpls), function(i){
         g = fill_template(tmpls[i], fields)
         sprintf("   template %d: %s\n%s", i, g, explain_glob(g))
      }, "")
      cmip_fail(ctx, "no files matched any ", run$era, " template",
                if (member != "*") paste0(" for member ", member) else "", "\n",
                paste(why, collapse = "\n"),
                "\n   Fix the center/model/member in cmip_runs or path_templates$", run$era,
                " (see find_cmip_1pct_models.sh).")
   }

   notes = character(0)
   choose = function(col, prefer = character(0), latest = FALSE){
      p = pick_one(df[[col]], prefer, latest)
      if (length(p$others)) notes <<- c(notes, sprintf("%s %s (also: %s)", col, p$chosen, paste(p$others, collapse = ", ")))
      if (!is.na(p$chosen)) df <<- df[!is.na(df[[col]]) & df[[col]] == p$chosen, , drop = FALSE]
      p$chosen
   }
   mem = choose("member", member_pref)
   grd = choose("grid", "gn")
   ver = choose("version", latest = TRUE)

   mo = file_months(df$path)
   o  = order(mo[, 1], df$path)
   df = df[o, , drop = FALSE]; mo = mo[o, , drop = FALSE]
   nmonths = NA
   if (!anyNA(mo)){
      bad = which(mo[-1, 1] != mo[-nrow(mo), 2] + 1)
      if (length(bad)){
         i = bad[1]
         cmip_fail(ctx, if (mo[i + 1, 1] > mo[i, 2] + 1) "gap" else "overlap", " between\n   ",
                   df$path[i], " (ends ", ym_str(mo[i, 2]), ")\n   ",
                   df$path[i + 1], " (starts ", ym_str(mo[i + 1, 1]), ")")
      }
      nmonths = mo[nrow(mo), 2] - mo[1, 1] + 1
      if (nmonths / 12 < min_years)
         cmip_fail(ctx, sprintf("only %.1f years (%s to %s, %d files); need %d. Member %s -- set member in cmip_runs to try another.",
                                nmonths / 12, ym_str(mo[1, 1]), ym_str(mo[nrow(mo), 2]), nrow(df), min_years, mem))
   }
   cat(sprintf("  %-9s %-5s %s%s%s: %d file(s), %s\n", expt, var, mem,
               if (!is.na(grd)) paste0(" ", grd) else "", if (!is.na(ver)) paste0(" ", ver) else "", nrow(df),
               if (is.na(nmonths)) "no dates in file names (checked when read)"
               else sprintf("%s to %s (%.1f yr)", ym_str(mo[1, 1]), ym_str(mo[nrow(mo), 2]), nmonths / 12)))
   for (n in notes) cat("              chose", n, "\n")
   list(files = df$path, member = mem, grid = grd, version = ver)
}

# Every (experiment, variable) of one run. The 1pctCO2 tas member fixes the member for the rest.
resolve_run = function(run){
   member = if (run$member == "-") "*" else run$member
   errs   = character(0)
   first  = NULL
   for (e in cmip_expts[[run$era]][["1pctCO2"]]){
      first = tryCatch(resolve_one(run, e, "tas", member), error = function(err){ errs <<- c(errs, conditionMessage(err)); NULL })
      if (!is.null(first)){ pert = e; break }
   }
   if (is.null(first)) stop(paste(errs, collapse = "\n"), call. = FALSE)
   members = c(piControl = if (run$pi_member == "-") first$member else run$pi_member, "1pctCO2" = first$member)
   names_  = c(piControl = cmip_expts[[run$era]]$piControl, "1pctCO2" = pert)
   out = list()
   for (ex in c("1pctCO2", "piControl")){
      out[[ex]] = list(experiment = names_[[ex]], member = members[[ex]])
      for (v in cmip_vars)
         out[[ex]][[v]] = if (ex == "1pctCO2" && v == "tas") first else resolve_one(run, names_[[ex]], v, members[[ex]])
   }
   out
}

# Day count of one unit of a CF time axis ("days since ...", "hours since ...").
time_unit_days = function(units){
   u = tolower(sub(" since.*", "", units))
   switch(u, days = 1, day = 1, hours = 1/24, hour = 1/24, minutes = 1/1440, seconds = 1/86400, NA)
}

# Year and month of a CF time value, calendar-aware enough for picking the first January.
decode_ym = function(t, units, calendar){
   m = regmatches(units, regexec("since\\s+(-?[0-9]+)-([0-9]+)-([0-9]+)", units))[[1]]
   s = time_unit_days(units)
   if (length(m) != 4 || is.na(s)) return(c(NA, NA))
   y0 = as.integer(m[2]); m0 = as.integer(m[3]); d0 = as.integer(m[4])
   days = floor(t * s)
   cal  = tolower(calendar)
   ml   = if (cal == "360_day") rep(30, 12)
          else if (cal %in% c("noleap", "365_day")) c(31,28,31,30,31,30,31,31,30,31,30,31)
          else if (cal %in% c("all_leap", "366_day")) c(31,29,31,30,31,30,31,31,30,31,30,31)
          else NULL
   if (is.null(ml)){                                  # standard / gregorian / proleptic_gregorian
      d = as.Date(sprintf("%04d-%02d-%02d", y0, m0, d0)) + days
      return(c(as.integer(format(d, "%Y")), as.integer(format(d, "%m"))))
   }
   tot = sum(ml[seq_len(m0 - 1)]) + d0 - 1 + days
   c(y0 + tot %/% sum(ml), findInterval(tot %% sum(ml), cumsum(c(0, ml))))
}

# Area-weighted global mean of a (lon, lat, time) variable, by ncap2 or in R.
nco_global_mean = function(path, var, latn, lonn, ctx){
   tmp  = tempfile(fileext = ".nc")
   on.exit(unlink(tmp))
   expr = sprintf("w=cos(%s*3.14159265358979/180.0); gm=(%s*w).total($%s,$%s)/(w.total()*$%s.size);",
                  latn, var, latn, lonn, lonn)
   out  = suppressWarnings(system2("ncap2", c("-O", "-v", "-s", shQuote(expr), shQuote(path), shQuote(tmp)),
                                   stdout = TRUE, stderr = TRUE))
   st   = attr(out, "status")
   if (!is.null(st) && st != 0)
      cmip_fail(ctx, "ncap2 failed (exit ", st, ") on ", path, ":\n   ", paste(out, collapse = "\n   "),
                "\n   (set spatial_mean = \"r\" at the top of the script to average in R instead)")
   nc = nc_open(tmp)
   on.exit(nc_close(nc), add = TRUE, after = FALSE)
   as.vector(ncvar_get(nc, "gm"))
}

r_global_mean = function(nc, var, lat, ilat, ilon, itime){
   nt  = nc$var[[var]]$dim[[itime]]$len
   w   = cos(lat * pi / 180)
   out = numeric(nt)
   for (s in seq(1, nt, by = 120)){                   # 10 years at a time keeps memory small
      n = min(120, nt - s + 1)
      start = c(1, 1, 1); count = c(-1, -1, -1); start[itime] = s; count[itime] = n
      a = ncvar_get(nc, var, start = start, count = count, collapse_degen = FALSE)
      a = aperm(a, c(ilon, ilat, itime))              # -> lon, lat, time
      out[s:(s + n - 1)] = colSums(colMeans(a, dims = 1) * w) / sum(w)
   }
   out
}

# One file: monthly global means plus its time axis and month lengths (from the time bounds).
read_global_mean = function(path, var, ctx){
   nc = nc_open(path)
   on.exit(nc_close(nc))
   v  = nc$var[[var]]
   if (is.null(v)) cmip_fail(ctx, "variable '", var, "' not in ", path, " (has: ", paste(names(nc$var), collapse = ", "), ")")
   dn    = vapply(v$dim, `[[`, "", "name")
   ilat  = grep("^lat", dn, ignore.case = TRUE)
   ilon  = grep("^lon", dn, ignore.case = TRUE)
   itime = grep("^time", dn, ignore.case = TRUE)
   if (length(dn) != 3 || length(ilat) != 1 || length(ilon) != 1 || length(itime) != 1)
      cmip_fail(ctx, "expected dimensions (lon, lat, time), found (", paste(dn, collapse = ", "), ") in ", path,
                "\n   This script assumes a regular lat/lon grid.")
   tdim = v$dim[[itime]]
   cal  = ncatt_get(nc, dn[itime], "calendar")
   cal  = if (cal$hasatt) cal$value else "standard"
   bn   = ncatt_get(nc, dn[itime], "bounds")
   bn   = if (bn$hasatt) bn$value else intersect(c("time_bnds", "time_bounds"), names(nc$var))[1]
   w    = if (!is.na(bn) && !is.null(nc$var[[bn]])){ b = ncvar_get(nc, bn); b[2, ] - b[1, ] } else NULL
   gm   = if (spatial_mean == "nco") nco_global_mean(path, var, dn[ilat], dn[ilon], ctx)
          else r_global_mean(nc, var, v$dim[[ilat]]$vals, ilat, ilon, itime)
   if (length(gm) != tdim$len) cmip_fail(ctx, "global mean has ", length(gm), " steps, time axis ", tdim$len, " in ", path)
   list(x = gm, time = tdim$vals, units = tdim$units, calendar = cal, w = w)
}

# Monthly pieces (files in order) -> annual means from the first January, weighted by month length.
annualise = function(pieces, ctx){
   x     = unlist(lapply(pieces, `[[`, "x"))
   p1    = pieces[[1]]
   units = vapply(pieces, `[[`, "", "units")
   if (length(unique(units)) == 1){
      t  = unlist(lapply(pieces, `[[`, "time")) * time_unit_days(p1$units)
      if (any(diff(t) <= 0)) cmip_fail(ctx, "time axis is not increasing across the files (overlap or wrong order)")
      dt = median(diff(t))
      if (length(t) > 1 && (dt < 27 || dt > 32)) cmip_fail(ctx, sprintf("time step is %.1f days; expected monthly data", dt))
   } else cat("  note:", ctx, "files use different time units; their order was not checked\n")
   have_w = all(!vapply(pieces, function(p) is.null(p$w), TRUE))
   w      = if (have_w) unlist(lapply(pieces, `[[`, "w")) else rep(1, length(x))
   if (!have_w) cat("  note:", ctx, "has no time bounds; months weighted equally\n")
   ym = decode_ym(p1$time[1], p1$units, p1$calendar)
   if (anyNA(ym)){
      cat("  note:", ctx, "could not decode the start date from '", p1$units, "'; assuming January\n")
      ym = c(NA, 1)
   }
   skip = (13 - ym[2]) %% 12                         # months before the first January
   nyr  = (length(x) - skip) %/% 12
   if (nyr < 1) cmip_fail(ctx, "less than one full year of data")
   rest = length(x) - skip - 12 * nyr
   if (skip || rest) cat(sprintf("  note: %s dropped %d leading and %d trailing months (partial years)\n", ctx, skip, rest))
   idx = skip + seq_len(12 * nyr)
   X = matrix(x[idx], nrow = 12); W = matrix(w[idx], nrow = 12)
   list(x = colSums(X * W) / colSums(W), start = ym[1] + (skip > 0))
}

# Read, average and combine all variables of a resolved run.
process_run = function(run, res){
   out = list()
   for (ex in c("piControl", "1pctCO2")){
      r   = res[[ex]]
      ann = list()
      for (v in cmip_vars){
         ctx = paste(run$label, r$experiment, v)
         cat(sprintf("  reading %-9s %-5s (%d file(s))\n", r$experiment, v, length(r[[v]]$files)))
         ann[[v]] = annualise(lapply(r[[v]]$files, read_global_mean, var = v, ctx = ctx), ctx)
      }
      # Align the variables on common years (by calendar year when known, else from the start).
      starts = vapply(ann, `[[`, 0, "start")
      lens   = vapply(ann, function(a) length(a$x), 0L)
      if (anyNA(starts)) starts[] = 0
      y0 = max(starts); y1 = min(starts + lens - 1)
      if (any(starts != y0) || any(starts + lens - 1 != y1))
         cat(sprintf("  note: %s %s years differ by variable (%s); kept the common %d years\n", run$label, r$experiment,
                     paste(sprintf("%s=%d", names(lens), lens), collapse = ", "), y1 - y0 + 1))
      ann = mapply(function(a, s) a$x[(y0 - s + 1):(y1 - s + 1)], ann, starts, SIMPLIFY = FALSE)
      n   = y1 - y0 + 1
      if (n < min_years) cmip_fail(paste(run$label, r$experiment), sprintf("only %d full years after reading; need %d", n, min_years))
      out[[ex]] = list(T2M = ann$tas, NETTOA = ann$rsdt - ann$rsut - ann$rlut,
                       years = if (y0 == 0) NA else y0:y1, experiment = r$experiment, member = r$tas$member,
                       file = unlist(lapply(cmip_vars, function(v) r[[v]]$files)))
   }
   f = expt_forcing[[res[["1pctCO2"]]$experiment]]
   if (is.null(f)) f = list(co2_multiple = NA, ramp_years = NA)
   c(out, list(source_model = run$model, center = run$center, era = run$era,
               co2_multiple = f$co2_multiple, ramp_years = f$ramp_years, co2_pct_per_yr = 1))
}

# ---- Main ----------------------------------------------------------------------------------------
if (dataset == "cmip_1pctCO2"){
   runs = cmip_runs
   if (!is.null(model_subset)){
      bad = setdiff(model_subset, runs$label)
      if (length(bad)) stop("unknown label(s) in --models: ", paste(bad, collapse = ", "),
                            " (labels: ", paste(runs$label, collapse = ", "), ")")
      runs = runs[runs$label %in% model_subset, , drop = FALSE]
   }
   if (!resolve_only){
      if (spatial_mean == "nco" && Sys.which("ncap2") == "")
         stop("ncap2 not found: run `module load nco` first, or set spatial_mean = \"r\"")
      if (!dir.exists(dirname(cmip_out))) stop("no ", dirname(cmip_out), " directory: run from the repo root")
   }
   for (era in unique(runs$era))
      if (!dir.exists(cmip_root[[era]])) cat("WARNING:", cmip_root[[era]], "does not exist; its models will fail\n")

   cmip_data = list()
   failures  = character(0)
   for (k in seq_len(nrow(runs))){
      run = as.list(runs[k, ])
      cat(sprintf("\n=== %s: %s %s %s\n", run$label, run$era, run$center, run$model))
      d = tryCatch({
         res = resolve_run(run)
         if (resolve_only) res else process_run(run, res)
      }, error = function(e){
         failures[[run$label]] <<- conditionMessage(e)
         cat("FAILED:\n", conditionMessage(e), "\n", sep = "")
         NULL
      })
      if (is.null(d) || resolve_only) next
      cmip_data[[run$label]] = d
      cat(sprintf("  piControl %d yr: mean T2M %.2f, NETTOA %.2f W/m2 | %s (%s) %d yr: NETTOA yrs 1-5 %.2f W/m2\n",
                  length(d$piControl$T2M), mean(d$piControl$T2M), mean(d$piControl$NETTOA),
                  d[["1pctCO2"]]$experiment, d[["1pctCO2"]]$member, length(d[["1pctCO2"]]$T2M),
                  mean(d[["1pctCO2"]]$NETTOA[1:5])))
   }

   cat("\n=== Summary", if (resolve_only) "(--resolve-only: nothing read or saved)", "\n")
   for (lab in runs$label)
      cat(sprintf("  %-7s %-9s %s\n", if (lab %in% names(failures)) "FAILED" else "OK", lab,
                  if (lab %in% names(failures)) strsplit(failures[[lab]], "\n")[[1]][1] else ""))
   if (!resolve_only){
      if (length(cmip_data)){
         models = names(cmip_data); Nmodels = length(models)
         expts  = c("piControl", "1pctCO2"); vars = c("T2M", "NETTOA")
         int_nettoa_1pctCO2_data = cmip_data
         save(models, Nmodels, expts, vars, int_nettoa_1pctCO2_data, file = cmip_out)
         cat("Saved", Nmodels, "model(s) to", cmip_out, "\n")
      } else cat("Nothing to save.\n")
   }
   if (length(failures)) cat(length(failures), "model(s) failed; the full messages are above.\n")
   quit(save = "no", status = if (length(failures)) 1 else 0)
}

# =================================================================================================
# LongRunMIP abrupt-4xCO2 + GFDL runs (dataset = "longrunmip", the default)
# =================================================================================================
datadir = "/archive/jpd/longrunmip"
homedir = "/home/Sanjit.Masanam/Documents/DeepOceanHeatUptakeProject"
models = c("CCSM3","CESM1","CNRMCM6","ECEARTH","ECHAM5","GISSE2R","IPSLCM5A","MIROC3","HadGEM2","MPIESM11", # exclude FAMOUS, HadCM3, MPIESM12 for now
           "CM3","ESM2M","ESM2G") # GFDL, from gfdldir below
setwd(datadir)

# GFDL runs (Nadir Jeevanjee). These are not abrupt-4xCO2: CM3 and ESM2M are abrupt 2xCO2, ESM2G
# ramps CO2 at 1%/yr to 4x and then holds it. They are still stored under the "4xCO2" key, so
# the Python loaders need no change, with co2_multiple / ramp_years recorded per model below.
# Files are global means (lat = lon = 1); {var} is replaced by the variable name, for ESM2G's
# one-file-per-variable layout.
gfdldir     = "/work/Nadir.Jeevanjee/23sensitivity"
gfdl_models = c("CM3","ESM2M","ESM2G")
gfdl_files  = list(
   CM3   = list(piControl = "CM3/CM3_CTL_TS.nc",                  "4xCO2" = "CM3/CM3_2X_TS.nc"),
   ESM2M = list(piControl = "ESM2M/ESM2M_CTL_TS.nc",              "4xCO2" = "ESM2M/ESM2M_2X_TS.nc"),
   ESM2G = list(piControl = "ESM2G/{var}_ESM2G_pi-control_C2.nc", "4xCO2" = "ESM2G/{var}_ESM2G-C2_1pct-co2_4x_U2.nc"))
co2_multiple = c(CM3 = 2, ESM2M = 2)              # default 4
ramp_years   = c(ESM2G = log(4)/log(1.01))        # 1%/yr to 4x = 139.3 yr; default 0 (abrupt)

# One GFDL variable as an annual series. Annual records pass through; monthly ones (365_day
# calendar, assumed to start in January) are averaged into years weighted by month length.
read_gfdl_var = function(pattern, var){
   file = file.path(gfdldir, gsub("{var}", var, pattern, fixed = TRUE))
   nc   = nc_open(file)
   x    = as.vector(ncvar_get(nc, var))
   dt   = median(diff(ncvar_get(nc, "time")))     # days
   nc_close(nc)
   if (abs(dt - 365) < 1) return(x)
   if (dt >= 28 && dt <= 31){
      ndays = c(31,28,31,30,31,30,31,31,30,31,30,31)
      nyr   = length(x) %/% 12
      return(colSums(matrix(x[1:(12*nyr)], nrow = 12) * ndays) / 365)
   }
   stop(paste("Unexpected time step of", dt, "days in", file))
}

tcr_vals  = c(1.5, 1.7, 2.1, NA, 2.2, 1.5, 2.0, 2.1, 2.5, 2) # from Google sheet
ecs_vals  = c(2.7, 2.9, 4.6, 3.3, 3.4, 2.2, 4.1, 4, 4.6 ,3.6)
Teq_vals  = c(NA,  5.8, NA,  NA, NA,  4.3, 8.1, NA, 9.1, 7.3)
lambda_vals=c(NA, 1.24, NA,  NA, NA,  1.7,0.79, NA,0.65, 1.14)
C_vals    = c(NA,  6.1, NA,  NA, NA,  4.7, 7.7, NA, 6.5, 7.3)
tauf_vals = c(NA,  2.8, NA,  NA, NA,  1.6, 5.5, NA, 5.3, 3.9)
taus_vals  = c(NA, 132, NA,  NA, NA,  184, 286, NA, 280, 164)

expts       = c("piControl","4xCO2")
Nmodels     = length(models)
vars        = c("T2M","NETTOA", "TAX_YEARS")
params      = c("tcr","ecs","Teq","lambda","C","tauf","taus")

# Dunne's script is at /net/jpd/ECS_analysis/ECS_calc_comparison_drift2_longrunmip_short_cumsum.m

CCSM3_piControl_nc    = "CCSM3_g025_piControl_1530_nettoa_t2m.nc"
CCSM3_4xCO2_nc        = "CCSM3_g025_abrupt4x_2120_nettoa_t2m.nc"
CESM1_piControl_nc    = "CESM104_piControl_1000_nettoa_t2m.nc"
CESM1_4xCO2_nc        = "CESM104_abrupt4x_5900_nettoa_t2m.nc"
CNRMCM6_piControl_nc  = "CNRMCM61_g025_piControl_2000_nettoa_t2m.nc"
CNRMCM6_4xCO2_nc      = "CNRMCM61_g025_abrupt4x_1850_nettoa_t2m.nc"
ECEARTH_piControl_nc  = "ECEARTH_g025_piControl_508_nettoa_t2m.nc"
ECEARTH_4xCO2_nc      = "ECEARTH_g025_abrupt4x_150_nettoa_t2m.nc"
ECHAM5_piControl_nc   = "ECHAM5MPIOM_piControl_100_nettoa_t2m.nc"
ECHAM5_4xCO2_nc       = "ECHAM5MPIOM_abrupt4x_1001_nettoa_t2m.nc"
GISSE2R_piControl_nc  = "GISSE2R_g025_piControl_5001_nettoa_t2m.nc"
GISSE2R_4xCO2_nc      = "GISSE2R_g025_abrupt4x_5001_nettoa_t2m.nc"
IPSLCM5A_piControl_nc = "IPSLCM5A_g025_piControl_1000_nettoa_t2m.nc"
IPSLCM5A_4xCO2_nc     = "IPSLCM5A_g025_abrupt4x_1000_nettoa_t2m.nc"
MIROC3_piControl_nc   = "MIROC32_g025_piControl_680_nettoa_t2m.nc"
MIROC3_4xCO2_nc       = "MIROC32_g025_abrupt4x_150_nettoa_t2m.nc"
HadGEM2_piControl_T2M_nc    = "HadGEM2_piControl_3000_t2m_g025.nc"
HadGEM2_piControl_NETTOA_nc = "HadGEM2_piControl_3000_nettoa_g025.nc"
HadGEM2_4xCO2_T2M_nc    = "HadGEM2_abrupt4x_3000_t2m_g025.nc"
HadGEM2_4xCO2_NETTOA_nc = "HadGEM2_abrupt4x_3000_nettoa_g025.nc"
MPIESM11_piControl_T2M_nc    = "MPIESM11_piControl_3000_t2m_g025.nc"
MPIESM11_piControl_NETTOA_nc = "MPIESM11_piControl_3000_nettoa_g025.nc"
MPIESM11_4xCO2_T2M_nc    = "MPIESM11_abrupt4x_3000_t2m_g025.nc"
MPIESM11_4xCO2_NETTOA_nc = "MPIESM11_abrupt4x_3000_nettoa_g025.nc"
 
int_nettoa_longrun_data = list() 

for (n_model in 1:Nmodels){
    model = models[n_model]
    print(paste("Model",model))
    int_nettoa_longrun_data[[model]] = list()
    # for (param in params){
    #     int_nettoa_longrun_data[[model]][[param]] = eval(as.name(paste(param,"_vals",sep="")))[n_model]
    # }
    
    for (expt in expts){
    	print(paste("Expt",expt))
    	int_nettoa_longrun_data[[model]][[expt]] = list()
	if (model %in% gfdl_models){
	   pattern = gfdl_files[[model]][[expt]]
	   int_nettoa_longrun_data[[model]][[expt]]$file = file.path(gfdldir, pattern)
	   # Read all four variables and cut the longer ones to the shortest, so T2M and
	   # NETTOA cover the same years (files for one run can differ in length).
	   v    = lapply(c(t_ref = "t_ref", swdn_toa = "swdn_toa", swup_toa = "swup_toa", olr = "olr"),
	                 function(var) read_gfdl_var(pattern, var))
	   lens = sapply(v, length)
	   n    = min(lens)
	   if (any(lens != n))
	      print(paste0("  ", expt, ": lengths ", paste(names(lens), lens, sep = "=", collapse = ", "),
	                   "; truncated to ", n, " yr"))
	   v = lapply(v, function(x) x[1:n])
	   int_nettoa_longrun_data[[model]][[expt]]$T2M    = v$t_ref
	   int_nettoa_longrun_data[[model]][[expt]]$NETTOA = v$swdn_toa - v$swup_toa - v$olr
	} else if ( !((model == "HadGEM2") | (model == "MPIESM11")) ){
	   file = eval(as.name(paste(model,expt,"nc",sep="_")))
	   int_nettoa_longrun_data[[model]][[expt]]$file = file
	   nc   = nc_open(file)
    	   for (var in vars){
	      #print(paste("var",var))
	      int_nettoa_longrun_data[[model]][[expt]][[var]] = ncvar_get(nc,var)
	   }
	} else {
    	   for (var in setdiff(vars, "TAX_YEARS")){ # only T2M/NETTOA files exist for these two
	      file = eval(as.name(paste(model,expt,var,"nc",sep="_")))
	      nc   = nc_open(file)
	      int_nettoa_longrun_data[[model]][[expt]][[var]] = ncvar_get(nc,var)
	   }
        } # if GFDL / not HadGEM or MPIESM
    } # expt

    # Perturbed-run forcing, read by the diffusive fit (missing = abrupt 4xCO2).
    int_nettoa_longrun_data[[model]]$co2_multiple = if (model %in% names(co2_multiple)) co2_multiple[[model]] else 4
    int_nettoa_longrun_data[[model]]$ramp_years   = if (model %in% names(ramp_years))   ramp_years[[model]]   else 0

    d = int_nettoa_longrun_data[[model]]
    print(sprintf("  piControl %d yr: mean T2M %.2f, NETTOA %.2f W/m2 | perturbed (%gx CO2, ramp %.1f yr) %d yr: NETTOA yrs 1-5 %.2f W/m2",
                  length(d$piControl$T2M), mean(d$piControl$T2M, na.rm = TRUE), mean(d$piControl$NETTOA, na.rm = TRUE),
                  d$co2_multiple, d$ramp_years, length(d[["4xCO2"]]$T2M), mean(d[["4xCO2"]]$NETTOA[1:5], na.rm = TRUE)))
} # model

setwd(homedir)
save(models,Nmodels,expts,vars,int_nettoa_longrun_data,file="./data/int_netToa_longrun.Rdata")
