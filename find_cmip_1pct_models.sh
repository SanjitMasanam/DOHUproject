#!/usr/bin/env bash
# find_cmip_1pct_models.sh -- read-only discovery of the 1pctCO2 / piControl runs on the GFDL
# CMIP archives, to fill in the center/model names and path templates at the top of
# get_int_nettoa_longrun_data.R (cmip_1pctCO2 dataset).
#
# Usage:  ./find_cmip_1pct_models.sh [root ...]      (default roots: /uda/cmip3 /uda/cmip5 /uda/cmip6)
#
# Environment overrides:
#   PATTERNS   space-separated, case-insensitive model-name fragments to look for
#   MAXDEPTH   deepest directory level searched for model directories (default 7)
#   INVENTORY  where the full file list goes (default ./cmip_1pct_inventory.tsv)
#   MIN_YEARS  run length flagged as SHORT below this (default 100)
#   FOLLOW     find symlink mode: -H (default) follows symlinked roots such as /uda/cmip6 -> CMIP6;
#              -L follows every symlink (slower, needed if links sit deeper in the tree)
#   MODELS     exact model dir names Stage B searches (default: the 13 in the R cmip_runs table);
#              MODELS=all searches every Stage A match (slow on /uda)
#   EXPT_DEPTH levels below a model dir searched for the 1pct*/piControl dirs (default 4)
#
# Stage A finds the model directories by name. It searches one level at a time and stops at
#         the first level with a match, so it never walks the large trees below the models.
# Stage B lists the monthly tas/rsdt/rsut/rlut files of the 1pct* and piControl experiments
#         inside those directories, writes them to $INVENTORY, and prints one line per run.
# Nothing is written anywhere except $INVENTORY.

set -u

if [ $# -gt 0 ]; then roots=("$@"); else roots=(/uda/cmip3 /uda/cmip5 /uda/cmip6); fi
patterns=${PATTERNS:-"ccsm3 ccsm4 cesm1 cnrm-cm6 ec-earth ecearth echam5 giss-e2-r ipsl-cm5a miroc3 hadgem2 mpi-esm gfdl-cm3 gfdl-esm2m gfdl-esm2g"}
maxdepth=${MAXDEPTH:-7}
inventory=${INVENTORY:-./cmip_1pct_inventory.tsv}
min_years=${MIN_YEARS:-100}
follow=${FOLLOW:--H}
wanted=${MODELS:-"ncar_ccsm3_0 CCSM4 CNRM-CM6-1 EC-EARTH mpi_echam5 GISS-E2-R IPSL-CM5A-LR miroc3_2_medres HadGEM2-ES MPI-ESM-LR GFDL-CM3 GFDL-ESM2M GFDL-ESM2G"}
expt_depth=${EXPT_DEPTH:-4}

expt_re='(1pct|piControl|picntrl)'
var_re='/(tas|rsdt|rsut|rlut)(/|$)'

# Directories that cannot hold monthly atmosphere output (frequencies, realms, CMIP5/6 tables):
# Stage B does not descend into them.
prune_args=('(')
for p in day Eday CFday Oday AERday 3hr E3hr CF3hr 1hr E1hr subhr CFsubhr 6hr 6hrLev 6hrPlev 6hrPlevPt \
         yr Oyr Eyr Odec Oclim fx Ofx Efx IfxAnt IfxGre Omon Lmon LImon OImon SImon AERmon CFmon Emon EmonZ \
         ImonAnt ImonGre ocean seaIce land landIce ocnBgchem aerosol; do
   [ ${#prune_args[@]} -gt 1 ] && prune_args+=(-o)
   prune_args+=(-name "$p")
done
prune_args+=(')')

# find(1) arguments:  ( -iname '*p1*' -o -iname '*p2*' ... )
name_args=('(')
for p in $patterns; do
   [ ${#name_args[@]} -gt 1 ] && name_args+=(-o)
   name_args+=(-iname "*${p}*")
done
name_args+=(')')

tmpd=$(mktemp -d "${TMPDIR:-/tmp}/cmipfind.XXXXXX") || exit 1
trap 'rm -rf "$tmpd"' EXIT
: > "$tmpd/errors"

printf 'era\tcenter\tmodel\texpt\tmember\tgrid\tversion\tvar\tstart\tend\tpath\n' > "$inventory" || {
   echo "cannot write $inventory (set INVENTORY=/some/writable/path)" >&2; exit 1; }

for root in "${roots[@]}"; do
   root=${root%/}
   era=$(basename "$root")
   echo
   echo "================ $root ================"
   if [ ! -d "$root" ]; then
      echo "!! not a directory -- skipped"
      continue
   fi

   # ---- Stage A: model directories, shallowest level that has any --------------------------
   # A level whose matches are all unwanted (e.g. the center dir EC-Earth-Consortium matching
   # "ec-earth") does not stop the search: keep going until a wanted model name turns up.
   : > "$tmpd/hits"
   found=0
   d=1
   while [ "$d" -le "$maxdepth" ]; do
      find "$follow" "$root" -mindepth "$d" -maxdepth "$d" -type d "${name_args[@]}" 2>>"$tmpd/errors" | sort > "$tmpd/hits"
      if [ -s "$tmpd/hits" ]; then
         if [ "$wanted" = all ] || awk -F/ -v w=" $wanted " 'index(w, " " $NF " ") { f = 1 } END { exit !f }' "$tmpd/hits"; then
            found=$d; break
         fi
         echo "  depth $d: matches but none in MODELS ($(awk -F/ '{ print $NF }' "$tmpd/hits" | sort -u | head -5 | tr '\n' ' ')...); going deeper"
      fi
      d=$((d + 1))
   done
   if [ "$found" -eq 0 ]; then
      echo "!! no wanted model directory within $maxdepth levels (try MAXDEPTH=10, MODELS=all or PATTERNS=...)"
      continue
   fi
   echo "Stage A: $(wc -l < "$tmpd/hits" | tr -d ' ') model directories at depth $found."
   echo "  model dir name -- parent dir (= center for cmip5/6) -- number of dirs with that name:"
   awk -F/ -v deep=$((found > 3)) '{ m = $NF; if (!(m in n)) { p[m] = deep ? "-" : $(NF-1); o[++k] = m } n[m]++ }
            END { for (i = 1; i <= k; i++) printf "    %-24s %-22s %d\n", o[i], p[o[i]], n[o[i]] }' "$tmpd/hits"

   # Deep layouts (cmip3: <expt>/atm/mo/<var>/<model>) give one model dir per experiment and
   # variable: keep only the ones under a wanted experiment and variable.
   if [ "$found" -gt 3 ]; then
      grep -E "$expt_re" "$tmpd/hits" | grep -E "$var_re" > "$tmpd/hits2"
      mv "$tmpd/hits2" "$tmpd/hits"
   fi

   # ---- Stage B: the files ---------------------------------------------------------------
   before=$(wc -l < "$inventory")
   echo "Stage B (models: ${wanted}):"
   while IFS= read -r dir; do
      model=$(basename "$dir")
      center=$(basename "$(dirname "$dir")")
      [ "$found" -gt 3 ] && center=-
      if [ "$wanted" != all ] && ! printf ' %s ' $wanted | grep -qF " $model "; then continue; fi
      printf '  %s ... ' "$dir"
      t0=$(date +%s); n0=$(wc -l < "$inventory")
      # Locate the wanted experiment dirs a few levels down, then search only inside them.
      # (Deep layouts already have the experiment above the model dir.)
      search=()
      if [ "$found" -le 3 ]; then
         while IFS= read -r e; do search+=("$e"); done < <(
            find "$follow" "$dir" -mindepth 1 -maxdepth "$expt_depth" -type d \
                 '(' -name '1pct*' -o -name piControl -o -name picntrl ')' -prune -print 2>>"$tmpd/errors")
      fi
      if [ ${#search[@]} -eq 0 ]; then
         search=("$dir")
         [ "$found" -le 3 ] && printf '(no 1pct*/piControl dir within %s levels: searching the whole tree) ' "$expt_depth"
      fi
      find "$follow" "${search[@]}" \
           '(' -type d "${prune_args[@]}" -prune ')' -o \
           '(' -type f -name '*.nc' \
               '(' -name 'tas_*' -o -name 'rsdt_*' -o -name 'rsut_*' -o -name 'rlut_*' ')' \
               '(' -path '*1pct*' -o -path '*piControl*' -o -path '*picntrl*' ')' \
               '(' -path '*Amon*' -o -path '*/mon/*' -o -path '*/mo/*' ')' -print ')' 2>>"$tmpd/errors" |
      awk -v era="$era" -v model="$model" -v center="$center" '
         BEGIN { FS = "/"; OFS = "\t" }
         {
            path = $0; expt = member = grid = version = start = end = ""
            for (i = 1; i < NF; i++) {
               c = $i
               if (expt == "" && (c ~ /^1pct/ || c == "piControl" || c == "picntrl")) expt = c
               if (c ~ /^v[0-9]+$/ || c == "latest") version = c
               if (c ~ /^r[0-9]+i[0-9]+p[0-9]+(f[0-9]+)?$/ || c ~ /^run[0-9]+$/) member = c
            }
            fname = $NF; sub(/\.nc$/, "", fname)
            n = split(fname, f, "_"); var = f[1]
            if (f[2] == "Amon" && n >= 6) {          # cmip5: 6 fields, cmip6: 7 (adds grid)
               expt = f[4]; member = f[5]
               if (n >= 7) grid = f[6]
               if (f[n] ~ /^[0-9]+-[0-9]+$/) {
                  split(f[n], dd, "-"); start = substr(dd[1], 1, 6); end = substr(dd[2], 1, 6)
               }
            }
            print era, center, model, expt, member, grid, version, var, start, end, path
         }' >> "$inventory"
      echo "$(( $(wc -l < "$inventory") - n0 )) files, $(( $(date +%s) - t0 )) s"
   done < "$tmpd/hits"
   after=$(wc -l < "$inventory")
   echo "Stage B: $((after - before)) files."
   sample=$(awk -F'\t' -v era="$era" '$1 == era { print $11; exit }' "$inventory")
   [ -n "$sample" ] && echo "  sample path (check the R path_templates against it):" && echo "    $sample"
done

# ---- Summary: one line per run --------------------------------------------------------------
echo
echo "================ runs (all four variables needed, >= $min_years yr) ================"
awk -v minyr="$min_years" '
   BEGIN { FS = OFS = "\t"; nv = split("tas rsdt rsut rlut", V, " ") }
   function mi(ym) { return substr(ym, 1, length(ym) - 2) * 12 + substr(ym, length(ym) - 1) - 1 }
   function ym(m)  { return sprintf("%04d-%02d", int(m / 12), m % 12 + 1) }
   NR > 1 {
      key = $1 FS $3 FS $4 FS $5; kv = key SUBSEP $8
      if (!(key in seen)) { seen[key] = 1; order[++nk] = key; cen[key] = $2 }
      nfiles[key]++; has[kv] = 1
      if ($6 != "" && index("," grids[key] ",", "," $6 ",") == 0) grids[key] = grids[key] (grids[key] == "" ? "" : ",") $6
      if ($9 != "" && $10 != "") {
         s = mi($9); e = mi($10)
         if (!(kv in smin) || s < smin[kv]) smin[kv] = s
         if (!(kv in emax) || e > emax[kv]) emax[kv] = e
      }
   }
   END {
      print "era", "model", "expt", "member", "center", "grid", "files", "span", "years", "status"
      for (k = 1; k <= nk; k++) {
         key = order[k]; miss = ""; minm = -1; s0 = e0 = -1
         for (j = 1; j <= nv; j++) {
            kv = key SUBSEP V[j]
            if (!(kv in has)) { miss = miss (miss == "" ? "" : ",") V[j]; continue }
            if (kv in smin) {
               m = emax[kv] - smin[kv] + 1
               if (minm < 0 || m < minm) { minm = m; s0 = smin[kv]; e0 = emax[kv] }
            }
         }
         if (miss != "")      status = "MISSING:" miss
         else if (minm < 0)   status = "NO_DATES_IN_NAME(check_in_R)"
         else if (minm / 12 < minyr) status = "SHORT"
         else                 status = "OK"
         span = (minm < 0) ? "-" : ym(s0) "_to_" ym(e0)
         yrs  = (minm < 0) ? "?" : sprintf("%.0f", minm / 12)
         print key, cen[key], (grids[key] == "" ? "-" : grids[key]), nfiles[key], span, yrs, status
      }
   }' "$inventory" > "$tmpd/summary"
if command -v column > /dev/null 2>&1; then
   { head -n 1 "$tmpd/summary"; tail -n +2 "$tmpd/summary" | sort; } | column -t -s "$(printf '\t')"
else
   { head -n 1 "$tmpd/summary"; tail -n +2 "$tmpd/summary" | sort; }
fi

nerr=$(wc -l < "$tmpd/errors" | tr -d ' ')
if [ "$nerr" -gt 0 ]; then
   echo
   echo "!! find reported $nerr errors (usually unreadable directories); first few:"
   head -n 5 "$tmpd/errors" | sed 's/^/   /'
fi
echo
echo "Full file list: $inventory"
echo "The span is taken from file names (first to last file, gaps not checked); the R script checks contiguity."
