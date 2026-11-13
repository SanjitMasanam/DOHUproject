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

expt_re='(1pct|piControl|picntrl)'
var_re='/(tas|rsdt|rsut|rlut)(/|$)'

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
   : > "$tmpd/hits"
   found=0
   d=1
   while [ "$d" -le "$maxdepth" ]; do
      find "$root" -mindepth "$d" -maxdepth "$d" -type d "${name_args[@]}" 2>>"$tmpd/errors" | sort > "$tmpd/hits"
      if [ -s "$tmpd/hits" ]; then found=$d; break; fi
      d=$((d + 1))
   done
   if [ "$found" -eq 0 ]; then
      echo "!! no directory matching the model patterns within $maxdepth levels (try MAXDEPTH=10 or PATTERNS=...)"
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
   while IFS= read -r dir; do
      model=$(basename "$dir")
      center=$(basename "$(dirname "$dir")")
      [ "$found" -gt 3 ] && center=-
      # In DRS layouts the experiments are children of the model dir; search only the wanted ones.
      search=()
      while IFS= read -r kid; do search+=("$dir/$kid"); done < <(ls "$dir" 2>/dev/null | grep -E "^${expt_re}")
      [ ${#search[@]} -eq 0 ] && search=("$dir")
      find "${search[@]}" -type f -name '*.nc' \
           '(' -name 'tas_*' -o -name 'rsdt_*' -o -name 'rsut_*' -o -name 'rlut_*' ')' \
           '(' -path '*1pct*' -o -path '*piControl*' -o -path '*picntrl*' ')' \
           '(' -path '*Amon*' -o -path '*/mo/*' ')' 2>>"$tmpd/errors" |
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
