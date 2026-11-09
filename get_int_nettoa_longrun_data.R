library(ncdf4)
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
	   int_nettoa_longrun_data[[model]][[expt]]$file   = file.path(gfdldir, pattern)
	   int_nettoa_longrun_data[[model]][[expt]]$T2M    = read_gfdl_var(pattern, "t_ref")
	   int_nettoa_longrun_data[[model]][[expt]]$NETTOA = read_gfdl_var(pattern, "swdn_toa") -
	      read_gfdl_var(pattern, "swup_toa") - read_gfdl_var(pattern, "olr")
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
