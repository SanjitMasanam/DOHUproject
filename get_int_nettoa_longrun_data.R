library(ncdf4)
datadir = "/archive/jpd/longrunmip"
homedir = "/home/Sanjit.Masanam/Documents/DeepOceanHeatUptakeProject"
models = c("CCSM3","CESM1","CNRMCM6","ECEARTH","ECHAM5","GISSE2R","IPSLCM5A","MIROC3","HadGEM2","MPIESM11") # exclude FAMOUS, HadCM3, MPIESM12 for now
setwd(datadir)

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
	if ( !((model == "HadGEM2") | (model == "MPIESM11")) ){
	   file = eval(as.name(paste(model,expt,"nc",sep="_")))
	   int_nettoa_longrun_data[[model]][[expt]]$file = file
	   nc   = nc_open(file)
    	   for (var in vars){
	      #print(paste("var",var))
	      int_nettoa_longrun_data[[model]][[expt]][[var]] = ncvar_get(nc,var)
	   }
	} else {
    	   for (var in vars){
	      file = eval(as.name(paste(model,expt,var,"nc",sep="_")))
	      nc   = nc_open(file)
	      int_nettoa_longrun_data[[model]][[expt]][[var]] = ncvar_get(nc,var)
	   }
        } # if not HadGEM or MPIESM
    } # expt
} # model

setwd(homedir)
save(models,Nmodels,expts,vars,int_nettoa_longrun_data,file="./data/int_netToa_longrun.Rdata")
