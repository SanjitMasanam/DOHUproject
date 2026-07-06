#!/usr/bin/env python3

import sys
print("Python started", file=sys.stderr, flush=True)

try:
    import rpy2.robjects as ro
    print("rpy2 imported", file=sys.stderr, flush=True)
    
    from pathlib import Path
    rdata_file = Path("./data/int_netToa_longrun.Rdata")
    print(f"Loading R data from {rdata_file}", file=sys.stderr, flush=True)
    
    ro.r["load"](str(rdata_file))
    print("R data loaded successfully", file=sys.stderr, flush=True)
    
    data = ro.globalenv["int_nettoa_longrun_data"]
    models = list(ro.globalenv["models"])
    expts = list(ro.globalenv["expts"])
    
    print(f"Models: {models}", file=sys.stderr, flush=True)
    print(f"Expts: {expts}", file=sys.stderr, flush=True)
    print(f"Data object type: {type(data)}", file=sys.stderr, flush=True)
    
    # Test Step 2 calculation
    import numpy as np
    from scipy.optimize import curve_fit
    
    model = models[0]
    print(f"Testing with model: {model}", file=sys.stderr, flush=True)
    
    model_data = data.rx2(model)
    expt_data = model_data.rx2("4xCO2")
    
    t2m_first10 = np.array(expt_data.rx2("T2M")).ravel()[0:10]
    t2m = np.array(expt_data.rx2("T2M")).ravel()[30:151]
    nettoa = np.array(expt_data.rx2("NETTOA")).ravel()[30:151]
    
    print(f"t2m_first10 shape: {t2m_first10.shape}", file=sys.stderr, flush=True)
    print(f"t2m shape: {t2m.shape}", file=sys.stderr, flush=True)
    print(f"nettoa shape: {nettoa.shape}", file=sys.stderr, flush=True)
    
    # Simulate Step 2 operation
    T_eq = 8.0
    epsilon = 1.2
    tau_s = 200
    a_s = 0.6
    a_f = 1 - a_s
    
    def early_response(t, tau_f):
        return T_eq * a_f * (1 - np.exp(-t / tau_f))
    
    t_early = np.arange(1, 11, 1)
    mask_early = t_early < len(t2m_first10)
    t_early_valid = t_early[mask_early]
    T_early_valid = t2m_first10[mask_early]
    
    print(f"Fitting early response with {len(T_early_valid)} points", file=sys.stderr, flush=True)
    
    try:
        popt_tau_f, pcov_tau_f = curve_fit(
            early_response, 
            t_early_valid, 
            T_early_valid, 
            p0=[4.0],
            maxfev=10000
        )
        tau_f = float(popt_tau_f[0])
        print(f"tau_f fitted successfully: {tau_f}", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"Error in early_response fitting: {e}", file=sys.stderr, flush=True)
    
    print("Test completed successfully!", file=sys.stderr, flush=True)
    
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
