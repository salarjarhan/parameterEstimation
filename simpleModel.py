import os
import numpy as np
import platform
import flopy.modflow as mf
import flopy.utils as fu
import shutil
import json


def calculate_model(z1_hk, z2_hk, z3_hk):
    # Z1_hk = 15  # 3<Z1_hk<15
    # Z2_hk = 15  # 3<Z2_hk<15
    # Z3_hk = 3  # 3<Z3_hk<15

    hobs = [
        [0, 20, 10, 69.52],
        [0, 40, 10, 71.44],
        [0, 60, 10, 72.99],
        [0, 80, 10, 73.86],
        [0, 20, 45, 58.73],
        [0, 40, 45, 50.57],
        [0, 60, 45, 54.31],
        [0, 80, 45, 58.06],
        [0, 20, 80, 56.31],
        [0, 40, 80, 52.32],
        [0, 60, 80, 46.35],
        [0, 80, 80, 29.01],
        [0, 20, 100, 57.24],
        [0, 40, 100, 54.24],
        [0, 60, 100, 39.48],
        [0, 80, 100, 48.47],
    ]

    model_path = os.path.join('_model')

    if os.path.exists(model_path):
        shutil.rmtree(model_path)

    modelname = 'parEstMod'

    version = 'mf2005'
    exe_name = 'mf2005'
    if platform.system() == 'Windows':
        exe_name = 'mf2005.exe'

    ml = mf.Modflow(modelname=modelname, exe_name=exe_name, version=version, model_ws=model_path)

    nlay = 1
    nrow = 90
    ncol = 120

    area_width_y = 9000
    area_width_x = 12000

    delc = area_width_x / ncol
    delr = area_width_y / nrow

    nper = 1

    top = 100
    botm = 0

    dis = mf.ModflowDis(
        ml,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
        nper=nper,
        steady=True
    )

    ibound = 1
    strt = 100
    bas = mf.ModflowBas(ml, ibound=ibound, strt=strt)

    mask_arr = np.zeros((nlay, nrow, ncol))
    mask_arr[:, :, 0] = 80
    mask_arr[:, :, -1] = 60

    ghb_spd = {0: []}
    for layer_id in range(nlay):
        for row_id in range(nrow):
            for col_id in range(ncol):
                if mask_arr[layer_id][row_id][col_id] > 0:
                    ghb_spd[0].append([layer_id, row_id, col_id, mask_arr[layer_id][row_id][col_id], 200])

    ghb = mf.ModflowGhb(ml, stress_period_data=ghb_spd)

    rch = 0.0002
    rech = {}
    rech[0] = rch
    rch = mf.ModflowRch(ml, rech=rech, nrchop=3)

    welSp = {}
    welSp[0] = [
        [0, 20, 20, -20000],
        [0, 40, 40, -20000],
        [0, 60, 60, -20000],
        [0, 80, 80, -20000],
        [0, 60, 100, -20000],
    ]

    wel = mf.ModflowWel(ml, stress_period_data=welSp)

    hk = np.zeros((nlay, nrow, ncol))
    hk[:, :, 0:40] = z1_hk
    hk[:, :, 40:80] = z2_hk
    hk[:, :, 80:120] = z3_hk
    
    lpf = mf.ModflowLpf(ml, hk=hk, layavg=0, layvka=0, sy=0.3, ipakcb=53)
   
    pcg = mf.ModflowPcg(ml, rclose=1e-1)
    oc = mf.ModflowOc(ml)

    ml.write_input()
    ml.run_model(silent=True)
    hds = fu.HeadFile(os.path.join(model_path, modelname + '.hds'))
    times = hds.get_times()

    response = []

    for hob in hobs:
        observed = hob[3]
        calculated = hds.get_data(totim=times[-1])[hob[0]][hob[1]][hob[2]]
        response.append(observed - calculated)

    return json.dumps(response)
