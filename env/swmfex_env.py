import numpy as np
from matplotlib import pyplot as plt
from swmfex21.custom_bathy import get_custom_bathy_r
from swmfex21.ctd_proc import read_ctd
from env.env.env_loader import Env

"""
Description:
SWMFEX21 Experiment moored source to moored array environment
Range independent SSP
Range dependent bathy

Date:
10/14/2021

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""

def extrap_top_pt(ctd_arr):
    """
    Just double it...(assumes constant sound speed near surface (down to ctd_arr[0,0])
    """
    z0 = ctd_arr[0,0]
    if z0 > 0.1:
        c0 = ctd_arr[0,1]
        new_row = np.array([0.1, c0]).reshape(1,2)
        ctd_arr = np.vstack((new_row, ctd_arr))
    return ctd_arr

def get_ssp_info(ctd_num, zmax):
    ctd_full_arr = read_ctd(ctd_num)
    ctd_arr = ctd_full_arr[:,:2]
    ctd_arr = extrap_top_pt(ctd_arr) 
    z_ss = ctd_arr[:,0]
    z_ss[-1] = zmax
    rp_ss = np.array([0])
    cw = ctd_arr[:,1].reshape(z_ss.size,1)
    return  z_ss, rp_ss,  cw

class SWMFEXBuilder:
    """
    Build the simple range-independent SSP, bathymetry env
    for modeling from MFSrc2 to MFNA
    """
    def __init__(self):
        self._instance = None
    def __call__(self, **kwargs):
        """ DIFFERENT CTDS GIVE DIFFERENT BATHY FILES. 
        2 and 3 were both relevant to the survey used """
        print('Warning: if using env for pyram, z_sb must be\
                 relative to lowest val in z_ss (so first point is 0')
        BATHY_CTD_NUM = 2 
        CTD_NUM = 2 

        r,z  = get_custom_bathy_r(BATHY_CTD_NUM)
        r = r*1e3 # convert to meters
        rbzb = np.zeros((r.size, 2))
        rbzb[:,0] = r
        rbzb[:,1] = z

        zmax = np.max(rbzb[:,1])

        z_ss, rp_ss, cw = get_ssp_info(CTD_NUM, zmax)

        z_sb = np.array([rbzb[0,1]]) 
        rp_sb = np.array([0.0])
        cb = np.array([[1700]])
        rhob = np.array([[1.5]])
        attn = np.array([[.5]])

        if 'cb' in kwargs.keys():
            cb = kwargs['cb']
        if 'rhob' in kwargs.keys():
            rhob = kwargs['rhob']
        if 'attn' in kwargs.keys():
            attn = kwargs['attn']
        if 'z_sb' in kwargs.keys():
            z_sb = kwargs['z_sb']
        if 'ctd_num' in kwargs.keys():
            print('updating dfault profile to use CTD ' + str(kwargs['ctd_num']))
            z_ss, rp_ss, cw = get_ssp_info(kwargs['ctd_num'],zmax)
        env = Env(z_ss, rp_ss, cw, z_sb, rp_sb, cb, rhob, attn, rbzb)
        self._instance = env
        return env



