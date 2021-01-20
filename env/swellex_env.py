import numpy as np
from matplotlib import pyplot as plt
from env.env.env_loader import Env, env_from_dict
from env.env.json_reader import write_json, read_json
from pyat.pyat.readwrite import read_env
import os

'''
Description:
Create a json template for the swellex environment and a builder to initialize

Author: Hunter Akins
'''


def create_json(root=None):
    if type(root) == type(None):
        mod_name = __name__
        curr_mod = __import__(mod_name)
        root = os.path.dirname(curr_mod.__file__)
    TitleEnv, freq, ssp, bdry, pos, beam, cint, RMax = read_env(root + '/env/s5_default.env', 'KRAKEN')
    bott_bndry = bdry.Bot
    if bott_bndry.hs.alphaR.size == 0:
        hs_speed = ssp.raw[-1].alphaR*1.5
        hs_attn = ssp.raw[-1].alphaI*1.5
        hs_rho = ssp.raw[-1].rho*1.5
    else:
        hs_speed = bott_bndry.hs.alphaR
        hs_attn =bott_bndry.hs.alphaI
        hs_rho = bott_bndry.hs.rho

    num_layers = len(ssp.raw)
    z_ss = np.array(ssp.raw[0].z)
    rp_ss = np.array([0])
    cw = np.array(ssp.raw[0].alphaR)
    cw = cw.reshape(len(cw), 1)
    bottom_depths = ssp.depth[1:] 
    z_sb = [bottom_depths[0]]
    for zb in bottom_depths[1:]:
        z_sb.append(zb)
        z_sb.append(zb)
    z_sb = np.array(z_sb)
   # z_sb = np.array(ssp.depth[1:] + [ssp.depth[-1]*2]) # add a virtual point for halfspace
    rp_sb=np.array([0])
    #rhob = np.array([[x.rho[0] for x in ssp.raw[1:]] + [ssp.raw[-1].rho[1], hs_rho]]).reshape(4,1) 
    #attn = np.array([[x.alphaI[0] for x in ssp.raw[1:]] + [ssp.raw[-1].alphaI[1], hs_attn]]).reshape(4,1)
    cb = []
    rhob = []
    attn = []
    for x in ssp.raw[1:]:
        cb.append(x.alphaR[0])
        cb.append(x.alphaR[1])
        rhob.append(x.rho[0])
        rhob.append(x.rho[1])
        attn.append(x.alphaI[0])
        attn.append(x.alphaI[1])
    cb.append(hs_speed)
    rhob.append(hs_rho)
    attn.append(hs_attn)
    cb = np.array(cb).reshape(len(cb), 1)
    rhob = np.array(rhob).reshape(len(rhob),1)
    attn = np.array(attn).reshape(len(attn),1)
    #cb = np.array([[x.alphaR[0] for x in ssp.raw[1:]] + [ssp.raw[-1].alphaR[1], hs_speed]]).reshape(4,1)
    env_inputs = dict(  z_ss=z_ss,
                    rp_ss=rp_ss,
                    cw = cw,
                    z_sb=z_sb,
                    rp_sb=rp_sb,
                    cb=cb,
                    rhob=rhob,
                    attn=attn,
                    rbzb=np.array([[0, z_sb[0]]]))
    write_json(root + '/env/swellex/swellex.json', env_inputs)
    return


class SwellexBuilder:
    """
    Build an waveguide with i605.prn sound speed and seabed parameters for SwellEx experiment
    """

    def __init__(self):
        self._instance = None

    def __call__(self, **kwargs):
        mod_name = __name__
        curr_mod = __import__(mod_name)
        root = os.path.dirname(curr_mod.__file__)
        env_dict = read_json(root + '/env/swellex/swellex.json')
        for key in kwargs.keys():
            env_dict[key] = kwargs[key]
        env = env_from_dict(env_dict, swellex=True)
        self._instance = env
        return env

if __name__ == '__main__':
    create_json(root='../')
