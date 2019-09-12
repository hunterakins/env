import numpy as np
from matplotlib import pyplot as plt
from .json_reader import write_json, read_json
from .env_loader import Env, env_from_dict
from pyat.pyat.readwrite import read_env
import os

'''
Description:
Create a json template for the swellex environment and a builder to initialize

Author: Hunter Akins
'''


def create_json():
    mod_name = __name__
    curr_mod = __import__(mod_name)
    root = os.path.dirname(curr_mod.__file__)
    TitleEnv, freq, ssp, bdry, pos, beam, cint, RMax = read_env(root + '/s5_default.env', 'KRAKEN')
    num_layers = len(ssp.raw)
    z_ss = np.array(ssp.raw[0].z)
    rp_ss = np.array([0])
    cw = np.array(ssp.raw[0].alphaR)
    cw = cw.reshape(len(cw), 1)
    z_sb = np.array(ssp.depth[1:])
    rp_sb=np.array([0])
    rhob = np.array([[x.rho[0] for x in ssp.raw[1:]] + [ssp.raw[-1].rho[1]]]).reshape(3,1) 
    attn = np.array([[x.alphaI[0] for x in ssp.raw[1:]] + [ssp.raw[-1].alphaI[1]]]).reshape(3,1)
    cb = np.array([[x.alphaR[0] for x in ssp.raw[1:]] + [ssp.raw[-1].alphaR[1]]]).reshape(3,1)
    env_inputs = dict(  z_ss=z_ss,
                    rp_ss=rp_ss,
                    cw = cw,
                    z_sb=z_sb,
                    rp_sb=rp_sb,
                    cb=cb,
                    rhob=rhob,
                    attn=attn,
                    rbzb=np.array([[0, z_sb[0]]]))
    write_json(root + '/swellex/swellex.json', env_inputs)
    return

create_json()

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
        env_dict = read_json(root + '/swellex/swellex.json')
        for key in kwargs.keys():
            env_dict[key] = kwargs[key]
        env = env_from_dict(env_dict)
        self._instance = env
        return env
