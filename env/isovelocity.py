import numpy as np
from matplotlib import pyplot as plt
from .json_reader import write_json, read_json
from .env_loader import Env, env_from_dict
import os

'''
Description:
Create a json template for an isovelocity profile

Author: Hunter Akins
'''

bottom_depth = 100
cw = 1500.0
cb = 1600.0
pb = 1.5
ab = .5
rmax = 10*1e3
dr = 1
dz = 2
zmax = 2.5*bottom_depth # halfspace depth

env_inputs = dict(  z_ss=np.array([0, bottom_depth]).tolist(),
                rp_ss=np.array([0]).tolist(),
                cw = np.array([cw]*2).reshape(2,1).tolist(),
                z_sb=np.array([bottom_depth, zmax]).tolist(),
                rp_sb=np.array([0]).tolist(),
                cb=np.array([[cb, cb]]).reshape(2,1).tolist(),
                rhob=np.array([[pb, pb]]).reshape(2,1).tolist(),
                attn=np.array([[ab, pb]]).reshape(2,1).tolist(),
                rmax=rmax,
                dr=dr,
                dz=dz,
                zmplt=zmax,
                rbzb=np.array([[0, bottom_depth]]).tolist())

mod_name = __name__
curr_mod = __import__(mod_name)
root = os.path.dirname(curr_mod.__file__)

write_json(root + '/env/iso/iso.json', env_inputs)

class IsoBuilder:
    """
    Build an isovelocity Pekeris waveguide with default arguments stored in a json    
    """

    def __init__(self):
        self._instance = None

    def __call__(self, **kwargs):
        mod_name = __name__
        curr_mod = __import__(mod_name)
        root = os.path.dirname(curr_mod.__file__)
        env_dict = read_json(root + '/env/iso/iso.json')
        for key in kwargs.keys():
            env_dict[key] = kwargs[key]
        env = env_from_dict(env_dict)
        self._instance = env
        return env
                

