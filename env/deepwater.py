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

bottom_depth = 5000

z1 = [0.0,  200.0,  250.0,  400.0,  600.0,  800.0, 1000.0, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0, 2200.0, 2400.0, 2600.0, 2800.0, 3000.0, 3200.0, 3400.0, 3600.0, 3800.0, 4000.0, 4200.0, 4400.0, 4600.0, 4800.0, 5000.0]

alphaR = [1548.52,1530.29,1526.69,1517.78,1509.49,1504.30,1501.38,1500.14,1500.12,1501.02,1502.57,1504.62,1507.02,1509.69,1512.55,1515.56,1518.67,1521.85,1525.10,1528.38,1531.70,1535.04,1538.39,1541.76,1545.14,1548.52,1551.91]

cw = np.zeros((len(alphaR), 1))
cw[:,0] = alphaR

cb = 1600.0
pb = 1.5
ab = 0.0 # no attenuation
rmax = 100*1e3
dr = 100
dz = 20
zmax = 2.5*bottom_depth # halfspace depth


env_inputs = dict(  z_ss=np.array(z1).tolist(),
                rp_ss=np.array([0]).tolist(),
                cw = cw.tolist(),
                z_sb=np.array([bottom_depth]).tolist(),
                rp_sb=np.array([0]).tolist(),
                cb=np.array([[cb]]).reshape(1,1).tolist(),
                rhob=np.array([[pb]]).reshape(1,1).tolist(),
                attn=np.array([[ab]]).reshape(1,1).tolist(),
                rmax=rmax,
                dr=dr,
                dz=dz,
                zmplt=zmax,
                rbzb=np.array([[0, bottom_depth]]).tolist())

mod_name = __name__
curr_mod = __import__(mod_name)
root = os.path.dirname(curr_mod.__file__)

write_json(root + '/env/deepwater/deepwater.json', env_inputs)

class DeepWaterBuilder:
    """
    Build a deep water Munk waveguide with default arguments stored in a json    
    """

    def __init__(self):
        self._instance = None

    def __call__(self, **kwargs):
        mod_name = __name__
        curr_mod = __import__(mod_name)
        root = os.path.dirname(curr_mod.__file__)
        env_dict = read_json(root + '/env/deepwater/deepwater.json')
        for key in kwargs.keys():
            env_dict[key] = kwargs[key]
        env = env_from_dict(env_dict)
        self._instance = env
        return env
                

