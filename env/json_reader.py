import numpy as np
from matplotlib import pyplot as plt
import json

'''
Description:
Read and write json files that store the environmental information.

Author: Hunter Akins
'''

def listify(env_dict):
    # cast arrays in dict to lists
    for key in env_dict.keys():
        obj = env_dict[key]
        if type(obj) is np.ndarray:
            obj = obj.tolist()
            env_dict[key] = obj
    return

def arrayify(env_dict):
    # cast lists to arrays
    for key in env_dict.keys():
        obj = env_dict[key]
        if type(obj) is list:
            obj = np.array(obj)
            env_dict[key] = obj
    return

def write_json(fname, env_dict):
    # write env_dict to a json file fname
    listify(env_dict)
    json_str = json.dumps(env_dict)
    with open(fname, 'w') as f:
        f.write(json_str)
    return

def read_json(fname):
    # read dictionary stored as json text to a dictionary
    with open(fname, 'r') as f:
        lines = f.readlines()
        json_str = lines[0]
        env_dict = json.loads(json_str)
    arrayify(env_dict)
    return env_dict


