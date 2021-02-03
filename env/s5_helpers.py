import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

"""
Description:
Helper scripts specific to the s5 event of swellex

Date: 
2/2/2021


Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""




prog_path = '/home/hunter/research/code/env/env/'

def load_s5_z():
    x = np.load(prog_path + 'swellex/s5_bathy.npy')
    t = x[0,:]
    z = x[1,:]
    return t, z

def load_s5_r():
    r = np.load(prog_path + 'swellex/s5_r.npy')
    r =r *1000 # convert to meters
    r = r[:,0]
    t = np.load(prog_path + 'swellex/s5_r_tgrid.npy')
    t *= 60 # convert to seconds
    return t, r

def s5_approach_D():
    """
    Specific to the S5 approach event, a function
    that provides a source depth as a function of source 
    range
    
    A few notes:
    -Since the source doesn't go beyond roughly 8 km, I added
    some dummy points further out
    - The ridge at around 8km means extrapolation will be ill-conditioned,
        as a result I simply truncate the bathymetry log at 8km to 
        remove the ridge feature
        Then I add a dummy point at 10km and 20 km that's just the last value repeated
    - Also add a dummy point at r=0 that's equal to the value at t_cpa
    """
    t, z = load_s5_z()
    t_r, r =  load_s5_r()
    t_cpa = 3540
    approach_inds = [i for i in range(len(t)) if t[i] < t_cpa]
    t_appr = t[approach_inds]
    z_appr = z[approach_inds]
    approach_inds = [i for i in range(len(t_r)) if t_r[i] < t_cpa]
    t_r_appr = t_r[approach_inds]
    r_appr = r[approach_inds]

    """ Now r_appr and z_appr are on the same time grid
    Add dummy points """
    """FIrst remove furthest points with ridge feature """
    r_appr = r_appr[5:]
    t_appr = t_appr[5:]
    t_r_appr = t_r_appr[5:]
    dt = t_appr[1]-t_appr[0]
    z_appr = z_appr[5:]
    r_appr = np.concatenate((r_appr, np.array([0])))
    z_appr = np.concatenate((z_appr, np.array([z_appr[-1]])))
    t_appr = np.concatenate((t_appr, np.array([t_appr[-1] + dt])))
    t_r_appr = np.concatenate((t_r_appr, np.array([t_r_appr[-1] + dt])))
   
    r_appr = np.concatenate((np.array([20000]), r_appr)) 
    z_appr = np.concatenate((np.array([z_appr[0]]), z_appr))
    d_r = interp1d(r_appr, z_appr)
    return d_r


def get_range_correction(zr, tilt_angle):
    """
    For receiving array tilted at angle tilt_angle (degrees)
    towards the source, comput the range correction for each element in zr
    A negative angle means that the top element is closer to the source than the bottom element
    (leaning forward)
    Positive means its further away (leaning back)
    Input
    zr - np array
    ASSUMED SORTED FROM SMALLEST TO LARGEST
    tilt_angle - int or float in degrees
    Output
    r_corr - np array same shape as zr
    If the array is at the origin (r=0), then r_corr gives their
    corrected locations relative to the origin
    """
    tilt_rad = np.pi / 180 * tilt_angle
    zr_rel = zr - zr[0]
    r_corr = np.tan(tilt_rad)*zr_rel
    r_corr -= np.mean(r_corr)
    return r_corr

def get_approach_tilt_func():
    """
    Return a function that takes in
    the source range, rs, and returns the tilt
    angle in degrees...
    """
    return

if __name__ == '__main__':
    s5_approach_D()
    zr =np.linspace(95, 200, 10)
    rr = np.array([0]*zr.size)
    rcorr = get_range_correction(zr, -2)
    plt.figure()
    plt.plot(rr, zr)
    plt.plot(rcorr, zr)
    rcorr = get_range_correction(zr, 2)
    plt.plot(rcorr, zr)
    plt.gca().invert_yaxis()
    plt.show()
        
            
